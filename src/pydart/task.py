from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx
import os
import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx import Node as FxNode

from .metrics import HPCMakespanMetric
from .node import Node
from .profiler import Profiler
from .scheduler import MinMinEFTScheduler
from .utils import _resolve_attr_path, move_tensor_to_device, resolve_arg

##############################################################################
# Task Class with DP-Based Partitioning (Minimizing Makespan)
##############################################################################
class Task:
    def __init__(self, task_id: str, model: nn.Module, input_data: torch.Tensor, model_name: str,
                 profiler: Profiler, load_metric: Optional[HPCMakespanMetric] = None):
        self.task_id = task_id
        self.model = model
        self.input_data = input_data
        self.model_name = model_name
        self.profiler = profiler
        self.load_metric = load_metric if load_metric else HPCMakespanMetric()
        self.stages: Dict[str, Stage] = {}
        self.graph = nx.DiGraph()
        self.start_time: Optional[float] = None
        self.finish_time: Optional[float] = None
        self.output_data: Optional[torch.Tensor] = None
        self.busy_time: float = 0.0
        self.computation_time: float = 0.0
        self.transfer_time: float = 0.0
        self.prof_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._available_nodes: List[Node] = []
        self.init_traced_graph: List[str] = []
        self.placeholder_names: Set[str] = set()
        self._initialize_dag()

    def _initialize_dag(self):
        tracer = fx.symbolic_trace(self.model)
        fx_nodes = list(tracer.graph.nodes)
        self.init_traced_graph = [node.name for node in fx_nodes]
        self.placeholder_names = set(n.name for n in fx_nodes if n.op == "placeholder")

    def get_forward_pass_time(self, sum_across_compute: bool = False) -> float:
        if not self.profiler:
            return 0.0
        df = self.profiler.get_profile_db()
        mask = (df['Task_ID'] == self.task_id) & (df['Model'] == self.model_name) & (df['Layer'] == 'forward_pass')
        matched = df.loc[mask]
        if matched.empty:
            return 0.0
        times = matched['Total Execution Time (us)']
        return float(times.sum()) if sum_across_compute else float(times.max())

    def populate_profile_records(self):
        if not self.profiler:
            return
        df = self.profiler.get_profile_db()
        mask_all = (df['Task_ID'] == self.task_id) & (df['Model'] == self.model_name)
        relevant = df.loc[mask_all]
        all_computes = relevant['Compute'].unique()
        for comp in all_computes:
            mask_comp = (relevant['Compute'] == comp)
            subdf = relevant.loc[mask_comp]
            for layer in self.init_traced_graph:
                row = subdf.loc[subdf['Layer'] == layer]
                if not row.empty:
                    self.prof_records[(comp, layer)] = row.iloc[0].to_dict()
                else:
                    self.prof_records[(comp, layer)] = None
            
    def run_offline_partition_makespan(self):
        """
        DP to partition layers into K blocks (K = number of available nodes).
        This method now decides only the stage boundaries.
        It stores per-node runtime for each stage in Stage._per_node_runtime,
        and leaves Stage.assigned_node = None for the Scheduler to fill.
        """
        print(self.load_metric)
        tracer = fx.symbolic_trace(self.model)
        fx_nodes = list(tracer.graph.nodes)
        L = len(fx_nodes)
        K = len(self._available_nodes)
        if L == 0 or K == 0:
            return

        # Avoid empty blocks if fewer layers than nodes
        K = max(1, min(K, L))

        # Build DP cost table using the active load_metric (your 1/AI)
        cost = [[0.0 for _ in range(L)] for _ in range(L)]
        for i in range(L):
            running = [0.0] * len(self._available_nodes)
            for j in range(i, L):
                # min across nodes for this layer per your metric
                best_here = float("inf")
                for n_idx, node in enumerate(self._available_nodes):
                    running[n_idx] += self.load_metric.compute_layer(self, node, fx_nodes[j])
                    best_here = min(best_here, running[n_idx])
                cost[i][j] = best_here

        # Standard linear partitioning DP (minimize max block cost)
        dp = [[float("inf")] * (K + 1) for _ in range(L)]
        split = [[-1] * (K + 1) for _ in range(L)]
        for i in range(L):
            dp[i][1] = cost[0][i]
        for k in range(2, K + 1):
            for i in range(L):
                for x in range(0, i):
                    cand = max(dp[x][k - 1], cost[x + 1][i])
                    if cand < dp[i][k]:
                        dp[i][k] = cand
                        split[i][k] = x

        # Backtrack to get K contiguous blocks
        partitions = []
        k_val, i_val = K, L - 1
        while k_val > 1:
            x = split[i_val][k_val]
            partitions.append((x + 1, i_val))
            i_val = x
            k_val -= 1
        partitions.append((0, i_val))
        partitions.reverse()

        # Build Stage objects (no node bound yet), but store per-node predicted time using TIME metric
        time_metric = HPCMakespanMetric()
        self.stages.clear()
        self.graph.clear()

        for idx, (start, end) in enumerate(partitions):
            stage_id = f"{self.task_id}-stage-{idx + 1}"
            stg = Stage(stage_id, fx_nodes[start:end + 1], assigned_node=None, task=self)

            # Fill per-node runtime (sum of per-layer times) for this block
            per_node = {}
            for node in self._available_nodes:
                s = 0.0
                for l in range(start, end + 1):
                    s += time_metric.compute_layer(self, node, fx_nodes[l])
                per_node[node.node_id] = float(s)
            stg._per_node_runtime = per_node
            self.stages[stage_id] = stg
            self.graph.add_node(stage_id, stage=stg)

        # Link stages sequentially
        stage_ids = sorted(self.stages.keys(), key=lambda sid: int(sid.split("-")[-1]))
        for i in range(len(stage_ids) - 1):
            self.stages[stage_ids[i + 1]].add_dependency(stage_ids[i])
            self.stages[stage_ids[i]].add_dependent(stage_ids[i + 1])
            self.graph.add_edge(stage_ids[i], stage_ids[i + 1])


    def get_execution_order(self) -> List[str]:
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Cycle in stage DAG")

    def update_busy_time(self, exec_time: float, transfer_time: float = 0.0):
        self.busy_time += exec_time
        self.transfer_time += transfer_time
        self.computation_time += (exec_time - transfer_time)

    def set_output_data(self, output: torch.Tensor):
        self.output_data = output.cpu() if output is not None else None
        self.finish_time = time.perf_counter()

    def get_total_execution_time(self) -> float:
        if self.start_time and self.finish_time:
            return self.finish_time - self.start_time
        return 0.0

    def print_stage_allocations(self):
        print(f"=== Stage Allocations for Task {self.task_id} ===")
        for sid, stg in self.stages.items():
            layer_names = [fxn.name for fxn in stg.nodes]
            node_id = stg.assigned_node.node_id if stg.assigned_node else "Unassigned"
            print(f"{sid}: Node={node_id}, Layers={layer_names}, Deps={stg.dependencies}")

    def __repr__(self):
        return f"Task({self.task_id}, model={self.model_name})"
    
##############################################################################
# Stage Class
##############################################################################
class Stage:
    def __init__(self, stage_id: str, nodes: List[FxNode], assigned_node: Node, task: Task):
        self.stage_id = stage_id
        self.nodes = nodes
        self.assigned_node = assigned_node
        self.dependencies: List[str] = []
        self.dependents: List[str] = []
        self.execution_time: Optional[float] = None
        self.transfer_time: float = 0.0
        self.output_data: Optional[torch.Tensor] = None
        self.task = task
        self.stage_device: str = "cpu"

        self._per_node_runtime: Dict[str, float] = {}
        self._per_node_transfer: Dict[str, float] = {}

    def add_dependency(self, stage_id: str):
        self.dependencies.append(stage_id)

    def add_dependent(self, stage_id: str):
        self.dependents.append(stage_id)

    def run_stage(self, node_outputs: Dict[str, torch.Tensor]):
        start_time = time.perf_counter()
        transfer_time = 0.0

        if self.assigned_node and self.assigned_node.gpu is not None and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.assigned_node.gpu}")
            self.stage_device = str(device)
        else:
            device = torch.device("cpu")
            self.stage_device = "cpu"

        try:
            with torch.no_grad():
                for fx_node in self.nodes:
                    resolved_args = resolve_arg(fx_node.args, node_outputs)
                    resolved_kwargs = resolve_arg(fx_node.kwargs, node_outputs)

                    # Move tensors for args/kwargs to the stage device
                    t_start = time.perf_counter()
                    resolved_args = move_tensor_to_device(resolved_args, device)
                    resolved_kwargs = move_tensor_to_device(resolved_kwargs, device)
                    t_end = time.perf_counter()
                    transfer_time += (t_end - t_start)

                    if fx_node.op == "placeholder":
                        out = self.task.input_data.to(device, non_blocking=True)

                    elif fx_node.op == "get_attr":
                        # --- CHANGED: robust dotted-path resolver + ensure Tensor on correct device ---
                        out = _resolve_attr_path(self.task.model, fx_node.target)
                        if isinstance(out, torch.Tensor) and out.device != device:
                            out = out.to(device, non_blocking=True)
                        # --- END CHANGED ---

                    elif fx_node.op == "call_module":
                        submodule = self.task.model.get_submodule(fx_node.target)
                        submodule.to(device)
                        out = submodule(*resolved_args, **resolved_kwargs)

                    elif fx_node.op == "call_function":
                        out = fx_node.target(*resolved_args, **resolved_kwargs)

                    elif fx_node.op == "call_method":
                        method = getattr(resolved_args[0], fx_node.target)
                        out = method(*resolved_args[1:], **resolved_kwargs)

                    elif fx_node.op == "output":
                        # --- CHANGED: resolve to the actual tensor referenced by the output node ---
                        out = resolve_arg(fx_node.args[0], node_outputs)
                        # --- END CHANGED ---

                    else:
                        raise NotImplementedError(f"Operation '{fx_node.op}' is not supported.")

                    node_outputs[fx_node.name] = out

        except Exception as e:
            print(f"[Stage] {self.stage_id} error: {e}")
            self.execution_time = float('inf')
            self.transfer_time = float('inf')
            node_outputs[self.stage_id] = None
            return

        finally:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            end_time = time.perf_counter()
            self.execution_time = end_time - start_time
            self.transfer_time = transfer_time

        self.task.update_busy_time(self.execution_time, self.transfer_time)

        # --- CHANGED: bullet-proof final output capture using the FX 'output' node if present ---
        if not self.dependents:
            out_node = next((n for n in self.nodes if n.op == 'output'), None)
            if out_node is not None:
                final_tensor = resolve_arg(out_node.args[0], node_outputs)
            else:
                # fallback to last produced node in this stage
                final_tensor = node_outputs.get(self.nodes[-1].name, None)

            if isinstance(final_tensor, torch.Tensor):
                self.task.set_output_data(final_tensor)
            else:
                self.task.set_output_data(None)
        # --- END CHANGED ---

        node_label = self.assigned_node.node_id if self.assigned_node else "None"
        print(f"[Stage] {self.stage_id}: Executed on {node_label} in {self.execution_time:.4f} s, Transfer: {self.transfer_time:.4f} s.")

    def __repr__(self):
        return (f"Stage({self.stage_id}, device={self.stage_device}, "
                f"node={self.assigned_node.node_id if self.assigned_node else 'None'}, "
                f"deps={self.dependencies}, exec_time={self.execution_time}, "
                f"transfer_time={self.transfer_time})")
    


##############################################################################
# Taskset Class (Using DP-Based Partitioning for Makespan)
##############################################################################
class Taskset:
    def __init__(self, tasks: List[Task], available_nodes: List[Node],
                 metric: Optional[HPCMakespanMetric] = None):
        self.tasks = tasks
        self.available_nodes = sorted(available_nodes, key=lambda n: 0 if n.gpu is not None else 1)
        # self.metric = metric if metric else HPCMakespanMetric()
        # print(self.metric)
        # if self.tasks and self.available_nodes:
        #     sample_tensor = self.tasks[0].input_data
        #     penalty = measure_max_transfer_penalty(self.available_nodes, sample_tensor)
        #     print(f"Measured max transfer penalty: {penalty:.6f} s")
        #     self.metric.transfer_penalty = penalty
        self.total_utilization = 0.0
        self.average_turnaround_time = 0.0
        self.throughput = 0.0
        self.makespan = 0.0
        self.task_completion_rate = 0.0
        self.loads: Dict[str, float] = {}
        for t in self.tasks:
            # t.load_metric = self.metric
            self.loads[t.task_id] = t.load_metric.compute_task(t, self)
            t._available_nodes = self.available_nodes
            t.run_offline_partition_makespan()

        # sched = Scheduler(self.available_nodes)
        # sched.schedule(self.tasks)
        # NEW: deterministic Min-Min (EFT)
        sched = MinMinEFTScheduler(self.available_nodes, gpu_gate_pct=None, debug=False)
        sched.schedule(self.tasks)

        # printing the schedule
        # for t in self.tasks:
        #   print(f"[Schedule] {t.task_id}")
        #   for sid, stg in sorted(t.stages.items(), key=lambda kv: kv[0]):
        #       best = stg.assigned_node.node_id if stg.assigned_node else "None"
        #       est = stg._per_node_runtime.get(best, float('nan'))
        #       print(f"  {sid}: -> {best} (est {est:.2f} us)")
        for t in self.tasks:
            print(f"[Schedule] {t.task_id}")
            for sid, stg in sorted(t.stages.items(), key=lambda kv: kv[0]):
                best = stg.assigned_node.node_id if stg.assigned_node else "None"
                est_us = stg._per_node_runtime.get(best, float("inf"))
                if not (est_us > 0 and est_us != float("inf")):
                    others = [v for v in stg._per_node_runtime.values() if v > 0 and v != float("inf")]
                    est_us = min(others) if others else float("nan")
                print(f"  {sid}: -> {best} (pred run {est_us:.2f} µs)")


    def execute_all(self):
        threads = []
        for t in self.tasks:
            thr = threading.Thread(target=self._execute_task, args=(t,))
            thr.start()
            threads.append(thr)
        for thr in threads:
            thr.join()
        self._calculate_metrics()
    def _execute_task(self, task: Task):
        print(f"[Taskset] Starting Task {task.task_id}")
        task.start_time = time.perf_counter()
        try:
            order = task.get_execution_order()
        except ValueError as e:
            print(f"[Taskset] {task.task_id} error: {e}")
            return
        node_outputs = {}
        for sid in order:
            stg = task.stages[sid]
            def run_stage(s=stg):
                s.run_stage(node_outputs)
            rq = stg.assigned_node.assign_task(run_stage)
            rq.get()
        task.finish_time = time.perf_counter()
        print(f"[Taskset] Completed Task {task.task_id} in {task.finish_time - task.start_time:.2f} s")

    def _calculate_metrics(self):
        total_busy_time = sum(stg.execution_time for t in self.tasks for stg in t.stages.values() if stg.execution_time)
        earliest_start = min((t.start_time for t in self.tasks if t.start_time), default=None)
        latest_finish = max((t.finish_time for t in self.tasks if t.finish_time), default=None)
        obs = (latest_finish - earliest_start) if earliest_start and latest_finish else 0.0
        used_nodes = {stg.assigned_node.node_id for t in self.tasks for stg in t.stages.values() if stg.assigned_node}
        total_available_time = obs * len(used_nodes)
        self.total_utilization = total_busy_time / total_available_time if total_available_time > 0 else 0.0
        ttimes = [t.get_total_execution_time() for t in self.tasks if t.start_time and t.finish_time]
        self.average_turnaround_time = sum(ttimes) / len(ttimes) if ttimes else 0.0
        self.makespan = obs
        self.throughput = len(self.tasks) / obs if obs > 0 else 0.0
        done = [t for t in self.tasks if t.output_data is not None]
        self.task_completion_rate = len(done) / len(self.tasks) if self.tasks else 0.0
    def __repr__(self):
        return (f"Taskset(num_tasks={len(self.tasks)}, makespan={self.makespan:.4f}s, "
                f"utilization={self.total_utilization:.2%}, throughput={self.throughput:.3f} tasks/s, "
                f"avg_turnaround={self.average_turnaround_time:.4f}s, completion_rate={self.task_completion_rate:.2%})")
    



##############################################################################
# Evaluator Class
##############################################################################
class Evaluator:
    def __init__(self, taskset: Taskset, profiler: Profiler):
        self.taskset = taskset
        self.profiler = profiler
        self.naive_outputs: Dict[str, torch.Tensor] = {}
        self.parallel_outputs: Dict[str, torch.Tensor] = {}
        self.naive_execution_times: Dict[str, float] = {}
        self.parallel_execution_times: Dict[str, float] = {}
        self.naive_completion_times: Dict[str, float] = {}
        self.parallel_completion_times: Dict[str, float] = {}
        self.naive_makespan: float = 0.0
        self.parallel_makespan: float = 0.0
        self.speedup_makespan: float = 0.0
        self.throughput_makespan: float = 0.0

    def run_baseline_execution(self, mode: str = "sequential"):
        if mode == "sequential":
            return self.run_naive_execution()
        if mode == "async":
            return self.run_async_naive_execution()
        raise ValueError(f"Unknown baseline mode: {mode}")

    def run_naive_execution(self):
        print("[Evaluator] Starting Naive Execution.")
        # print("[Evaluator] Starting Naive Execution.")
        self.naive_outputs.clear()
        self.naive_execution_times.clear()
        self.naive_completion_times.clear()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
          print("Running on cuda")
        else:
          print("Running on cpu")
        start = time.perf_counter()
        for task in self.taskset.tasks:
            task.model.to(device)
            inp = task.input_data.to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = task.model(inp)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            et = t1 - t0
            self.naive_execution_times[task.task_id] = et
            self.naive_completion_times[task.task_id] = time.perf_counter() - start
            self.naive_outputs[task.task_id] = out.cpu()
            print(f"[Evaluator] Task {task.task_id}: Naive exec time: {et:.4f}s")
        self.naive_makespan = time.perf_counter() - start
        print(f"[Evaluator] Naive makespan: {self.naive_makespan:.4f}s\n")

    def run_async_naive_execution(self):
        print("[Evaluator] Starting Async Naive Execution.")
        self.naive_outputs.clear()
        self.naive_execution_times.clear()
        self.naive_completion_times.clear()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print("Running on cuda")
        else:
            print("Running on cpu")

        start = time.perf_counter()

        def execute_task(task):
            task.model.to(device)
            inp = task.input_data.to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                out = task.model(inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            exec_time = t1 - t0
            completion_time = time.perf_counter() - start
            return task.task_id, exec_time, completion_time, out.cpu()
        
        
        task_count = len(self.taskset.tasks)
        cpu_count = os.cpu_count() or 1
        max_workers = max(1, min(task_count, cpu_count - 1 if cpu_count > 1 else 1))
        print(f"No of Workers launching are: {max_workers}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(execute_task, task) for task in self.taskset.tasks]

            for future in as_completed(futures):
                task_id, exec_time, completion_time, output = future.result()
                self.naive_execution_times[task_id] = exec_time
                self.naive_completion_times[task_id] = completion_time
                self.naive_outputs[task_id] = output
                print(f"[Evaluator] Task {task_id}: Async naive exec time: {exec_time:.4f}s")

        self.naive_makespan = time.perf_counter() - start
        print(f"[Evaluator] Async naive makespan: {self.naive_makespan:.4f}s\n")

    def run_parallel_execution(self):
        print("[Evaluator] Starting Parallel Execution.")
        self.parallel_outputs.clear()
        self.parallel_execution_times.clear()
        self.parallel_completion_times.clear()
        par_start = time.perf_counter()
        self.taskset.execute_all()
        par_end = time.perf_counter()
        self.parallel_makespan = par_end - par_start
        for task in self.taskset.tasks:
            self.parallel_outputs[task.task_id] = task.output_data.cpu() if task.output_data is not None else None
            self.parallel_execution_times[task.task_id] = task.get_total_execution_time()
            if task.finish_time:
                self.parallel_completion_times[task.task_id] = task.finish_time - par_start
            else:
                self.parallel_completion_times[task.task_id] = float('nan')

        print(f"[Evaluator] Parallel makespan: {self.parallel_makespan:.4f}s\n")

    def compare_outputs(self):
        print("[Evaluator] Comparing Outputs.")
        all_match = True
        for tid, naive_out in self.naive_outputs.items():
            par_out = self.parallel_outputs.get(tid)
            if naive_out is None or par_out is None:
                print(f"[Evaluator] Task {tid} missing output.")
                all_match = False
                continue
            if torch.equal(naive_out, par_out) or torch.allclose(naive_out, par_out, rtol=1e-3, atol=1e-4):
                print(f"[Evaluator] Task {tid}: Outputs match.")
            else:
                print(f"[Evaluator] Task {tid}: Outputs do NOT match.")
                all_match = False
        if all_match:
            print("[Evaluator] All outputs match.\n")
        else:
            print("[Evaluator] Some outputs differ.\n")

    def analyze_speedup_throughput(self):
        print("[Evaluator] Analyzing Speedup and Throughput.\n")
        total_naive = sum(self.naive_execution_times.values())
        total_parallel = sum(self.parallel_execution_times.values())
        
        # print("--- Sum-of-times ---")
        # print(f"Naive total: {total_naive:.4f}s, Parallel total: {total_parallel:.4f}s")
        
        # speedup_sum = total_naive / total_parallel if total_parallel > 0 else float('inf')
        n = len(self.taskset.tasks)

        # print(f"Speedup (sum-of-times): {speedup_sum:.2f}x")
        # print(f"Naive Throughput: {n / total_naive:.2f} tasks/s, Parallel Throughput: {n / total_parallel:.2f} tasks/s\n")

        print("--- Makespan ---")
        print(f"Naive makespan: {self.naive_makespan:.4f}s, Parallel makespan: {self.parallel_makespan:.4f}s")

        self.speedup_makespan = self.naive_makespan / self.parallel_makespan if self.parallel_makespan > 0 else float('inf')
        self.throughput_makespan = n / self.parallel_makespan if self.parallel_makespan > 0 else 0.0
        print(f"Speedup (makespan): {self.speedup_makespan:.2f}x")
        print(f"Naive Throughput (makespan): {n / self.naive_makespan:.2f} tasks/s, Parallel Throughput (makespan): {self.throughput_makespan:.2f} tasks/s\n")
        print("--- Task Completion Times ---")
        for tid in self.naive_completion_times:
            nf = self.naive_completion_times[tid]
            pf = self.parallel_completion_times.get(tid, float('nan'))
            print(f"Task {tid}: Naive finish: {nf:.4f}s, Parallel finish: {pf:.4f}s")
        print()

    def __repr__(self):
        return f"Evaluator(Taskset with {len(self.taskset.tasks)} tasks)"