import os
import io
import copy
import time
from typing import Any, Dict, Tuple

import pandas as pd
import torch
import torch.fx as fx
import torch.nn as nn
import torch.profiler

from .node import _AffinityShim,Node

try:
    from fvcore.nn import FlopCountAnalysis
    _FVCORE_OK = True
except Exception:
    FlopCountAnalysis = None
    _FVCORE_OK = False


# ------------------------------ Memory hook ---------------------------------
MEMORY_ACCESSED_RECORDS: Dict[str, float] = {}
FLOPS_RECORDS: Dict[str, float] = {}

def memory_hook(module_or_func, input, output):
    """Approx bytes read/written for a single op/module call."""
    input_bytes = sum(t.numel() * t.element_size() for t in input if torch.is_tensor(t))
    if torch.is_tensor(output):
        output_bytes = output.numel() * output.element_size()
    elif isinstance(output, (list, tuple)):
        output_bytes = sum(t.numel() * t.element_size() for t in output if torch.is_tensor(t))
    else:
        output_bytes = 0
    return input_bytes + output_bytes


#------------------------------- Profiler -----------------------------------
class Profiler:
    """
    Full DAG profiling (CPU, CUDA, memory, FLOPs) with FX wrapping.
    Writes/updates CSV db and caches per (model_name, node_id).
    """
    def __init__(self, mode: str, profile_db_path='profiling_results.csv', log_dir='logs'):
        assert mode in ['init', 'runtime'], "Mode must be 'init' or 'runtime'"
        self.mode = mode
        self.profile_db_path = profile_db_path
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.columns = [
            'Task_ID', 'Model', 'Layer', 'Compute',
            'Self CPU (us)', 'CPU Total (us)', 'CUDA Total (us)',
            'Self CPU Mem (bytes)', 'Self CUDA Mem (bytes)',
            'Total Execution Time (us)', 'Total Memory Used (bytes)',
            'FLOPs', 'Memory Accessed (bytes)'
        ]

        if os.path.exists(self.profile_db_path):
            import pandas as pd
            self.profile_db = pd.read_csv(self.profile_db_path)
        else:
            import pandas as pd
            self.profile_db = pd.DataFrame(columns=self.columns)

        self.runtime_csv = os.path.join(self.log_dir, 'runtime_results.csv')
        if not os.path.exists(self.runtime_csv):
            import pandas as pd
            pd.DataFrame(columns=['Task_ID', 'Model', 'Layer', 'Compute', 'Execution Time (us)']).to_csv(
                self.runtime_csv, index=False
            )

        self.observation_window = 0.0
        self.profile_cache: Dict[Tuple[str, str], 'pd.DataFrame'] = {}
        self.flops_by_module_operator: Dict[str, float] = {}

    def profile_model(self,
                      model: nn.Module,
                      input_data: Any,
                      node: Node,
                      task_id: str,
                      warmup_iters: int = 3,
                      profile_iters: int = 5):

        # ------------- FLOPs via fvcore (best-effort) -------------
        self.flops_by_module_operator = {}
        if _FVCORE_OK:
            try:
                flops_analysis = FlopCountAnalysis(model, input_data)
                self.flops_by_module_operator = flops_analysis.by_module_and_operator()
            except Exception as e:
                print(f"FLOPs analysis failed: {e}")

        cache_key = (model.__class__.__name__, node.node_id)
        if cache_key in self.profile_cache:
            cached_data = self.profile_cache[cache_key].copy()
            cached_data['Task_ID'] = task_id
            self.profile_db = pd.concat([self.profile_db, cached_data], ignore_index=True)
            print(f"[Profiler] Reused cached profiling data for {model.__class__.__name__} on {node.node_id}.")
            return

        # ------------- Device / affinity setup -------------
        original_affinity = _AffinityShim.get_process_affinity()

        try:
            # Pin to requested CPUs (if any)
            if getattr(node, "cpus", None):
                _AffinityShim.set_thread_affinity(set(int(c) for c in node.cpus))

            # Select device
            if getattr(node, "gpu", None) is not None and torch.cuda.is_available():
                torch.cuda.set_device(node.gpu)
                device = torch.device(f"cuda:{node.gpu}")
            else:
                device = torch.device("cpu")

            # Clone & instrument model
            model_copy = self._clone_model_safely(model)
            instrumented_model = self._trace_and_instrument_model(model_copy)
            instrumented_model.to(device)
            instrumented_model.eval()

            with torch.no_grad():
                for _ in range(warmup_iters):
                    _ = instrumented_model(input_data.to(device))

            print(f"[Profiler] Starting profiling for Task '{task_id}' on {node.node_id} (device={device}).")

            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=profile_iters),
                on_trace_ready=lambda prof: self._trace_handler(
                    prof, task_id, model.__class__.__name__, node.node_id, group_depth=100
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                for _ in range(profile_iters):
                    with torch.no_grad():
                        _ = instrumented_model(input_data.to(device))
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    prof.step()

            time.sleep(0.0001)

            new_rows = self.profile_db[self.profile_db['Task_ID'] == task_id].copy()
            self.profile_cache[cache_key] = new_rows
            self.profile_db.to_csv(self.profile_db_path, index=False)
            print(f"[Profiler] Profiling complete. Data saved to {self.profile_db_path}.")

        finally:
            # Unpin to original process-allowed set
            _AffinityShim.set_thread_affinity(original_affinity)
            # Optional: sync if we used a GPU
            if getattr(node, "gpu", None) is not None and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize(node.gpu)
                except Exception:
                    pass

    def _clone_model_safely(self, model: nn.Module) -> nn.Module:
        try:
            return copy.deepcopy(model)
        except Exception as e:
            print(f"[Profiler] deepcopy failed: {e}. Falling back to torch.save/load.")
            buffer = io.BytesIO()
            torch.save(model, buffer)
            buffer.seek(0)
            return torch.load(buffer)

    def _trace_and_instrument_model(self, model: nn.Module) -> fx.GraphModule:
        tracer = fx.Tracer()
        graph = tracer.trace(model)
        graph_module = fx.GraphModule(model, graph)

        profiler_attr_prefix = "_profiler_wrapped_"
        for node in list(graph.nodes):
            if node.op == 'placeholder':
                continue

            if node.op == 'call_module':
                submodule = dict(model.named_modules())[node.target]
                target_key = node.target                           # e.g., "layer1.0.bn1"
                profile_key = target_key.replace('.', '_')         # used as profiler key
                original_forward = submodule.forward

                def make_wrapped_forward(orig_fwd, profile_key, submodule, target_key):
                    def wrapped_forward(*args, **kwargs):
                        with torch.profiler.record_function(profile_key):
                            result = orig_fwd(*args, **kwargs)
                        mem_bytes = memory_hook(submodule, args, result)
                        MEMORY_ACCESSED_RECORDS[profile_key] = mem_bytes
                        submodule._last_memory_accessed = mem_bytes
                        flops_val = self.flops_by_module_operator.get(target_key, 0.0)
                        # If fvcore returns dict per-op, sum it
                        if isinstance(flops_val, dict):
                            flops_val = float(sum(flops_val.values()))
                        FLOPS_RECORDS[profile_key] = float(flops_val)
                        return result
                    return wrapped_forward

                wrapped_forward = make_wrapped_forward(original_forward, profile_key, submodule, target_key)
                setattr(submodule, f"{profiler_attr_prefix}{profile_key}", wrapped_forward)
                submodule.forward = getattr(submodule, f"{profiler_attr_prefix}{profile_key}")

            elif node.op == 'call_function':
                profile_key = node.name
                func = node.target

                def make_wrapped_func(original_func, profile_key):
                    def wrapped(*args, **kwargs):
                        with torch.profiler.record_function(profile_key):
                            result = original_func(*args, **kwargs)
                        mem_bytes = memory_hook(original_func, args, result)
                        MEMORY_ACCESSED_RECORDS[profile_key] = mem_bytes
                        FLOPS_RECORDS[profile_key] = 0.0
                        return result
                    return wrapped

                wrapped_func = make_wrapped_func(func, profile_key)
                setattr(model, f"{profiler_attr_prefix}{profile_key}", wrapped_func)
                node.target = getattr(model, f"{profiler_attr_prefix}{profile_key}")

            elif node.op == 'call_method':
                profile_key = node.name
                method_name = node.target
                obj = node.args[0]
                original_method = getattr(obj, method_name, None)
                if original_method is None:
                    continue

                def make_wrapped_method(orig_meth, profile_key, obj):
                    def wrapped_method(*args, **kwargs):
                        with torch.profiler.record_function(profile_key):
                            result = orig_meth(*args, **kwargs)
                        mem_bytes = memory_hook(obj, args, result)
                        MEMORY_ACCESSED_RECORDS[profile_key] = mem_bytes
                        setattr(obj, '_last_memory_accessed', mem_bytes)
                        FLOPS_RECORDS[profile_key] = 0.0
                        return result
                    return wrapped_method

                setattr(obj, f"{profiler_attr_prefix}{profile_key}", make_wrapped_method(original_method, profile_key, obj))

        graph_module.recompile()
        return graph_module

    def _trace_handler(self, prof, task_id: str, model_name: str, node_id: str, group_depth: int):
        key_avgs = prof.key_averages(group_by_stack_n=group_depth)
        self._process_profiler_data(key_avgs, task_id, model_name, node_id)

    def _process_profiler_data(self, key_avgs, task_id: str, model_name: str, node_id: str):
        import pandas as pd

        aggregated: Dict[str, Dict[str, Any]] = {}
        forward_pass = {
            'Task_ID': task_id, 'Model': model_name, 'Layer': 'forward_pass', 'Compute': node_id,
            'Self CPU (us)': 0.0, 'CPU Total (us)': 0.0, 'CUDA Total (us)': 0.0,
            'Self CPU Mem (bytes)': 0, 'Self CUDA Mem (bytes)': 0,
            'Total Execution Time (us)': 0.0, 'Total Memory Used (bytes)': 0,
            'FLOPs': 0.0, 'Memory Accessed (bytes)': 0.0
        }

        for evt in key_avgs:
            layer_key = evt.key
            # Skip raw aten operators (keep only our wrapped keys)
            if layer_key.startswith("aten::"):
                continue

            # Accumulate forward summary
            forward_pass['Self CPU (us)'] += evt.self_cpu_time_total
            forward_pass['CPU Total (us)'] += evt.cpu_time_total
            forward_pass['CUDA Total (us)'] += getattr(evt, 'cuda_time_total', 0.0)
            forward_pass['Self CPU Mem (bytes)'] += getattr(evt, 'self_cpu_memory_usage', 0) or 0
            forward_pass['Self CUDA Mem (bytes)'] += getattr(evt, 'self_cuda_memory_usage', 0) or 0
            forward_pass['Total Execution Time (us)'] += evt.cpu_time_total + getattr(evt, 'cuda_time_total', 0.0)
            forward_pass['Total Memory Used (bytes)'] += (getattr(evt, 'self_cpu_memory_usage', 0) or 0) + (getattr(evt, 'self_cuda_memory_usage', 0) or 0)

            flops_val = FLOPS_RECORDS.get(layer_key, 0.0)
            forward_pass['FLOPs'] += float(flops_val)
            forward_pass['Memory Accessed (bytes)'] += float(MEMORY_ACCESSED_RECORDS.get(layer_key, 0.0))

            if layer_key not in aggregated:
                aggregated[layer_key] = {
                    'Task_ID': task_id, 'Model': model_name, 'Layer': layer_key, 'Compute': node_id,
                    'Self CPU (us)': 0.0, 'CPU Total (us)': 0.0, 'CUDA Total (us)': 0.0,
                    'Self CPU Mem (bytes)': 0, 'Self CUDA Mem (bytes)': 0,
                    'Total Execution Time (us)': 0.0, 'Total Memory Used (bytes)': 0,
                    'FLOPs': 0.0, 'Memory Accessed (bytes)': 0.0
                }

            aggregated[layer_key]['Self CPU (us)'] += evt.self_cpu_time_total
            aggregated[layer_key]['CPU Total (us)'] += evt.cpu_time_total
            aggregated[layer_key]['CUDA Total (us)'] += getattr(evt, 'cuda_time_total', 0.0)
            aggregated[layer_key]['Self CPU Mem (bytes)'] += getattr(evt, 'self_cpu_memory_usage', 0) or 0
            aggregated[layer_key]['Self CUDA Mem (bytes)'] += getattr(evt, 'self_cuda_memory_usage', 0) or 0
            aggregated[layer_key]['Total Execution Time (us)'] += evt.cpu_time_total + getattr(evt, 'cuda_time_total', 0.0)
            aggregated[layer_key]['Total Memory Used (bytes)'] += (getattr(evt, 'self_cpu_memory_usage', 0) or 0) + (getattr(evt, 'self_cuda_memory_usage', 0) or 0)
            aggregated[layer_key]['FLOPs'] = float(flops_val)
            aggregated[layer_key]['Memory Accessed (bytes)'] = float(MEMORY_ACCESSED_RECORDS.get(layer_key, 0.0))

        self.profile_db = self._upsert(self.profile_db, forward_pass)
        for data in aggregated.values():
            self.profile_db = self._upsert(self.profile_db, data)
        self.profile_db.to_csv(self.profile_db_path, index=False)

    def _upsert(self, df, row: Dict[str, Any]):
        import pandas as pd
        if df.empty:
            return pd.DataFrame([row], columns=self.columns)

        mask = (
            (df['Task_ID'] == row['Task_ID']) &
            (df['Model'] == row['Model']) &
            (df['Layer'] == row['Layer']) &
            (df['Compute'] == row['Compute'])
        )
        if mask.any():
            existing_time = df.loc[mask, 'Total Execution Time (us)'].max()
            if row['Total Execution Time (us)'] > existing_time:
                for key in self.columns:
                    df.loc[mask, key] = row[key]
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        return df

    def get_profile_db(self):
        return self.profile_db

    def print_profile_db(self):
        if self.profile_db.empty:
            print("ProfileDB is empty.")
        else:
            print("ProfileDB:")
            print(self.profile_db.to_string(index=False))