from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional
import torch
from .node import Node

if TYPE_CHECKING:
    from .task import Stage, Task

class Scheduler:
    def __init__(self, nodes: List[Node], comm_penalty_s: float = 0.001, bias_idle: float = 5e-4):
        """
        nodes: available compute nodes
        comm_penalty_s: small penalty if a stage moves to a different node than its predecessor
        bias_idle: small additive bias favoring less-busy nodes (reduces GPU overloading)
        """
        self.nodes = nodes
        self.node_available: Dict[str, float] = {n.node_id: 0.0 for n in nodes}
        self.comm_penalty_s = float(comm_penalty_s)
        self.bias_idle = float(bias_idle)

    def _penalty(self, prev_node_id: Optional[str], node_id: str) -> float:
        if prev_node_id is None or prev_node_id == node_id:
            return 0.0
        return self.comm_penalty_s

    def schedule(self, tasks: List["Task"]):
        # indegree per stage; initial ready set across ALL tasks
        import heapq
        indeg: Dict[str, int] = {}
        stage_ref: Dict[str, Stage] = {}
        task_of: Dict[str, Task] = {}
        for t in tasks:
            for sid, stg in t.stages.items():
                indeg[sid] = len(stg.dependencies)
                stage_ref[sid] = stg
                task_of[sid] = t

        def priority(stg: Stage) -> float:
            # longest predicted time first (max over nodes) — simple but effective
            return max(stg._per_node_runtime.values() or [0.0])

        ready = []
        for sid, stg in stage_ref.items():
            if indeg[sid] == 0:
                heapq.heappush(ready, (-priority(stg), sid))  # max-heap via negative

        # track last placed node per task to add comm penalty
        last_node_for_stage: Dict[str, str] = {}

        while ready:
            _, sid = heapq.heappop(ready)
            stg = stage_ref[sid]
            t = task_of[sid]

            # find predecessor node (for penalty)
            prev_sid = stg.dependencies[-1] if stg.dependencies else None
            prev_node_id = None
            if prev_sid:
                prev_node_id = stage_ref[prev_sid].assigned_node.node_id if stage_ref[prev_sid].assigned_node else None

            # choose node that minimizes earliest finish time
            best_node = None
            best_finish = float("inf")
            for n in self.nodes:
                run_t = stg._per_node_runtime.get(n.node_id, float("inf"))
                if run_t == float("inf"):
                    continue
                start_t = max(self.node_available[n.node_id], 0.0)
                finish_t = start_t + self._penalty(prev_node_id, n.node_id) + run_t

                # tiny bias so we don't always pile onto the fastest node
                finish_t += self.bias_idle * self.node_available[n.node_id]

                if finish_t < best_finish:
                    best_finish = finish_t
                    best_node = n

            # assign and update calendars
            stg.assigned_node = best_node
            self.node_available[best_node.node_id] = best_finish
            last_node_for_stage[sid] = best_node.node_id

            # unlock dependents
            for dep_sid in stg.dependents:
                indeg[dep_sid] -= 1
                if indeg[dep_sid] == 0:
                    heapq.heappush(ready, (-priority(stage_ref[dep_sid]), dep_sid))


# ===== Min-Min (EFT) deterministic scheduler =====
class MinMinEFTScheduler:
    """
    Deterministic Min-Min scheduler for heterogeneous nodes.
    - Computes EFT (earliest finish time) for each (ready stage, node).
    - Picks the global minimum with deterministic tie-breakers.
    - OOM-aware; size-aware comm penalty.
    - Optional GPU-gate: only offload to CPU if it beats the best-GPU EFT by a % margin.
    """
    _BW = {("cpu","cpu"): 30e9, ("cpu","gpu"): 12e9, ("gpu","cpu"): 12e9, ("gpu","gpu"): 300e9}

    def __init__(self, nodes, gpu_gate_pct: Optional[float] = None, debug: bool = False):
        """
        gpu_gate_pct: None to disable; else a fraction (e.g., 0.02 for 2%)
                      CPU candidate must finish <= (1 - gpu_gate_pct) * best_GPU_EFT
        """
        self.nodes = list(nodes)
        self.node_available = {n.node_id: 0.0 for n in nodes}  # wall-clock seconds (predicted)
        self.gpu_gate_pct = gpu_gate_pct
        self.debug = debug

    # ----- helpers -----
    def _is_gpu(self, node_id: str) -> bool:
        n = next(nd for nd in self.nodes if nd.node_id == node_id)
        return n.gpu is not None

    def _runtime_s(self, stg, node_id: str) -> float:
        # stg._per_node_runtime is in microseconds; fall back to median*1.2 if missing
        us = stg._per_node_runtime.get(node_id, float("inf"))
        if not (us > 0 and us != float("inf")):
            others = [v for k, v in stg._per_node_runtime.items() if v > 0 and v != float("inf")]
            if others:
                import statistics as stats
                us = 1.2 * stats.median(others)
        return float(us) / 1e6 if us != float("inf") else float("inf")

    def _comm_s(self, prev_node_id: str | None, node_id: str, stg) -> float:
        if not prev_node_id or prev_node_id == node_id:
            return 0.0
        bytes_hint = float(getattr(stg, "_bytes_hint", 0.0))
        a = "gpu" if self._is_gpu(prev_node_id) else "cpu"
        b = "gpu" if self._is_gpu(node_id) else "cpu"
        bw = self._BW.get((a, b), 12e9)
        return bytes_hint / bw

    def _fits(self, stg, node_id: str) -> bool:
        # crude OOM guard using bytes_hint; skip for CPU
        n = next(nd for nd in self.nodes if nd.node_id == node_id)
        if n.gpu is None or not torch.cuda.is_available():
            return True
        try:
            free, total = torch.cuda.mem_get_info(n.gpu)
            need = float(getattr(stg, "_bytes_hint", 0.0))
            return need < 0.80 * free  # 80% headroom
        except Exception:
            return True

    # ----- main -----
    def schedule(self, tasks):
        import time
        # Build indegree and a map
        stage_ref, indeg = {}, {}
        for t in tasks:
            for sid, stg in t.stages.items():
                stage_ref[sid] = stg
                indeg[sid] = len(stg.dependencies)

        # initial ready set
        ready = [sid for sid, d in indeg.items() if d == 0]

        while ready:
            now = time.monotonic()
            best = None  # (finish_s, sid, node_id)

            for sid in ready:
                stg = stage_ref[sid]
                # predecessor node (for comm)
                prev_sid = stg.dependencies[-1] if stg.dependencies else None
                prev_node_id = stage_ref[prev_sid].assigned_node.node_id if prev_sid and stage_ref[prev_sid].assigned_node else None

                # compute best GPU EFT for the gate (if any GPU exists)
                gpu_eft = float("inf")
                if any(self._is_gpu(n.node_id) for n in self.nodes):
                    for g in self.nodes:
                        if not self._is_gpu(g.node_id): continue
                        if not self._fits(stg, g.node_id): continue
                        run_s_g = self._runtime_s(stg, g.node_id)
                        if run_s_g == float("inf"): continue
                        start_g = max(self.node_available[g.node_id], now)
                        eft_g = start_g + self._comm_s(prev_node_id, g.node_id, stg) + run_s_g
                        if eft_g < gpu_eft: gpu_eft = eft_g

                for n in self.nodes:
                    if not self._fits(stg, n.node_id):
                        continue
                    run_s = self._runtime_s(stg, n.node_id)
                    if run_s == float("inf"):
                        continue
                    start = max(self.node_available[n.node_id], now)
                    finish = start + self._comm_s(prev_node_id, n.node_id, stg) + run_s

                    # Optional GPU-gate (percentage margin)
                    if self.gpu_gate_pct is not None and not self._is_gpu(n.node_id) and gpu_eft != float("inf"):
                        # allow CPU only if it beats GPU by gpu_gate_pct fraction
                        if finish > (1.0 - self.gpu_gate_pct) * gpu_eft:
                            continue

                    cand = (finish, sid, n.node_id)
                    if (best is None or finish < best[0] or
                        (abs(finish - best[0]) <= 1e-9 and (sid < best[1] or (sid == best[1] and n.node_id < best[2])))):
                        best = cand

            # assign best pair
            if best is None:
                # no feasible node (OOM everywhere?) -> fall back to fastest GPU ignoring fits
                sid = ready[0]
                stg = stage_ref[sid]
                n = min(self.nodes, key=lambda nd: self._runtime_s(stg, nd.node_id))
                best = (now + self._runtime_s(stg, n.node_id), sid, n.node_id)

            finish_s, sid, node_id = best
            stg = stage_ref[sid]
            node = next(nd for nd in self.nodes if nd.node_id == node_id)
            stg.assigned_node = node
            self.node_available[node_id] = finish_s
            if self.debug:
                print(f"[MinMinEFT] {sid} -> {node_id} (pred_finish={finish_s:.6f}s)")

            # pop from ready, unlock dependents
            ready.remove(sid)
            for dep_sid in stg.dependents:
                indeg[dep_sid] -= 1
                if indeg[dep_sid] == 0:
                    ready.append(dep_sid)
