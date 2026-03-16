from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Tuple

from torch.fx import Node as FxNode

if TYPE_CHECKING:
    from .node import Node
    from .task import Task, Taskset


# Define the metric interface as an abstract base class.
class MetricInterface(ABC):
    @abstractmethod
    def compute_task(self, task: Task, taskset: Taskset) -> float:
        """
        Compute an overall metric for the entire task.
        """
        pass

    @abstractmethod
    def compute_layer(self, task: Task, node: Node, fx_node: FxNode) -> float:
        """
        Compute the cost metric for a single layer (fx_node) executed on a specific node.
        """
        pass

# Implement the HPCMakespanMetric that follows the MetricInterface.
class HPCMakespanMetric(MetricInterface):
    def __init__(self):
        pass

    def compute_task(self, task: Task, taskset: Taskset) -> float:
        # Compute the overall forward pass time of the task as the task-level metric.
        return float(task.get_forward_pass_time(sum_across_compute=True))

    def compute_layer(self, task: Task, node: Node, fx_node: FxNode) -> float:
        # Use a composite key (node ID and FX node name) to retrieve profiling records.
        key: Tuple[str, str] = (node.node_id, fx_node.name)
        record: Dict[str, Any] = task.prof_records.get(key, None)
        if record is None:
            return 0.0
        return float(record.get("Total Execution Time (us)", 0.0))


class ArithmeticIntensityMetric(MetricInterface):
    def __init__(self):
        pass

    def compute_task(self, task: Task, taskset: Taskset) -> float:
        """
        For the whole task, use the 'forward_pass' row from the profiler data.
        Returns 1/AI = (Memory Accessed) / (FLOPs).
        - If memory accessed is zero and FLOPs is high, return 0.0.
        - If FLOPs is zero (regardless of memory), return float('inf').
        """
        df = task.profiler.get_profile_db()
        mask = (df['Task_ID'] == task.task_id) & (df['Layer'] == 'forward_pass')
        matched = df.loc[mask]
        if matched.empty:
            return 0.0
        total_flops = matched['FLOPs'].sum()
        total_memory = matched['Memory Accessed (bytes)'].sum()
        # Case: FLOPs is zero -> memory-bound --> return inf
        if total_flops == 0:
            return float('-inf')
        # If memory is zero and FLOPs > 0 then 1/AI = 0.
        if total_memory == 0:
            return 0.0

        # print(float(total_memory / total_flops))
        return float(total_memory / total_flops) * 100

    def compute_layer(self, task: Task, node: Node, fx_node: FxNode) -> float:
        """
        For a given layer, retrieve its profile record and compute 1/AI = (Memory Accessed) / (FLOPs).
        - If FLOPs is zero (even if memory is nonzero or zero), return float('inf').
        - If memory is zero and FLOPs > 0, return 0.0.
        """
        key = (node.node_id, fx_node.name)
        record = task.prof_records.get(key, None)
        if record is None:
            return 0.0
        flops = record.get("FLOPs", 0.0)
        memory_accessed = record.get("Memory Accessed (bytes)", 0.0)
        if flops == 0:
            return float('-inf')
        if memory_accessed == 0:
            return 0.0

        # print(float(memory_accessed / flops))
        return float(memory_accessed / flops) * 100
