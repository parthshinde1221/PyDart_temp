# import os
# import io
# import copy
# import time
# import queue
# import threading
# import torch
# import torch.fx as fx
# import torch.nn as nn
# import torch.profiler
# import torch.cuda
# import networkx as nx
# import pandas as pd
# from torch.fx import Node as FxNode
# from torch.utils.data import DataLoader, TensorDataset
# from torchvision import models
# from typing import Callable, Any, List, Dict, Optional, Set, Tuple
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from concurrent.futures import ThreadPoolExecutor, as_completed

"""## Node

**Description:**  
The **Node** class represents a compute node (a CPU core or a GPU with an associated CPU core) responsible for executing tasks. 
Each node maintains its own task queue and runs a dedicated worker thread. 
Before executing a task, it sets the proper CPU affinity and GPU device context. 
Additionally, the class provides a static method to discover available nodes based on the system’s resources.
---
"""

import os
import io
import copy
import time
import queue
import threading
import logging
import ctypes
from ctypes import wintypes
from typing import Any, Dict, Tuple, Callable, List, Optional

import torch
import torch.fx as fx
import torch.nn as nn
from viztracer import get_tracer
from contextlib import nullcontext

# ----------------------------- Optional fvcore ------------------------------
try:
    from fvcore.nn import FlopCountAnalysis
    _FVCORE_OK = True
except Exception:
    _FVCORE_OK = False
    FlopCountAnalysis = None  # type: ignore

# ----------------------------- Logging setup -------------------------------
logging.basicConfig(
    filename='trace.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(threadName)s - %(message)s'
)

# ------------------- Cross-platform CPU affinity (no psutil) ----------------
# Fallback for DWORD_PTR on some Python builds
if not hasattr(wintypes, "DWORD_PTR"):
    DWORD_PTR = ctypes.c_size_t
else:
    DWORD_PTR = wintypes.DWORD_PTR

class _AffinityShim:
    """Best-effort helpers for getting/setting CPU affinity without psutil."""
    _is_windows = (os.name == "nt")
    _has_sched = hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity")

    @staticmethod
    def cpu_count() -> int:
        return os.cpu_count() or 1

    @staticmethod
    def get_process_affinity() -> Optional[set]:
        """
        Return set of CPU indices allowed for the *current process* or None if unsupported.
        Linux: os.sched_getaffinity(0)
        Windows: GetProcessAffinityMask
        """
        if _AffinityShim._is_windows:
            try:
                k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                GetCurrentProcess = k32.GetCurrentProcess
                GetProcessAffinityMask = k32.GetProcessAffinityMask
                GetProcessAffinityMask.argtypes = [
                    wintypes.HANDLE,
                    ctypes.POINTER(DWORD_PTR),
                    ctypes.POINTER(DWORD_PTR),
                ]
                GetProcessAffinityMask.restype = wintypes.BOOL

                hProc = GetCurrentProcess()
                proc_mask = DWORD_PTR()
                sys_mask = DWORD_PTR()
                ok = GetProcessAffinityMask(hProc, ctypes.byref(proc_mask), ctypes.byref(sys_mask))
                if not ok:
                    return None
                mask = int(proc_mask.value)
                return {i for i in range(_AffinityShim.cpu_count()) if (mask >> i) & 1} or None
            except Exception:
                return None

        if _AffinityShim._has_sched:
            try:
                return set(os.sched_getaffinity(0))
            except Exception:
                return None

        return None

    @staticmethod
    def set_thread_affinity(cpus: Optional[set]) -> bool:
        """
        Best-effort set affinity for the *current thread*.
        If cpus is None/empty, reset to process-allowed set.
        """
        if _AffinityShim._is_windows:
            try:
                k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                SetThreadAffinityMask = k32.SetThreadAffinityMask
                GetCurrentThread = k32.GetCurrentThread

                # Reset to allowed set if cpus not provided
                if not cpus:
                    cpus = _AffinityShim.get_process_affinity()
                    if not cpus:
                        return True  # no-op

                mask = 0
                for i in cpus:
                    mask |= (1 << int(i))
                prev = SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(mask))
                return bool(prev)
            except Exception:
                return False

        if _AffinityShim._has_sched:
            try:
                if cpus:
                    os.sched_setaffinity(0, cpus)
                else:
                    allowed = _AffinityShim.get_process_affinity()
                    if allowed:
                        os.sched_setaffinity(0, allowed)
                return True
            except Exception:
                return False

        return True  # unsupported -> no-op


# --------------------------------- Node -------------------------------------
class Node:
    """
    Lightweight worker bound to optional CPUs and/or one GPU device.
    Tasks execute in the worker thread; we pin affinity before each task
    and restore it after to avoid leaking affinity changes.
    """
    def __init__(self, node_id: str, cpus=None, gpu=None):
        self._node_id = node_id
        self._cpus = tuple(cpus or [])
        self._gpu = gpu

        self._task_queue = queue.Queue()
        self._stop_signal = False
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"Worker-{node_id}"
        )
        self._worker_thread.start()
        self.current_load = 0.0
        self.assigned_stages = []

    @property
    def node_id(self):
        return self._node_id

    @property
    def cpus(self):
        return self._cpus

    @property
    def gpu(self):
        return self._gpu

    def assign_task(self, func: Callable, *args, **kwargs) -> queue.Queue:
        result_queue = queue.Queue(maxsize=1)
        self._task_queue.put((func, args, kwargs, result_queue))
        return result_queue

    def stop(self):
        self._stop_signal = True
        self._task_queue.put(None)
        self._worker_thread.join()

    def _worker_loop(self):
        while not self._stop_signal:
            item = self._task_queue.get()
            if item is None:
                break

            func, args, kwargs, result_queue = item
            tracer = get_tracer()
            trace_ctx = (
                tracer.log_event(f"Task_{func.__name__}")
                if tracer is not None
                else nullcontext()
            )

            with trace_ctx:
                logging.debug(f"Task START: {func.__name__}")
                start_time = time.time()

                # Capture original process-allowed set; used to 'unpin' on exit
                original_affinity = _AffinityShim.get_process_affinity()

                try:
                    self._set_context()
                    result = func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"Task ERROR: {func.__name__} with error {e}")
                    result = None
                finally:
                    # Restore affinity to original allowed set (best effort)
                    _AffinityShim.set_thread_affinity(original_affinity)
                    # Sync GPU if this node used one (best effort)
                    if self._gpu is not None and torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize(self._gpu)
                        except Exception:
                            pass

                end_time = time.time()
                logging.debug(f"Task FINISH: {func.__name__} Duration: {end_time - start_time:.4f}s")
                result_queue.put(result)

    def _set_context(self):
        # CPU affinity (best-effort)
        if self._cpus:
            _AffinityShim.set_thread_affinity(set(int(c) for c in self._cpus))

        # GPU device context (best-effort)
        if self._gpu is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_device(self._gpu)
                torch.cuda.synchronize(self._gpu)
            except Exception:
                pass  # don't fail the task on device errors

    @staticmethod
    def discover_nodes(disjoint: bool = True) -> List["Node"]:
        nodes: List[Node] = []
        num_cpus = os.cpu_count() or 1
        num_gpus = torch.cuda.device_count()

        if not disjoint:
            # Every CPU core as a CPU-only node
            for core_id in range(num_cpus):
                nodes.append(Node(node_id=f"CPU-{core_id}", cpus=[core_id]))

            # Every GPU paired with every CPU core
            for gpu_id in range(num_gpus):
                for core_id in range(num_cpus):
                    nodes.append(
                        Node(
                            node_id=f"GPU-{gpu_id}-CPU-{core_id}",
                            cpus=[core_id],
                            gpu=gpu_id,
                        )
                    )
            return nodes

        # Disjoint mode:
        # First create CPU-only nodes, then assign some CPUs to GPU-backed nodes
        cpu_nodes = [Node(node_id=f"CPU-{i}", cpus=[i]) for i in range(num_cpus)]
        gpu_nodes: List[Node] = []

        for gpu_id in range(num_gpus):
            if cpu_nodes:
                cpu_node = cpu_nodes.pop()
                gpu_nodes.append(
                    Node(
                        node_id=f"GPU-{gpu_id}-CPU-{cpu_node.cpus[0]}",
                        cpus=[cpu_node.cpus[0]],
                        gpu=gpu_id,
                    )
                )
            else:
                gpu_nodes.append(
                    Node(
                        node_id=f"GPU-{gpu_id}",
                        cpus=[],
                        gpu=gpu_id,
                    )
                )

        nodes = gpu_nodes + cpu_nodes
        return nodes

    @staticmethod
    def discover_shared_gpu_workers(num_workers: Optional[int] = None,
                                    gpu_id: int = 0) -> List["Node"]:
        """
        Create `num_workers` symmetric workers, all sharing the same GPU device.
        Each worker is pinned to one CPU core (round-robin if fewer cores).
        """
        n_cpus = _AffinityShim.cpu_count()
        if num_workers is None:
            num_workers = n_cpus  # or min(4, n_cpus) if you want fewer

        nodes: List[Node] = []
        for i in range(num_workers):
            core = i % n_cpus
            nodes.append(
                Node(
                    node_id=f"W{i}",
                    cpus=[core],
                    gpu=gpu_id      # <- ALL share GPU 0
                )
            )
        return nodes

    def __repr__(self):
        return f"Node({self._node_id}, cpus={self._cpus}, gpu={self._gpu})"
