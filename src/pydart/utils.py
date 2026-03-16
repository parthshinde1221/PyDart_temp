from __future__ import annotations

import time
from typing import Any, Dict, List

import torch
from torch.fx import Node as FxNode

from .node import Node

def resolve_arg(arg: Any, node_outputs: Dict[str, torch.Tensor]) -> Any:
    if isinstance(arg, FxNode):
        return node_outputs.get(arg.name, arg)
    elif isinstance(arg, (list, tuple)):
        return type(arg)(resolve_arg(a, node_outputs) for a in arg)
    elif isinstance(arg, dict):
        return {k: resolve_arg(v, node_outputs) for k, v in arg.items()}
    else:
        return arg

def move_tensor_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True) if obj.device != device else obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_tensor_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_tensor_to_device(v, device) for k, v in obj.items()}
    else:
        return obj

def measure_max_transfer_penalty(available_nodes: List[Node], sample_tensor: torch.Tensor) -> float:
    max_time = 0.0
    for src in available_nodes:
        src_device = torch.device(f"cuda:{src.gpu}") if src.gpu is not None and torch.cuda.is_available() else torch.device("cpu")
        tensor_on_src = sample_tensor.to(src_device)
        for dst in available_nodes:
            dst_device = torch.device(f"cuda:{dst.gpu}") if dst.gpu is not None and torch.cuda.is_available() else torch.device("cpu")
            start = time.time()
            _ = tensor_on_src.to(dst_device)
            if dst_device.type == 'cuda':
                torch.cuda.synchronize(dst_device)
            elapsed = time.time() - start
            max_time = max(max_time, elapsed)
    return max_time


def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

# placeholder
def _resolve_attr_path(root, path: str):
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj