from __future__ import annotations

import copy
import time
from pathlib import Path
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from viztracer import VizTracer

from pydart import (
    ArithmeticIntensityMetric,
    Evaluator,
    ModelSpec,
    Node,
    Profiler,
    Task,
    Taskset,
)
from pydart.utils import set_seed

from pydart.paths import CUSTOM_OUTPUTS_DIR

CUSTOM_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
if CUSTOM_OUTPUTS_DIR.exists():
    shutil.rmtree(CUSTOM_OUTPUTS_DIR)
    print("Old Outputs deleted")

CUSTOM_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

print("------CUSTOM OUTPUT DIR------")
print(CUSTOM_OUTPUTS_DIR)
print("------CUSTOM OUTPUT DIR------")

class CustomMLP(nn.Module):
    def __init__(
        self,
        in_features: int = 128,
        hidden: int = 2048,
        out_features: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DemoSwinTiny(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = models.swin_t(weights=None)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_mlp_dataloader() -> DataLoader:
    inputs = torch.randn(100, 128)
    targets = torch.randint(0, 64, (100,))
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=16)


def build_swin_dataloader() -> DataLoader:
    inputs = torch.randn(32, 3, 224, 224)
    targets = torch.randint(0, 10, (32,))
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=2)


def build_custom_mlp_spec() -> ModelSpec:
    return ModelSpec(
        name="custom_mlp",
        builder=lambda: CustomMLP(),
        input_size=(128,),
        batch_size=16,
        num_samples=100,
        num_classes=64,
        dataloader_fn=build_mlp_dataloader,
    )


def build_swin_spec() -> ModelSpec:
    return ModelSpec(
        name="swin_tiny",
        builder=lambda: DemoSwinTiny(num_classes=10),
        input_size=(3, 224, 224),
        batch_size=2,
        num_samples=32,
        num_classes=10,
        dataloader_fn=build_swin_dataloader,
    )


def create_task_from_spec(
    task_id: str,
    spec: ModelSpec,
    profiler: Profiler,
    metric: ArithmeticIntensityMetric,
) -> Task:
    model = spec.build_model().eval()
    input_data = spec.build_input_batch()

    return Task(
        task_id=task_id,
        model=model,
        input_data=input_data,
        model_name=model.__class__.__name__,
        profiler=profiler,
        load_metric=metric,
    )


def main() -> None:
    set_seed(42)

    nodes = Node.discover_shared_gpu_workers(num_workers=2, gpu_id=0)
    print(f"Using {len(nodes)} shared-GPU workers: {nodes}")

    profiler = Profiler(mode="init")
    metric = ArithmeticIntensityMetric()

    mlp_spec = build_custom_mlp_spec()
    swin_spec = build_swin_spec()

    tasks = []

    for i in range(2):
        tasks.append(
            create_task_from_spec(
                task_id=f"mlp_task_{i+1}",
                spec=mlp_spec,
                profiler=profiler,
                metric=metric,
            )
        )

    for i in range(2):
        tasks.append(
            create_task_from_spec(
                task_id=f"swin_task_{i+1}",
                spec=swin_spec,
                profiler=profiler,
                metric=metric,
            )
        )

    for task in tasks:
        for node in nodes:
            inp_copy = copy.deepcopy(task.input_data)
            profiler.profile_model(
                model=task.model,
                input_data=inp_copy,
                node=node,
                task_id=task.task_id,
            )
            time.sleep(0.05)

    for task in tasks:
        task.populate_profile_records()

    taskset = Taskset(tasks, nodes)
    evaluator = Evaluator(taskset, profiler)

    naive_trace = CUSTOM_OUTPUTS_DIR / "custom_mix_naive_trace.html"
    parallel_trace = CUSTOM_OUTPUTS_DIR / "custom_mix_parallel_trace.html"

    tracer_naive = VizTracer(output_file=str(naive_trace))
    tracer_naive.start()
    evaluator.run_baseline_execution(mode="async")
    tracer_naive.stop()
    tracer_naive.save()

    tracer_parallel = VizTracer(output_file=str(parallel_trace))
    tracer_parallel.start()
    evaluator.run_parallel_execution()
    tracer_parallel.stop()
    tracer_parallel.save()

    evaluator.compare_outputs()
    evaluator.analyze_speedup_throughput()

    print(f"Naive trace saved to: {naive_trace}")
    print(f"Parallel trace saved to: {parallel_trace}")
    print(f"All custom outputs stored in: {CUSTOM_OUTPUTS_DIR.resolve()}")

    for node in nodes:
        node.stop()


if __name__ == "__main__":
    main()