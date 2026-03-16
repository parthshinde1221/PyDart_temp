from __future__ import annotations

import copy
import csv
import gc
import os
import time
import warnings
import shutil
from collections import defaultdict
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import torch
from viztracer import VizTracer

from .metrics import ArithmeticIntensityMetric
from .model_spec import get_model_spec
from .node import Node
from .profiler import Profiler
from .task import Evaluator, Task, Taskset
from .utils import set_seed
from .paths import OUTPUTS_DIR,OUTPUTS_DIR_RUN
warnings.filterwarnings("ignore")

HEAVY_EXPERIMENT_SPECS = ["heavy_resnet50"]
LIGHT_EXPERIMENT_SPECS = ["light_cnn"]




# from pathlib import Path

# REPO_ROOT = Path.cwd().parents[1]
# OUTPUTS_DIR = REPO_ROOT / "outputs"
# TRACE_OUTPUT_FILE = REPO_ROOT / "outputs" / 
# OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
# print(OUTPUTS_DIR)
# print()


def create_task_from_spec(
    task_id: str,
    spec_name: str,
    profiler: Profiler,
    metric: Optional[ArithmeticIntensityMetric] = None,
) -> Task:
    spec = get_model_spec(spec_name)
    model = spec.build_model()
    input_batch = spec.build_input_batch()

    return Task(
        task_id=task_id,
        model=model,
        input_data=input_batch,
        model_name=model.__class__.__name__,
        profiler=profiler,
        load_metric=metric or ArithmeticIntensityMetric(),
    )


def run_experiment(
    k: int,
    heavy_light_ratio: Tuple[int, int],
    num_tasks: int = 10,
) -> Evaluator:
    set_seed(42)

    OUTPUTS_DIR_RUN.mkdir(parents=True, exist_ok=True)

    nodes = Node.discover_shared_gpu_workers(num_workers=k, gpu_id=0)
    print(f"Using {len(nodes)} shared-GPU workers: {nodes}")

    profiler = Profiler(mode="init")

    heavy_count = (num_tasks * heavy_light_ratio[0]) // (
        heavy_light_ratio[0] + heavy_light_ratio[1]
    )
    light_count = num_tasks - heavy_count
    print(f"Creating {heavy_count} heavy tasks and {light_count} light tasks.")

    tasks: List[Task] = []

    for i in range(heavy_count):
        spec_name = HEAVY_EXPERIMENT_SPECS[i % len(HEAVY_EXPERIMENT_SPECS)]
        task = create_task_from_spec(
            task_id=f"heavy_{i+1}",
            spec_name=spec_name,
            profiler=profiler,
            metric=ArithmeticIntensityMetric(),
        )
        tasks.append(task)

    for i in range(light_count):
        spec_name = LIGHT_EXPERIMENT_SPECS[i % len(LIGHT_EXPERIMENT_SPECS)]
        task = create_task_from_spec(
            task_id=f"light_{i+1}",
            spec_name=spec_name,
            profiler=profiler,
            metric=ArithmeticIntensityMetric(),
        )
        tasks.append(task)

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

    for task in taskset.tasks:
        task.print_stage_allocations()

    evaluator = Evaluator(taskset, profiler)

    tracer_naive = VizTracer(
        output_file=f"{OUTPUTS_DIR_RUN}/result_naive_{heavy_light_ratio}_list_scheduling.html"
    )
    tracer_naive.start()
    evaluator.run_naive_execution()
    tracer_naive.stop()
    tracer_naive.save()

    tracer_parallel = VizTracer(
        output_file=f"{OUTPUTS_DIR_RUN}/result_parallel_{heavy_light_ratio}_list_scheduling.html"
    )
    tracer_parallel.start()
    evaluator.run_parallel_execution()
    tracer_parallel.stop()
    tracer_parallel.save()

    evaluator.compare_outputs()
    evaluator.analyze_speedup_throughput()

    # del evaluator
    # gc.collect()

    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_peak_memory_stats()
    
    for node in nodes:
        node.stop()

    return evaluator


def run_multiple_experiments(k: int, num_tasks: int = 15) -> None:
    ratios = [round(i / 10, 1) for i in range(1, 10)]
    speedups = []
    parallel_throughputs = []
    naive_throughputs = []
    makespans = []
    config = defaultdict(dict)

    # from pathlib import Path

    if OUTPUTS_DIR.exists():
        shutil.rmtree(OUTPUTS_DIR)
        print("Old Outputs deleted")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("------OUTPUT DIR------")
    print(OUTPUTS_DIR)
    print("------OUTPUT DIR------")

    csv_filename = OUTPUTS_DIR / "experiment_results.csv"
    pdf_filename = OUTPUTS_DIR / "experiment_plots.pdf"
    profiling_file = OUTPUTS_DIR / "profiling_results.csv"

    for ratio in ratios:
        heavy_count = int(num_tasks * ratio)
        light_count = num_tasks - heavy_count
        print(f"Running experiment with {heavy_count} heavy and {light_count} light tasks.")

        if os.path.exists(profiling_file):
            os.remove(profiling_file)
            print("The profiling file was removed.")

        evaluator = run_experiment(
            k=k,
            heavy_light_ratio=(heavy_count, light_count),
            num_tasks=num_tasks,
        )

        sp = evaluator.speedup_makespan
        par_th = evaluator.throughput_makespan
        naive_thr = (num_tasks / evaluator.naive_makespan) if evaluator.naive_makespan > 0 else 0
        mk = evaluator.parallel_makespan

        speedups.append(sp)
        parallel_throughputs.append(par_th)
        naive_throughputs.append(naive_thr)
        makespans.append(mk)

        config[ratio] = {
            "speedup": sp,
            "parallel_throughput": par_th,
            "naive_throughput": naive_thr,
            "makespan": mk,
        }

        print(f"Speedup for ratio {ratio}: {sp:.2f}")
        print(f"Parallel Throughput for ratio {ratio}: {par_th:.2f}")
        print(f"Naive Throughput for ratio {ratio}: {naive_thr:.2f}")
        print(f"Makespan for ratio {ratio}: {mk:.2f}\n")

        del evaluator
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    print("Final Results:")
    print("Speedups:", speedups)
    print("Parallel Throughputs:", parallel_throughputs)
    print("Naive Throughputs:", naive_throughputs)
    print("Makespans:", makespans)

    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Heavy Task Ratio", "Speedup", "Parallel Throughput", "Naive Throughput", "Makespan"]
        )
        for i, ratio in enumerate(ratios):
            writer.writerow(
                [ratio, speedups[i], parallel_throughputs[i], naive_throughputs[i], makespans[i]]
            )

    print(f"Experiment results saved to {csv_filename}")

    max_ratio = max(config, key=lambda r: config[r]["speedup"])
    best_config = config[max_ratio]

    print(f"Best configuration: {best_config}")
    print(f"Max Speedup Ratio: {max_ratio}\n")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(ratios, speedups, marker="o", linestyle="-", label="Speedup")
    axes[0].set_xlabel("Heavy Task Ratio")
    axes[0].set_ylabel("Speedup (makespan)")
    axes[0].set_title("Speedup vs Heavy Task Ratio")
    axes[0].legend()

    axes[1].plot(
        ratios,
        parallel_throughputs,
        marker="s",
        linestyle="-",
        label="Parallel Throughput",
    )
    axes[1].plot(
        ratios,
        naive_throughputs,
        marker="^",
        linestyle="--",
        color="orange",
        label="Naive Throughput",
    )
    axes[1].set_xlabel("Heavy Task Ratio")
    axes[1].set_ylabel("Throughput (tasks/s)")
    axes[1].set_title("Throughput Comparison")
    axes[1].legend()

    axes[2].plot(ratios, makespans, marker="d", linestyle="-", label="Makespan")
    axes[2].set_xlabel("Heavy Task Ratio")
    axes[2].set_ylabel("Makespan (s)")
    axes[2].set_title("Makespan vs Heavy Task Ratio")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(pdf_filename)
    print(f"Plots saved to {pdf_filename}")
    plt.show()