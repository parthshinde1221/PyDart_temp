from __future__ import annotations

import argparse
import gc

import torch

from pydart.experiment import run_experiment, run_multiple_experiments


def parse_ratio(text: str) -> tuple[int, int]:
    try:
        heavy, light = text.split(":")
        heavy_i = int(heavy)
        light_i = int(light)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Ratio must be in the form H:L, for example 1:1 or 2:3."
        ) from exc

    if heavy_i < 0 or light_i < 0:
        raise argparse.ArgumentTypeError("Ratio values must be non-negative.")

    if heavy_i == 0 and light_i == 0:
        raise argparse.ArgumentTypeError("At least one side of the ratio must be > 0.")

    return heavy_i, light_i


def cmd_run(args: argparse.Namespace) -> None:
    ratio = parse_ratio(args.ratio)

    print(
        f"Running single built-in PyDart experiment "
        f"(workers={args.workers}, ratio={ratio}, tasks={args.tasks}, "
        f"baseline_mode={args.baseline_mode})."
    )

    evaluator = run_experiment(
        k=args.workers,
        heavy_light_ratio=ratio,
        num_tasks=args.tasks,
        mode=args.baseline_mode,
    )

    del evaluator

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def cmd_sweep(args: argparse.Namespace) -> None:
    print(
        f"Running built-in PyDart experiment sweep "
        f"(workers={args.workers}, tasks={args.tasks}, "
        f"baseline_mode={args.baseline_mode})."
    )

    run_multiple_experiments(
        k=args.workers,
        num_tasks=args.tasks,
        mode=args.baseline_mode,
    )


def build_parser() -> argparse.ArgumentParser:
    print("Welcome to PyDart CLI.")
    parser = argparse.ArgumentParser(description="PyDart CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a single built-in experiment using the default model registry.",
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of workers to use. Recommended: number of CPU cores. Default: 2",
    )
    run_parser.add_argument(
        "--ratio",
        type=str,
        default="1:1",
        help="Heavy:light task ratio in H:L format. Default: 1:1",
    )
    run_parser.add_argument(
        "--tasks",
        type=int,
        default=10,
        help="Total number of tasks. Default: 10",
    )
    run_parser.add_argument(
        "--baseline-mode",
        type=str,
        choices=["sequential", "async"],
        default="sequential",
        help="Baseline execution mode. Choices: sequential or async. Default: sequential",
    )
    run_parser.set_defaults(func=cmd_run)

    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Run multiple built-in experiments using the default model registry.",
    )
    sweep_parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of workers to use. Recommended: number of CPU cores. Default: 2",
    )
    sweep_parser.add_argument(
        "--tasks",
        type=int,
        default=10,
        help="Total number of tasks per experiment. Default: 10",
    )
    sweep_parser.add_argument(
        "--baseline-mode",
        type=str,
        choices=["sequential", "async"],
        default="sequential",
        help="Baseline execution mode. Choices: sequential or async. Default: sequential",
    )
    sweep_parser.set_defaults(func=cmd_sweep)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    if args.workers <= 0:
        parser.error("--workers must be greater than 0")

    if args.tasks <= 0:
        parser.error("--tasks must be greater than 0")

    args.func(args)


if __name__ == "__main__":
    main()