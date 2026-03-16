# PyDart

PyDart is a lightweight experimental framework for studying and optimizing multi-model inference execution and scheduling across shared compute resources. It is designed to make it easy to compare simple baseline execution against PyDart’s scheduled parallel execution, while keeping the workflow minimal and understandable.

> **Note:** This repo is made for the demo and is mainly experimental.

## Current Scope

PyDart currently supports:

- built-in model-registry-based experiments
- manual Python workflows for custom models and custom task construction
- sequential baseline execution through `run_naive_execution()`
- scheduled parallel execution through the PyDart task execution pipeline

The codebase is structured so that simple experiments can be run from the CLI, while advanced or custom experiments are better handled through Python scripts or notebooks.

## Execution Model

At the moment, PyDart compares two primary execution paths:

### 1. Sequential baseline
This is the current `run_naive_execution()` path in `Evaluator`.

In this mode:

- tasks are executed one by one
- each task is run in a simple sequential loop
- execution time, completion time, outputs, and makespan are recorded
- this acts as the current baseline for comparison

### 2. PyDart scheduled parallel execution
This is the current `run_parallel_execution()` path in `Evaluator`.

In this mode:

- tasks are assigned through the PyDart execution framework
- the `Taskset` executes all tasks across workers/nodes
- outputs, per-task execution times, completion times, and makespan are collected
- this is the main framework-driven execution path

## Planned Baseline Extension

A fully async parallel baseline is also planned.

The async baseline code path has already been explored and will be integrated into the baseline execution flow. This will provide an additional comparison point beyond the current sequential baseline.

The intended purpose of that async baseline is to represent a completely parallel host-side launch strategy, where all tasks are submitted asynchronously as a baseline against PyDart’s structured scheduling.

For now:

- `run_naive_execution()` should be treated as the **sequential baseline**
- the async baseline is acknowledged as part of the roadmap and ongoing implementation work

## System Diagram
![PyDart System Diagram](System_Diagram.png)

## Installation

### Requirements
- Python 3.9+
- pip
- PyTorch installed for your platform/device

> **Important**
> - A virtual environment is strongly recommended before installing PyDart.
> - PyTorch should be installed first, using the correct build for your system (CPU or CUDA, CUDA is best (CPU+GPU) recommeneded systems), before installing or running PyDart.

### Setup and Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/parthshinde1221/PyDart_temp.git
   cd PyDart_temp

2. **Clone the repository**
   ```bash
   create your virtual env using venv or uv python module and activate it

3. **Clone the repository**
   ```bash
   pip install -e .


3. **Verify the Installation**
   ```bash
   pydart
   pydart --help

## CLI

PyDart provides a minimal CLI for built-in experiments.

The CLI is intentionally small and only exposes the simplest built-in execution paths using the default model registry.

### Commands

1. **Show CLI help**
    ```bash
    pydart
    pydart --help

2. **Run single built-in experiment**
   ```bash
    pydart run --workers 2 --ratio 1:1 --tasks 8

3. **Run multiple built-in experiments**
   ```bash
   pydart sweep --workers 2 --tasks 8


## Python for custom workflows


Use Python scripts or notebooks when you want to:

a. define custom nn.Modules
b. define custom ModelSpecs
c. create custom dataloaders
d. control tracing manually
e. build tasks explicitly
f. profile tasks manually
g. experiment with evaluator logic directly

This separation keeps the CLI minimal while still letting PyDart act like a flexible library.

### Running a Custom Python File

For custom experiments , run a python file directly:
python examples/custom_file_name.py

Output folder relvance is its used outputs for all the single or multiple experiments run and clears everytime you run the experiment
---
