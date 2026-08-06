from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.train_ppo_variant import (
    REPOSITORY_ROOT,
    TRAINING_VARIANTS,
    training_command,
)


BENCHMARK_SCHEMA = "training_device_benchmark_v1"
DEFAULT_OUTPUT = Path("runs/validation/training_device_benchmark")
BATCH_SIZE = 4 * 256
CUDA_SELECTION_MARGIN = 0.10


#################################################################################
# region Safety and selection

# Resolve one path relative to the repository root.
def repository_path(path: Path) -> Path:
#{
    return path if path.is_absolute() else REPOSITORY_ROOT / path
#} End function repository_path


# Restrict benchmark cleanup to the intended ignored validation subtree.
def validate_cleanup_target(path: Path) -> None:
#{
    resolved = path.resolve()
    validation_root = (REPOSITORY_ROOT / "runs" / "validation").resolve()

    if resolved == validation_root or validation_root not in resolved.parents:
        raise ValueError(f"Unsafe training-device benchmark path: {resolved}")

#} End function validate_cleanup_target


# Choose one final training device from stable benchmark rows.
def select_training_device(rows: pd.DataFrame) -> dict[str, object]:
#{
    required_cpu = {
        ("baseline", "cpu"),
        ("projection", "cpu"),
    }
    successful = {
        (str(row.variant), str(row.requested_device))
        for row in rows.itertuples(index=False)
        if row.status == "PASS"
    }

    missing_cpu = sorted(required_cpu - successful)
    if missing_cpu:
        raise ValueError(f"CPU benchmark conditions did not pass: {missing_cpu}")

    def elapsed(variant: str, device: str) -> float:
    # {
        selected = rows[
            (rows["variant"] == variant)
            & (rows["requested_device"] == device)
            & (rows["status"] == "PASS")
        ]
        if len(selected) != 1:
            raise ValueError(
                f"Expected one passing {variant}/{device} benchmark row; found {len(selected)}."
            )
        return float(selected.iloc[0]["elapsed_seconds"])
    # } End function elapsed

    cpu_cost = 2.0 * elapsed("baseline", "cpu") + elapsed("projection", "cpu")
    required_cuda = {
        ("baseline", "cuda"),
        ("projection", "cuda"),
    }
    cuda_complete = required_cuda.issubset(successful)

    if not cuda_complete:
        return {
            "selected_device": "cpu",
            "selection_reason": "CUDA benchmark was unavailable or did not complete stably.",
            "cpu_weighted_seconds": cpu_cost,
            "cuda_weighted_seconds": None,
            "cuda_speedup_fraction": None,
            "cuda_benchmark_complete": False,
            "cuda_selection_margin": CUDA_SELECTION_MARGIN,
        }

    cuda_cost = 2.0 * elapsed("baseline", "cuda") + elapsed("projection", "cuda")
    speedup = 1.0 - cuda_cost / cpu_cost

    if speedup >= CUDA_SELECTION_MARGIN:
        selected_device = "cuda"
        reason = (
            "CUDA completed all benchmark conditions and reduced the weighted "
            f"runtime by {100.0 * speedup:.1f}%."
        )
    else:
        selected_device = "cpu"
        reason = (
            "CUDA did not improve the weighted runtime by the predeclared "
            f"{100.0 * CUDA_SELECTION_MARGIN:.0f}% margin."
        )

    return {
        "selected_device": selected_device,
        "selection_reason": reason,
        "cpu_weighted_seconds": cpu_cost,
        "cuda_weighted_seconds": cuda_cost,
        "cuda_speedup_fraction": speedup,
        "cuda_benchmark_complete": True,
        "cuda_selection_margin": CUDA_SELECTION_MARGIN,
    }

#} End function select_training_device

# end region Safety and selection


#################################################################################
# region Execution

# Verify one completed benchmark checkpoint and return its runtime metadata.
def inspect_checkpoint(
    checkpoint_path: Path,
    *,
    variant_name: str,
    requested_device: str,
    seed: int,
    total_timesteps: int,
) -> dict[str, object]:
#{
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args")

    if not isinstance(args, dict):
        raise ValueError("Benchmark checkpoint does not contain an argument mapping.")

    variant = TRAINING_VARIANTS[variant_name]
    checks = {
        "method": (str(args.get("method")), variant.method),
        "seed": (int(args.get("seed", -1)), seed),
        "total_timesteps": (int(args.get("total_timesteps", -1)), total_timesteps),
        "global_step": (int(checkpoint.get("global_step", -1)), total_timesteps),
    }

    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"Benchmark checkpoint {name}={actual!r}; expected {expected!r}."
            )

    actual_device = str(checkpoint.get("device", ""))
    if actual_device != requested_device:
        raise ValueError(
            f"Requested {requested_device} but checkpoint recorded {actual_device!r}."
        )

    state_dict = checkpoint.get("agent_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Benchmark checkpoint does not contain a model state dictionary.")

    for name, value in state_dict.items():
        array = value.detach().cpu().numpy()
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Benchmark checkpoint tensor is non-finite: {name}")

    run_name = str(checkpoint.get("run_name", ""))
    if not run_name:
        raise ValueError("Benchmark checkpoint does not record its TensorBoard run name.")

    run_directory = REPOSITORY_ROOT / "runs" / run_name
    if not run_directory.is_dir():
        raise ValueError(f"Benchmark TensorBoard directory was not created: {run_directory}")

    return {
        "actual_device": actual_device,
        "run_name": run_name,
        "run_directory": str(run_directory),
    }

#} End function inspect_checkpoint


# Execute one isolated benchmark condition and preserve its complete log.
def run_condition(
    *,
    variant_name: str,
    device: str,
    seed: int,
    total_timesteps: int,
    output_directory: Path,
) -> dict[str, object]:
#{
    checkpoint_path = output_directory / "checkpoints" / f"{variant_name}_{device}.pt"
    log_path = output_directory / "logs" / f"{variant_name}_{device}.log"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_name = (
        f"training_device_benchmark_{variant_name}_{device}_"
        f"{total_timesteps}_seed{seed}"
    )
    command = training_command(
        TRAINING_VARIANTS[variant_name],
        seed,
        total_timesteps,
        checkpoint_path,
        experiment_name=experiment_name,
        device=device,
    )
    started = time.perf_counter()

    with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )

    elapsed_seconds = time.perf_counter() - started
    row: dict[str, object] = {
        "benchmark_schema_version": BENCHMARK_SCHEMA,
        "variant": variant_name,
        "method": TRAINING_VARIANTS[variant_name].method,
        "requested_device": device,
        "actual_device": "",
        "seed": seed,
        "total_timesteps": total_timesteps,
        "elapsed_seconds": elapsed_seconds,
        "transitions_per_second": total_timesteps / elapsed_seconds,
        "return_code": completed.returncode,
        "status": "FAIL",
        "checkpoint": str(checkpoint_path),
        "run_name": "",
        "run_directory": "",
        "log": str(log_path),
        "command": subprocess.list2cmdline(command),
        "failure_reason": "",
    }

    if completed.returncode != 0:
        row["failure_reason"] = f"Training process returned {completed.returncode}."
        return row
    if not checkpoint_path.is_file():
        row["failure_reason"] = "Training process did not create the benchmark checkpoint."
        return row

    try:
        metadata = inspect_checkpoint(
            checkpoint_path,
            variant_name=variant_name,
            requested_device=device,
            seed=seed,
            total_timesteps=total_timesteps,
        )
    except Exception as error:
        row["failure_reason"] = str(error)
        return row

    row.update(metadata)
    row["status"] = "PASS"
    return row

#} End function run_condition


# Run the complete engineering benchmark and freeze a device recommendation.
def benchmark_training_devices(
    *,
    seed: int,
    total_timesteps: int,
    output_directory: Path,
    replace: bool,
) -> dict[str, Path]:
#{
    if seed < 0:
        raise ValueError("Benchmark seed must be nonnegative.")
    if total_timesteps <= 0 or total_timesteps % BATCH_SIZE != 0:
        raise ValueError(
            f"Benchmark timesteps must be a positive multiple of {BATCH_SIZE}."
        )

    output = repository_path(output_directory).resolve()
    validate_cleanup_target(output)

    if output.exists():
        if not replace:
            raise FileExistsError(
                f"Training-device benchmark directory already exists: {output}"
            )
        shutil.rmtree(output)

    output.mkdir(parents=True)
    environment_path = output / "benchmark_environment.json"
    runs_path = output / "benchmark_runs.csv"
    decision_path = output / "benchmark_decision.json"
    summary_path = output / "benchmark_summary.txt"
    cuda_available = bool(torch.cuda.is_available())
    environment = {
        "benchmark_schema_version": BENCHMARK_SCHEMA,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if cuda_available else None
        ),
        "seed": seed,
        "total_timesteps": total_timesteps,
        "batch_size": BATCH_SIZE,
    }
    environment_path.write_text(
        json.dumps(environment, indent=2) + "\n",
        encoding="utf-8",
    )

    conditions = [
        ("baseline", "cpu"),
        ("projection", "cpu"),
    ]
    if cuda_available:
        conditions.extend(
            [
                ("baseline", "cuda"),
                ("projection", "cuda"),
            ]
        )

    rows = []

    for variant_name, device in conditions:
        print(f"Running benchmark: variant={variant_name}, device={device}")
        row = run_condition(
            variant_name=variant_name,
            device=device,
            seed=seed,
            total_timesteps=total_timesteps,
            output_directory=output,
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(runs_path, index=False)
        print(
            f"status={row['status']}, elapsed_seconds={row['elapsed_seconds']:.3f}, "
            f"transitions_per_second={row['transitions_per_second']:.3f}"
        )

        if device == "cpu" and row["status"] != "PASS":
            raise RuntimeError(
                f"Required CPU benchmark failed; see {row['log']}: "
                f"{row['failure_reason']}"
            )

    runs = pd.DataFrame(rows)
    decision = {
        "benchmark_schema_version": BENCHMARK_SCHEMA,
        **select_training_device(runs),
        "seed": seed,
        "total_timesteps": total_timesteps,
        "benchmark_runs": str(runs_path),
        "benchmark_environment": str(environment_path),
    }
    decision_path.write_text(
        json.dumps(decision, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_lines = [
        "status=PASS",
        f"benchmark_schema_version={BENCHMARK_SCHEMA}",
        f"selected_device={decision['selected_device']}",
        f"selection_reason={decision['selection_reason']}",
        f"cpu_weighted_seconds={decision['cpu_weighted_seconds']}",
        f"cuda_weighted_seconds={decision['cuda_weighted_seconds']}",
        f"cuda_speedup_fraction={decision['cuda_speedup_fraction']}",
        f"cuda_benchmark_complete={decision['cuda_benchmark_complete']}",
        f"seed={seed}",
        f"total_timesteps={total_timesteps}",
        f"benchmark_runs={runs_path}",
        f"benchmark_decision={decision_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "environment": environment_path,
        "runs": runs_path,
        "decision": decision_path,
        "summary": summary_path,
    }

#} End function benchmark_training_devices

# end region Execution


#################################################################################
# region Command line

# Parse one reproducible benchmark configuration.
def parse_args() -> argparse.Namespace:
#{
    parser = argparse.ArgumentParser(
        description="Benchmark CPU and CUDA throughput for final PPO training."
    )
    parser.add_argument("--seed", type=int, default=9902)
    parser.add_argument("--total-timesteps", type=int, default=10240)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()

#} End function parse_args


# Run the requested benchmark.
def main() -> int:
#{
    args = parse_args()
    outputs = benchmark_training_devices(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        output_directory=args.output_directory,
        replace=args.replace,
    )
    print("Training-device benchmark completed successfully.")

    for name, path in outputs.items():
        print(f"{name}: {path}")

    return 0

#} End function main

# end region Command line


if __name__ == "__main__":
    raise SystemExit(main())
