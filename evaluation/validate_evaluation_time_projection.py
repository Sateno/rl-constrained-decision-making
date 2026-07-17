from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cvxpy as cp
import gymnasium
import numpy as np
import osqp
import pandas as pd
import torch

from evaluation.evaluate_projection_pair import make_output_paths
from projection.cbf_qp_projection import ProjectionParams, project_physical_action


#################################################################################
# region Configuration

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONDA_ENVIRONMENT = "RL_PROJECTS"
EXPECTED_TEST_COUNT = 19
EXPECTED_CHECKPOINT_SHA256 = (
    "3c06bd19ee42914aef49f049de88c165190f745ca1c4cdbb3ac23bb7497da1c3"
)
EXPECTED_TRAJECTORY_ARCHIVE_VERSION = "evaluation_trajectory_v1"
RUNTIME_CALL_COUNT = 300
RUNTIME_GREEN_LIMIT_MILLISECONDS = 20.0

CHECKPOINT_PATH = Path("runs/checkpoints/ppo_baseline_51200_seed1.pt")
VALIDATION_DIRECTORY = Path("runs/validation")
EVALUATION_DIRECTORY = Path("runs/evaluation")
MASTER_LOG_PATH = VALIDATION_DIRECTORY / "evaluation_time_projection_validation.log"
SUMMARY_PATH = VALIDATION_DIRECTORY / "evaluation_time_projection_validation_summary.txt"
ENVIRONMENT_PATH = VALIDATION_DIRECTORY / "projection_environment.txt"
CONDA_LIST_PATH = VALIDATION_DIRECTORY / "projection_conda_list.txt"
PYTEST_LOG_PATH = VALIDATION_DIRECTORY / "projection_pytest.txt"
CAPACITY_SMOKE_PATH = EVALUATION_DIRECTORY / "checkpoint_capacity3_active2_smoke.csv"
CAPACITY_MISMATCH_PATH = (
    EVALUATION_DIRECTORY / "checkpoint_capacity_mismatch_should_not_exist.csv"
)
RUNTIME_PATH = VALIDATION_DIRECTORY / "projection_runtime.txt"
DETERMINISTIC_PREFIX = (
    EVALUATION_DIRECTORY / "ppo_baseline_51200_seed1_projection_pair"
)
STOCHASTIC_PREFIX = (
    EVALUATION_DIRECTORY / "ppo_baseline_51200_seed1_projection_pair_stochastic"
)

# end region Configuration


#################################################################################
# region Logging and process helpers

# Return a local timestamp suitable for validation records.
def current_timestamp() -> str:
#{
    return datetime.now().astimezone().isoformat(timespec="seconds")
#} End function current_timestamp


# Append one line to the console and the master validation log.
def log(message: str = "") -> None:
#{
    print(message, flush=True)

    with MASTER_LOG_PATH.open("a", encoding="utf-8", newline="\n") as log_file:
        log_file.write(f"{message}\n")
#} End function log


# Append command output without adding an extra newline.
def log_command_output(output: str) -> None:
#{
    print(output, end="", flush=True)

    with MASTER_LOG_PATH.open("a", encoding="utf-8", newline="\n") as log_file:
        log_file.write(output)
#} End function log_command_output


# Fail one validation condition with a direct diagnostic.
def require(condition: bool, message: str) -> None:
#{
    if not condition:
        raise RuntimeError(message)
#} End function require


# Run one command, stream its combined output, and enforce its exit behavior.
def run_command(
    *,
    label: str,
    command: list[str],
    expect_failure: bool = False,
) -> str:
#{
    command_text = subprocess.list2cmdline(command)
    log("")
    log(label)
    log("-" * len(label))
    log(f"> {command_text}")

    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output_parts: list[str] = []

    if process.stdout is None:
        raise RuntimeError(f"Could not capture output for command: {command_text}")

    for line in process.stdout:
        output_parts.append(line)
        log_command_output(line)

    return_code = process.wait()
    output = "".join(output_parts)

    if expect_failure:
        require(
            return_code != 0,
            f"Expected command failure, but the command exited with code 0: {command_text}",
        )
        log(f"[PASS] Expected failure occurred with exit code {return_code}.")

    else:
        require(
            return_code == 0,
            f"Command failed with exit code {return_code}: {command_text}",
        )
        log("[PASS]")

    return output
#} End function run_command


# Remove all files derived from one paired-evaluation output prefix.
def remove_paired_artifacts(prefix: Path) -> None:
#{
    for path in prefix.parent.glob(f"{prefix.name}_*"):
        if path.is_file():
            path.unlink()
#} End function remove_paired_artifacts


# Return the canonical paired-evaluation command for one policy mode.
def paired_evaluation_command(*, output_prefix: Path, stochastic: bool) -> list[str]:
#{
    command = [
        sys.executable,
        "-m",
        "evaluation.evaluate_projection_pair",
        "--checkpoint",
        str(CHECKPOINT_PATH),
        "--episodes",
        "20",
        "--seed",
        "1000",
        "--max-episode-steps",
        "200",
        "--max-obstacles",
        "3",
        "--num-active-obstacles",
        "3",
        "--projection-lookahead-distance",
        "0.25",
        "--projection-alpha",
        "2.0",
        "--projection-slack-penalty",
        "1000.0",
        "--projection-extra-clearance",
        "0.0",
        "--no-cuda",
        "--output-prefix",
        str(output_prefix),
    ]

    if stochastic:
        command.insert(-3, "--stochastic")

    return command
#} End function paired_evaluation_command

# end region Logging and process helpers


#################################################################################
# region Environment and checkpoint validation

# Verify the expected Conda environment and save package evidence.
def validate_environment() -> None:
#{
    active_environment = os.environ.get("CONDA_DEFAULT_ENV", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    prefix_name = Path(conda_prefix).name if conda_prefix else ""
    environment_matches = (
        active_environment.casefold() == REQUIRED_CONDA_ENVIRONMENT.casefold()
        or prefix_name.casefold() == REQUIRED_CONDA_ENVIRONMENT.casefold()
    )

    require(
        environment_matches,
        (
            f"Activate the {REQUIRED_CONDA_ENVIRONMENT} Conda environment before validation. "
            f"CONDA_DEFAULT_ENV={active_environment!r}, CONDA_PREFIX={conda_prefix!r}."
        ),
    )

    installed_solvers = cp.installed_solvers()
    require("OSQP" in installed_solvers, "CVXPY does not report OSQP as an installed solver.")

    environment_lines = [
        f"validation_timestamp={current_timestamp()}",
        f"repository_root={REPOSITORY_ROOT}",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(os.linesep, ' ')}",
        f"platform={platform.platform()}",
        f"conda_default_env={active_environment}",
        f"conda_prefix={conda_prefix}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"torch={torch.__version__}",
        f"gymnasium={gymnasium.__version__}",
        f"cvxpy={cp.__version__}",
        f"osqp={osqp.__version__}",
        f"installed_solvers={installed_solvers}",
    ]
    ENVIRONMENT_PATH.write_text(
        "\n".join(environment_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    conda_executable = shutil.which("conda")
    require(conda_executable is not None, "The conda executable is not available on PATH.")

    with CONDA_LIST_PATH.open("w", encoding="utf-8", newline="\n") as conda_file:
        result = subprocess.run(
            [conda_executable, "list"],
            cwd=REPOSITORY_ROOT,
            stdout=conda_file,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    require(
        result.returncode == 0,
        f"conda list failed with exit code {result.returncode}: {result.stderr}",
    )

    log("")
    log("Environment and solver validation")
    log("---------------------------------")

    for line in environment_lines:
        log(line)

    log(f"conda_package_list={CONDA_LIST_PATH}")
    log("[PASS]")
#} End function validate_environment


# Compute a SHA-256 digest without loading the whole file into memory.
def file_sha256(path: Path) -> str:
#{
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()
#} End function file_sha256


# Verify the established checkpoint exists and has the expected identity.
def validate_checkpoint() -> str:
#{
    require(CHECKPOINT_PATH.is_file(), f"Checkpoint not found: {CHECKPOINT_PATH}")
    checkpoint_sha256 = file_sha256(CHECKPOINT_PATH)
    require(
        checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256,
        (
            f"Checkpoint SHA-256 mismatch. Expected {EXPECTED_CHECKPOINT_SHA256}, "
            f"received {checkpoint_sha256}."
        ),
    )

    log("")
    log("Checkpoint identity")
    log("-------------------")
    log(f"checkpoint={CHECKPOINT_PATH}")
    log(f"sha256={checkpoint_sha256}")
    log("[PASS]")

    return checkpoint_sha256
#} End function validate_checkpoint

# end region Environment and checkpoint validation


#################################################################################
# region Regression and compatibility validation

# Compile the active Python source and run the complete lightweight suite.
def validate_compilation_and_tests() -> None:
#{
    run_command(
        label="Active source compilation",
        command=[
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "environments",
            "projection",
            "evaluation",
            "algorithms",
            "tests",
        ],
    )

    pytest_output = run_command(
        label="Complete lightweight regression suite",
        command=[sys.executable, "-m", "pytest", "-q", "-rs"],
    )
    PYTEST_LOG_PATH.write_text(
        pytest_output,
        encoding="utf-8",
        newline="\n",
    )

    summary_lines = [
        line.strip()
        for line in pytest_output.splitlines()
        if " passed" in line and " in " in line
    ]
    require(summary_lines, "Could not find the pytest completion summary.")
    pytest_summary = summary_lines[-1]
    require(
        re.search(rf"\b{EXPECTED_TEST_COUNT} passed\b", pytest_summary) is not None,
        (
            f"Expected exactly {EXPECTED_TEST_COUNT} passing tests, "
            f"but the pytest summary was: {pytest_summary}"
        ),
    )
    require("failed" not in pytest_summary, f"Pytest reported a failure: {pytest_summary}")
    require("error" not in pytest_summary, f"Pytest reported an error: {pytest_summary}")
    require("skipped" not in pytest_summary, f"Pytest reported a skip: {pytest_summary}")
#} End function validate_compilation_and_tests


# Verify active obstacle count does not change checkpoint observation compatibility.
def validate_active_obstacle_compatibility() -> None:
#{
    CAPACITY_SMOKE_PATH.unlink(missing_ok=True)

    run_command(
        label="Checkpoint compatibility with two active obstacles",
        command=[
            sys.executable,
            "-m",
            "evaluation.evaluate_policy",
            "--policy",
            "ppo",
            "--checkpoint",
            str(CHECKPOINT_PATH),
            "--episodes",
            "1",
            "--seed",
            "1000",
            "--max-episode-steps",
            "200",
            "--max-obstacles",
            "3",
            "--num-active-obstacles",
            "2",
            "--no-cuda",
            "--output",
            str(CAPACITY_SMOKE_PATH),
        ],
    )

    require(CAPACITY_SMOKE_PATH.is_file(), f"Expected output was not created: {CAPACITY_SMOKE_PATH}")
    smoke_results = pd.read_csv(CAPACITY_SMOKE_PATH)
    require(len(smoke_results) == 1, "Capacity compatibility smoke output must contain one row.")
    require(int(smoke_results.iloc[0]["seed"]) == 1000, "Capacity compatibility smoke seed mismatch.")
#} End function validate_active_obstacle_compatibility


# Verify an incompatible obstacle capacity is rejected before an output CSV is written.
def validate_capacity_mismatch_rejection() -> None:
#{
    CAPACITY_MISMATCH_PATH.unlink(missing_ok=True)

    output = run_command(
        label="Expected checkpoint rejection for five-obstacle capacity",
        command=[
            sys.executable,
            "-m",
            "evaluation.evaluate_policy",
            "--policy",
            "ppo",
            "--checkpoint",
            str(CHECKPOINT_PATH),
            "--episodes",
            "1",
            "--seed",
            "1000",
            "--max-episode-steps",
            "200",
            "--max-obstacles",
            "5",
            "--num-active-obstacles",
            "2",
            "--no-cuda",
            "--output",
            str(CAPACITY_MISMATCH_PATH),
        ],
        expect_failure=True,
    )

    expected_diagnostic = "Checkpoint obs_dim=21 does not match environment obs_dim=31."
    require(expected_diagnostic in output, "Expected checkpoint-dimension diagnostic was not reported.")
    require(
        not CAPACITY_MISMATCH_PATH.exists(),
        f"Incompatible evaluation unexpectedly created: {CAPACITY_MISMATCH_PATH}",
    )
#} End function validate_capacity_mismatch_rejection

# end region Regression and compatibility validation


#################################################################################
# region Evaluation artifact audits

# Convert one CSV Boolean column into a validated NumPy Boolean array.
def csv_boolean_values(series: pd.Series, column_name: str) -> np.ndarray:
#{
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool)

    normalized = series.astype(str).str.strip().str.casefold()
    invalid_values = set(normalized.unique()) - {"true", "false"}
    require(
        not invalid_values,
        f"CSV column {column_name} contains invalid Boolean values: {sorted(invalid_values)}",
    )

    return normalized.eq("true").to_numpy(dtype=bool)
#} End function csv_boolean_values


# Verify deterministic paired CSV alignment, identity, and noninterference.
def audit_deterministic_csv_artifacts(checkpoint_sha256: str) -> None:
#{
    paths = make_output_paths(DETERMINISTIC_PREFIX)
    projection_disabled = pd.read_csv(paths["projection_disabled"])
    projection_enabled = pd.read_csv(paths["projection_enabled"])
    paired_episodes = pd.read_csv(paths["paired_episodes"])
    paired_summary = pd.read_csv(paths["paired_summary"])

    require(len(paired_summary) == 1, "Deterministic paired summary must contain one row.")
    summary = paired_summary.iloc[0]
    expected_seeds = np.arange(1000, 1020, dtype=np.int64)

    require(
        len(projection_disabled) == len(projection_enabled) == len(paired_episodes) == 20,
        "Deterministic paired outputs must contain twenty aligned episodes.",
    )
    np.testing.assert_array_equal(projection_disabled["seed"].to_numpy(), expected_seeds)
    np.testing.assert_array_equal(projection_enabled["seed"].to_numpy(), expected_seeds)
    require(
        projection_disabled[["episode", "seed"]].equals(
            projection_enabled[["episode", "seed"]]
        ),
        "Deterministic projection-disabled and projection-enabled episode keys differ.",
    )
    require(
        paired_episodes[["episode", "seed"]].equals(
            projection_disabled[["episode", "seed"]]
        ),
        "Deterministic paired episode keys do not match the raw mode outputs.",
    )

    disabled_flags = csv_boolean_values(
        projection_disabled["projection_enabled"],
        "projection_enabled",
    )
    enabled_flags = csv_boolean_values(
        projection_enabled["projection_enabled"],
        "projection_enabled",
    )
    require(not disabled_flags.any(), "Projection-disabled rows contain an enabled projection flag.")
    require(enabled_flags.all(), "Projection-enabled rows contain a disabled projection flag.")

    np.testing.assert_allclose(
        projection_disabled["episode_return"],
        projection_enabled["episode_return"],
        rtol=0.0,
        atol=1.0e-5,
    )
    require(
        csv_boolean_values(projection_disabled["success"], "success").all()
        and csv_boolean_values(projection_enabled["success"], "success").all(),
        "Every deterministic episode must reach the goal in both modes.",
    )
    require(
        not csv_boolean_values(projection_disabled["collision"], "collision").any()
        and not csv_boolean_values(projection_enabled["collision"], "collision").any(),
        "Deterministic evaluation must remain collision-free in both modes.",
    )
    require(
        int(projection_enabled["projection_intervention_count"].sum()) == 0,
        "The safe deterministic policy should not require projection intervention.",
    )
    require(
        int(projection_enabled["projection_solver_failure_count"].sum()) == 0,
        "Deterministic projection evaluation reported a solver failure.",
    )
    require(
        float(projection_enabled["max_projection_correction_norm"].max()) <= 1.0e-6,
        "Deterministic projection correction exceeded the noninterference tolerance.",
    )
    require(
        str(summary["checkpoint_sha256"]) == checkpoint_sha256,
        "Deterministic paired summary checkpoint SHA-256 mismatch.",
    )
    require(
        abs(float(summary["without_projection_mean_return"]) - 12.319942) < 1.0e-3,
        "Projection-disabled deterministic return does not match the established reference.",
    )
    require(
        abs(float(summary["with_projection_mean_return"]) - 12.319942) < 1.0e-3,
        "Projection-enabled deterministic return does not match the established reference.",
    )
    require(
        float(summary["without_projection_success_rate"]) == 1.0
        and float(summary["with_projection_success_rate"]) == 1.0,
        "Deterministic success rate must remain 1.0 in both modes.",
    )
    require(
        float(summary["without_projection_collision_rate"]) == 0.0
        and float(summary["with_projection_collision_rate"]) == 0.0,
        "Deterministic collision rate must remain 0.0 in both modes.",
    )

    log("")
    log("Deterministic paired CSV audit")
    log("------------------------------")
    log(summary.to_string())
    log("[PASS]")
#} End function audit_deterministic_csv_artifacts


# Verify NPZ schema, state/action alignment, and raw/executed action identities.
def audit_deterministic_trajectory_artifacts() -> None:
#{
    paths = make_output_paths(DETERMINISTIC_PREFIX)

    with np.load(paths["projection_disabled_trajectories"], allow_pickle=False) as disabled:
        with np.load(paths["projection_enabled_trajectories"], allow_pickle=False) as enabled:
            disabled_keys = disabled["episode_keys"].tolist()
            enabled_keys = enabled["episode_keys"].tolist()

            require(
                disabled["trajectory_archive_version"].item()
                == EXPECTED_TRAJECTORY_ARCHIVE_VERSION,
                "Projection-disabled trajectory archive version mismatch.",
            )
            require(
                enabled["trajectory_archive_version"].item()
                == EXPECTED_TRAJECTORY_ARCHIVE_VERSION,
                "Projection-enabled trajectory archive version mismatch.",
            )
            require(
                int(disabled["episode_count"]) == int(enabled["episode_count"]) == 20,
                "Deterministic trajectory archives must contain twenty episodes.",
            )
            require(
                disabled_keys == enabled_keys,
                "Deterministic trajectory archive episode keys differ between modes.",
            )
            require(
                all(not disabled[name].dtype.hasobject for name in disabled.files),
                "Projection-disabled trajectory archive contains an object array.",
            )
            require(
                all(not enabled[name].dtype.hasobject for name in enabled.files),
                "Projection-enabled trajectory archive contains an object array.",
            )

            disabled_seeds = [int(disabled[f"{key}_seed"]) for key in disabled_keys]
            enabled_seeds = [int(enabled[f"{key}_seed"]) for key in enabled_keys]
            require(
                disabled_seeds == enabled_seeds == list(range(1000, 1020)),
                "Deterministic trajectory archive seeds are not aligned.",
            )

            for key in disabled_keys:
                disabled_transition_count = disabled[f"{key}_action_raw_physical"].shape[0]
                enabled_transition_count = enabled[f"{key}_action_exec_physical"].shape[0]

                require(
                    disabled[f"{key}_positions"].shape[0] == disabled_transition_count + 1,
                    f"Projection-disabled trajectory {key} violates the T + 1 state contract.",
                )
                require(
                    enabled[f"{key}_positions"].shape[0] == enabled_transition_count + 1,
                    f"Projection-enabled trajectory {key} violates the T + 1 state contract.",
                )
                require(
                    np.array_equal(
                        disabled[f"{key}_action_raw_physical"],
                        disabled[f"{key}_action_exec_physical"],
                    ),
                    f"Projection-disabled trajectory {key} changed the physical action.",
                )
                np.testing.assert_allclose(
                    enabled[f"{key}_action_exec_physical"]
                    - enabled[f"{key}_action_raw_physical"],
                    enabled[f"{key}_action_correction_physical"],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                require(
                    set(disabled[f"{key}_projection_solver_status"].tolist()) == {"disabled"},
                    f"Projection-disabled trajectory {key} has an invalid solver status.",
                )
                require(
                    set(enabled[f"{key}_projection_solver_status"].tolist()).issubset(
                        {"optimal", "optimal_inaccurate"}
                    ),
                    f"Projection-enabled trajectory {key} has an invalid solver status.",
                )
                require(
                    bool(enabled[f"{key}_projection_success"].all()),
                    f"Projection-enabled trajectory {key} contains a failed solve.",
                )

    log("")
    log("Deterministic trajectory and action audit")
    log("-----------------------------------------")
    log("Twenty aligned trajectory pairs passed schema, state, action, and solver checks.")
    log("[PASS]")
#} End function audit_deterministic_trajectory_artifacts


# Verify stochastic evaluation exercises intervention without solver failure.
def audit_stochastic_artifacts() -> None:
#{
    paths = make_output_paths(STOCHASTIC_PREFIX)
    projection_disabled = pd.read_csv(paths["projection_disabled"])
    projection_enabled = pd.read_csv(paths["projection_enabled"])
    paired_summary = pd.read_csv(paths["paired_summary"])

    require(len(paired_summary) == 1, "Stochastic paired summary must contain one row.")
    summary = paired_summary.iloc[0]
    require(
        projection_disabled[["episode", "seed"]].equals(
            projection_enabled[["episode", "seed"]]
        ),
        "Stochastic projection-disabled and projection-enabled episode keys differ.",
    )
    require(
        int(summary["with_projection_total_interventions"]) > 0,
        "Stochastic projection evaluation did not produce an intervention.",
    )
    require(
        float(summary["with_projection_mean_intervention_rate"]) > 0.0,
        "Stochastic projection mean intervention rate must be positive.",
    )
    require(
        float(summary["with_projection_mean_correction_norm"]) > 0.0,
        "Stochastic projection mean correction norm must be positive.",
    )
    require(
        int(summary["with_projection_total_solver_failures"]) == 0,
        "Stochastic projection evaluation reported a solver failure.",
    )
    require(
        np.isfinite(float(summary["with_projection_mean_slack_sum"])),
        "Stochastic mean projection slack is not finite.",
    )
    require(
        np.isfinite(float(summary["with_projection_max_slack"])),
        "Stochastic maximum projection slack is not finite.",
    )

    with np.load(paths["projection_enabled_trajectories"], allow_pickle=False) as trajectories:
        keys = trajectories["episode_keys"].tolist()
        require(
            trajectories["trajectory_archive_version"].item()
            == EXPECTED_TRAJECTORY_ARCHIVE_VERSION,
            "Stochastic trajectory archive version mismatch.",
        )
        require(
            any(trajectories[f"{key}_projection_intervened"].any() for key in keys),
            "Stochastic trajectory archive contains no intervened transition.",
        )

        for key in keys:
            np.testing.assert_allclose(
                trajectories[f"{key}_action_exec_physical"]
                - trajectories[f"{key}_action_raw_physical"],
                trajectories[f"{key}_action_correction_physical"],
                rtol=0.0,
                atol=1.0e-12,
            )
            require(
                bool(trajectories[f"{key}_projection_success"].all()),
                f"Stochastic trajectory {key} contains a failed projection solve.",
            )

    log("")
    log("Active projection diagnostic audit")
    log("----------------------------------")
    log(summary.to_string())
    log("[PASS]")
#} End function audit_stochastic_artifacts

# end region Evaluation artifact audits


#################################################################################
# region Runtime benchmark and final record

# Measure repeated active three-obstacle projection cost and enforce the green gate.
def validate_projection_runtime() -> tuple[float, float]:
#{
    params = ProjectionParams()
    projection_arguments = {
        "position": np.array([0.0, 0.0]),
        "heading": 0.0,
        "obstacle_centers": np.array(
            [
                [0.75, 0.0],
                [1.4, 0.6],
                [2.0, -0.5],
            ]
        ),
        "obstacle_radii": np.array([0.25, 0.30, 0.25]),
        "obstacle_mask": np.array([True, True, True]),
        "agent_radius": 0.10,
        "raw_action": np.array([1.0, 0.0]),
        "params": params,
    }

    for _ in range(10):
        project_physical_action(**projection_arguments)

    start_time = perf_counter()
    results = [
        project_physical_action(**projection_arguments)
        for _ in range(RUNTIME_CALL_COUNT)
    ]
    elapsed_seconds = perf_counter() - start_time
    milliseconds_per_call = 1000.0 * elapsed_seconds / RUNTIME_CALL_COUNT
    calls_per_second = RUNTIME_CALL_COUNT / elapsed_seconds

    require(
        all(result.success for result in results),
        "At least one projection runtime benchmark call failed.",
    )

    runtime_lines = [
        f"validation_timestamp={current_timestamp()}",
        f"calls={RUNTIME_CALL_COUNT}",
        f"elapsed_seconds={elapsed_seconds}",
        f"milliseconds_per_call={milliseconds_per_call}",
        f"calls_per_second={calls_per_second}",
        f"last_solver_status={results[-1].solver_status}",
        f"green_limit_milliseconds={RUNTIME_GREEN_LIMIT_MILLISECONDS}",
    ]
    RUNTIME_PATH.write_text(
        "\n".join(runtime_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    log("")
    log("Projection runtime benchmark")
    log("----------------------------")

    for line in runtime_lines:
        log(line)

    require(
        milliseconds_per_call < RUNTIME_GREEN_LIMIT_MILLISECONDS,
        (
            f"Projection runtime {milliseconds_per_call:.3f} ms/call exceeded the "
            f"{RUNTIME_GREEN_LIMIT_MILLISECONDS:.1f} ms green gate."
        ),
    )
    log("[PASS]")

    return milliseconds_per_call, calls_per_second
#} End function validate_projection_runtime


# Write a concise machine-readable text record only after every validation passes.
def write_pass_summary(
    *,
    checkpoint_sha256: str,
    milliseconds_per_call: float,
    calls_per_second: float,
    started_at: str,
) -> None:
#{
    summary_lines = [
        "status=PASS",
        f"started_at={started_at}",
        f"completed_at={current_timestamp()}",
        f"repository_root={REPOSITORY_ROOT}",
        f"conda_environment={REQUIRED_CONDA_ENVIRONMENT}",
        f"python_executable={sys.executable}",
        f"checkpoint={CHECKPOINT_PATH}",
        f"checkpoint_sha256={checkpoint_sha256}",
        f"pytest_passed={EXPECTED_TEST_COUNT}",
        f"trajectory_archive_version={EXPECTED_TRAJECTORY_ARCHIVE_VERSION}",
        f"deterministic_output_prefix={DETERMINISTIC_PREFIX}",
        f"stochastic_output_prefix={STOCHASTIC_PREFIX}",
        f"runtime_calls={RUNTIME_CALL_COUNT}",
        f"runtime_milliseconds_per_call={milliseconds_per_call}",
        f"runtime_calls_per_second={calls_per_second}",
        f"master_log={MASTER_LOG_PATH}",
        f"environment_record={ENVIRONMENT_PATH}",
        f"conda_package_list={CONDA_LIST_PATH}",
        f"runtime_record={RUNTIME_PATH}",
    ]
    SUMMARY_PATH.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
#} End function write_pass_summary

# end region Runtime benchmark and final record


#################################################################################
# region Main

# Run every canonical runtime validation in dependency order.
def main() -> int:
#{
    os.chdir(REPOSITORY_ROOT)
    VALIDATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    MASTER_LOG_PATH.unlink(missing_ok=True)
    SUMMARY_PATH.unlink(missing_ok=True)
    started_at = current_timestamp()

    log("Evaluation-Time Predictive Action Projection Runtime Validation")
    log("=============================================================")
    log(f"started_at={started_at}")
    log(f"repository_root={REPOSITORY_ROOT}")

    try:
    #{
        validate_environment()
        checkpoint_sha256 = validate_checkpoint()
        validate_compilation_and_tests()
        validate_active_obstacle_compatibility()
        validate_capacity_mismatch_rejection()

        remove_paired_artifacts(DETERMINISTIC_PREFIX)
        run_command(
            label="Canonical deterministic paired evaluation",
            command=paired_evaluation_command(
                output_prefix=DETERMINISTIC_PREFIX,
                stochastic=False,
            ),
        )
        audit_deterministic_csv_artifacts(checkpoint_sha256)
        audit_deterministic_trajectory_artifacts()

        remove_paired_artifacts(STOCHASTIC_PREFIX)
        run_command(
            label="Mandatory active projection diagnostic",
            command=paired_evaluation_command(
                output_prefix=STOCHASTIC_PREFIX,
                stochastic=True,
            ),
        )
        audit_stochastic_artifacts()

        milliseconds_per_call, calls_per_second = validate_projection_runtime()
        write_pass_summary(
            checkpoint_sha256=checkpoint_sha256,
            milliseconds_per_call=milliseconds_per_call,
            calls_per_second=calls_per_second,
            started_at=started_at,
        )

    #} End try
    except Exception as error:
    #{
        log("")
        log("VALIDATION FAILED")
        log("=================")
        log(str(error))
        log("")
        log(traceback.format_exc().rstrip())
        log(f"master_log={MASTER_LOG_PATH}")
        return 1

    #} End except validation_failure

    log("")
    log("VALIDATION PASSED")
    log("=================")
    log(f"summary={SUMMARY_PATH}")
    log(f"master_log={MASTER_LOG_PATH}")
    log("All canonical runtime checks completed successfully.")

    return 0
#} End function main

# end region Main


if __name__ == "__main__":
    raise SystemExit(main())
