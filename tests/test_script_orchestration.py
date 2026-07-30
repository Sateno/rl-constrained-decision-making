from pathlib import Path

import pytest

from analysis import build_projection_results
from evaluation import validate_pre_experiment_codebase
from experiments import calibrate_core_layouts
from experiments.train_ppo_variant import TRAINING_VARIANTS, training_command


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BATCH_FILES = {
    "calibrate_core_layouts.bat",
    "evaluate_policy.bat",
    "evaluate_projection_pair.bat",
    "inspect_pre_experiment_validation.bat",
    "plot_projection_results.bat",
    "run_ppo_baseline_clean.bat",
    "train_ppo_baseline.bat",
    "train_ppo_high_penalty.bat",
    "train_ppo_projection.bat",
    "validate_evaluation_time_projection.bat",
    "validate_pre_experiment_codebase.bat",
}


#################################################################################
# region Helpers

# Return the value following one command-line option.
def option_value(command: list[str], option: str) -> str:
#{
    index = command.index(option)
    return command[index + 1]
#} End function option_value

# end region Helpers


#################################################################################
# region Checks

# Verify every Windows batch file remains a thin Python launcher.
def test_batch_files_are_thin_launchers() -> None:
#{
    scripts_directory = REPOSITORY_ROOT / "scripts"
    batch_files = {path.name for path in scripts_directory.glob("*.bat")}
    assert batch_files == EXPECTED_BATCH_FILES
    forbidden = (
        "rmdir ",
        "mkdir ",
        "for /",
        "if exist",
        "python -c",
        "powershell",
        "call :",
        "type ",
    )

    for filename in sorted(batch_files):
        text = (scripts_directory / filename).read_text(encoding="ascii").casefold()
        assert "python -m " in text
        assert not any(token in text for token in forbidden)
        assert len([line for line in text.splitlines() if line.strip()]) <= 8

#} End function test_batch_files_are_thin_launchers


# Verify the three declared training variants differ only through intended settings.
def test_training_variant_commands_preserve_experiment_contract(tmp_path: Path) -> None:
#{
    commands = {
        name: training_command(
            variant,
            seed=7,
            total_timesteps=4096,
            checkpoint_path=tmp_path / f"{name}.pt",
        )
        for name, variant in TRAINING_VARIANTS.items()
    }
    assert option_value(commands["baseline"], "--method") == "ppo_baseline"
    assert option_value(commands["high_penalty"], "--method") == "ppo_high_penalty"
    assert option_value(commands["projection"], "--method") == "ppo_train_projection"
    assert option_value(commands["baseline"], "--collision-penalty") == "10.0"
    assert option_value(commands["high_penalty"], "--collision-penalty") == "50.0"
    assert option_value(commands["projection"], "--collision-penalty") == "10.0"
    assert "--no-enable-projection" in commands["baseline"]
    assert "--no-enable-projection" in commands["high_penalty"]
    assert "--enable-projection" in commands["projection"]
    assert "--projection-alpha" not in commands["baseline"]
    assert option_value(commands["projection"], "--projection-alpha") == "2.0"

#} End function test_training_variant_commands_preserve_experiment_contract


# Verify the result orchestrator writes PASS only after both builders complete.
def test_result_builder_writes_summary_after_tables_and_figures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
#{
    output_directory = tmp_path / "result_build"

    def fake_tables(protocol, evaluation, output):
    #{
        output.mkdir(parents=True)
        path = output / "method_summary.csv"
        path.write_text("method\nppo\n", encoding="utf-8")
        return {"methods": path}
    #} End function fake_tables

    def fake_figures(protocol, tables, evaluation, figures, runs):
    #{
        figures.mkdir(parents=True)
        path = figures / "evaluation_return.pdf"
        path.write_bytes(b"pdf")
        return {"evaluation_return.pdf": path}
    #} End function fake_figures

    monkeypatch.setattr(build_projection_results, "build_result_tables", fake_tables)
    monkeypatch.setattr(build_projection_results, "build_result_figures", fake_figures)
    _, _, summary_path = build_projection_results.build_results(
        protocol=tmp_path / "protocol.json",
        evaluation_directory=tmp_path / "evaluation",
        output_directory=output_directory,
        runs_directory=None,
    )
    summary = summary_path.read_text(encoding="utf-8")
    assert "status=PASS" in summary
    assert (output_directory / "tables" / "method_summary.csv").is_file()
    assert (output_directory / "figures" / "evaluation_return.pdf").is_file()

#} End function test_result_builder_writes_summary_after_tables_and_figures


# Verify destructive workflow guards reject repository-level paths.
def test_orchestration_cleanup_guards_reject_unsafe_paths() -> None:
#{
    with pytest.raises(ValueError, match="unsafe calibration path"):
        calibrate_core_layouts.validate_cleanup_target(REPOSITORY_ROOT)

    with pytest.raises(ValueError, match="unsafe validation path"):
        validate_pre_experiment_codebase.validate_cleanup_target(REPOSITORY_ROOT)

#} End function test_orchestration_cleanup_guards_reject_unsafe_paths

# end region Checks
