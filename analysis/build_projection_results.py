from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from analysis.aggregate_projection_results import build_result_tables
from analysis.plot_projection_results import build_result_figures


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


#################################################################################
# region Helpers

# Resolve one path relative to the repository root.
def repository_path(path: Path) -> Path:
#{
    return path if path.is_absolute() else REPOSITORY_ROOT / path
#} End function repository_path


# Write the PASS marker only after tables and figures are complete.
def write_summary(
    summary_path: Path,
    protocol: Path,
    evaluation_directory: Path,
    output_directory: Path,
    table_outputs: dict[str, Path],
    figure_outputs: dict[str, Path],
) -> None:
#{
    lines = [
        "status=PASS",
        f"completed_at={datetime.now(timezone.utc).isoformat()}",
        f"protocol={protocol}",
        f"evaluation_dir={evaluation_directory}",
        f"output_dir={output_directory}",
        f"table_output_count={len(table_outputs)}",
        f"figure_output_count={len(figure_outputs)}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
#} End function write_summary

# end region Helpers


#################################################################################
# region Build

# Build audited tables and figures from saved experiment artifacts.
def build_results(
    protocol: Path,
    evaluation_directory: Path,
    output_directory: Path,
    runs_directory: Path | None,
) -> tuple[dict[str, Path], dict[str, Path], Path]:
#{
    protocol_path = repository_path(protocol).resolve()
    evaluation_path = repository_path(evaluation_directory).resolve()
    output_path = repository_path(output_directory).resolve()
    runs_path = repository_path(runs_directory).resolve() if runs_directory is not None else None

    if output_path.exists():
        raise FileExistsError(
            f"Result output directory already exists: {output_path}. "
            "Choose a new output directory or remove the old build explicitly."
        )

    tables_directory = output_path / "tables"
    figures_directory = output_path / "figures"
    table_outputs = build_result_tables(
        protocol_path,
        evaluation_path,
        tables_directory,
    )
    figure_outputs = build_result_figures(
        protocol_path,
        tables_directory,
        evaluation_path,
        figures_directory,
        runs_path,
    )
    summary_path = output_path / "result_build_summary.txt"
    write_summary(
        summary_path=summary_path,
        protocol=protocol_path,
        evaluation_directory=evaluation_path,
        output_directory=output_path,
        table_outputs=table_outputs,
        figure_outputs=figure_outputs,
    )
    print("Result build completed successfully.")
    print(f"Tables:  {tables_directory}")
    print(f"Figures: {figures_directory}")
    print(f"Summary: {summary_path}")
    return table_outputs, figure_outputs, summary_path
#} End function build_results

# end region Build


#################################################################################
# region Command line

# Parse one result-build configuration.
def parse_args() -> argparse.Namespace:
#{
    parser = argparse.ArgumentParser(
        description="Build audited result tables and figures from saved artifacts."
    )
    parser.add_argument("protocol", type=Path)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("runs_dir", type=Path, nargs="?", default=None)
    return parser.parse_args()
#} End function parse_args


# Run one complete result build.
def main() -> int:
#{
    args = parse_args()
    build_results(
        protocol=args.protocol,
        evaluation_directory=args.evaluation_dir,
        output_directory=args.output_dir,
        runs_directory=args.runs_dir,
    )
    return 0
#} End function main

# end region Command line


if __name__ == "__main__":
    raise SystemExit(main())
