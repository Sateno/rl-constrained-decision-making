from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_ROOT = Path("runs/validation/pre_experiment_codebase")


#################################################################################
# region Helpers

# Resolve one path relative to the repository root.
def repository_path(path: Path) -> Path:
#{
    return path if path.is_absolute() else REPOSITORY_ROOT / path
#} End function repository_path


# Open one directory with the platform file browser.
def open_directory(path: Path) -> None:
#{
    if sys.platform == "win32":
        os.startfile(str(path))
        return
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
        return
    subprocess.run(["xdg-open", str(path)], check=False)
#} End function open_directory


# Print one text file under a stable heading.
def print_file(label: str, path: Path) -> None:
#{
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")

    print()
    print(label)
    print("-" * len(label))
    print(path.read_text(encoding="utf-8").rstrip())
#} End function print_file

# end region Helpers


#################################################################################
# region Inspection

# Inspect the automated validation summary and generated figures.
def inspect_validation(validation_root: Path, open_figures: bool) -> None:
#{
    root = repository_path(validation_root).resolve()
    summary = root / "pre_experiment_validation_summary.txt"
    manual_review = root / "manual_review.txt"
    figures = root / "result_build" / "figures"
    tables = root / "result_build" / "tables"
    print_file("Automated validation summary", summary)
    print_file("Manual review", manual_review)

    if not figures.is_dir():
        raise FileNotFoundError(f"Generated figures directory not found: {figures}")

    pdfs = sorted(figures.glob("*.pdf"))

    if not pdfs:
        raise RuntimeError(f"No generated PDF figures were found under: {figures}")

    print()
    print("Generated PDFs")
    print("--------------")

    for path in pdfs:
        print(path.name)

    print()
    print(f"Method table: {tables / 'method_summary.csv'}")
    print(f"Figures:      {figures}")

    if open_figures:
        open_directory(figures)
#} End function inspect_validation

# end region Inspection


#################################################################################
# region Command line

# Parse the inspection command.
def parse_args() -> argparse.Namespace:
#{
    parser = argparse.ArgumentParser(
        description="Inspect the completed pre-experiment validation artifacts."
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=DEFAULT_VALIDATION_ROOT,
    )
    parser.add_argument("--open", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()
#} End function parse_args


# Run the validation inspection.
def main() -> int:
#{
    args = parse_args()
    inspect_validation(args.validation_root, args.open)
    return 0
#} End function main

# end region Command line


if __name__ == "__main__":
    raise SystemExit(main())
