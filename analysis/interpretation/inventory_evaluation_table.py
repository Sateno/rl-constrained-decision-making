"""Create a deterministic schema/value inventory for one supplied episode CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__:
    from ._evaluation_evidence import (
        BOOLEAN_COLUMNS,
        EPISODE_NUMERIC_COLUMNS,
        EvidenceError,
        OUTPUT_SCHEMA,
        implementation_record,
        input_record,
        load_episode_table,
        write_json_exclusive,
    )
else:  # Support invocation by absolute script path from any working directory.
    sys.path.insert(0, str(Path(__file__).parent))
    from _evaluation_evidence import (  # type: ignore[no-redef]
        BOOLEAN_COLUMNS,
        EPISODE_NUMERIC_COLUMNS,
        EvidenceError,
        OUTPUT_SCHEMA,
        implementation_record,
        input_record,
        load_episode_table,
        write_json_exclusive,
    )


def _numeric_inventory(series: pd.Series) -> dict[str, object]:
    values = series.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "finite_count": int(finite.size),
        "maximum": None if not finite.size else float(np.max(finite)),
        "minimum": None if not finite.size else float(np.min(finite)),
        "missing_count": int(np.isnan(values).sum()),
        "negative_infinity_count": int(np.isneginf(values).sum()),
        "positive_infinity_count": int(np.isposinf(values).sum()),
        "zero_count": int(np.count_nonzero(finite == 0.0)),
    }


def _column_inventory(frame: pd.DataFrame, column: str) -> dict[str, object]:
    series = frame[column]
    record: dict[str, object] = {
        "column": column,
        "dtype_after_strict_parsing": str(series.dtype),
        "rows": int(len(series)),
        "unique_nonmissing": int(series.nunique(dropna=True)),
    }
    if column in EPISODE_NUMERIC_COLUMNS:
        record["kind"] = "numeric"
        record["overall"] = _numeric_inventory(series)
        record["by_projection_mode"] = {
            str(mode): _numeric_inventory(series[frame["projection_mode"].eq(mode)])
            for mode in sorted(frame["projection_mode"].astype(str).unique())
        }
    elif column in BOOLEAN_COLUMNS:
        record["kind"] = "boolean"
        record["overall"] = {
            "false_count": int((~series).sum()),
            "true_count": int(series.sum()),
        }
        record["by_projection_mode"] = {
            str(mode): {
                "false_count": int((~series[frame["projection_mode"].eq(mode)]).sum()),
                "true_count": int(series[frame["projection_mode"].eq(mode)].sum()),
            }
            for mode in sorted(frame["projection_mode"].astype(str).unique())
        }
    else:
        record["kind"] = "text_or_categorical"
        record["missing_count"] = int(series.isna().sum())
        record["examples"] = sorted(str(value) for value in series.dropna().unique())[:8]
        record["by_projection_mode"] = {
            str(mode): {
                "missing_count": int(series[frame["projection_mode"].eq(mode)].isna().sum()),
                "rows": int(frame["projection_mode"].eq(mode).sum()),
                "unique_nonmissing": int(
                    series[frame["projection_mode"].eq(mode)].nunique(dropna=True)
                ),
            }
            for mode in sorted(frame["projection_mode"].astype(str).unique())
        }
    return record


def build_inventory(episodes_path: str, label: str | None = None) -> dict[str, object]:
    episodes = load_episode_table(episodes_path)
    result: dict[str, object] = {
        "columns": [_column_inventory(episodes, column) for column in episodes.columns],
        "implementation": implementation_record(__file__),
        "input": input_record(episodes_path),
        "output_schema": OUTPUT_SCHEMA,
        "row_count": int(len(episodes)),
        "script": "inventory_evaluation_table",
    }
    if label is not None:
        result["label"] = label
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory one explicitly supplied evaluation episode table."
    )
    parser.add_argument("--episodes", required=True, help="Evaluation episode CSV to inspect.")
    parser.add_argument("--output", required=True, help="New JSON output path; existing files are refused.")
    parser.add_argument("--label", help="Optional descriptive label recorded verbatim in the output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = build_inventory(args.episodes, args.label)
        write_json_exclusive(args.output, result)
    except EvidenceError as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
