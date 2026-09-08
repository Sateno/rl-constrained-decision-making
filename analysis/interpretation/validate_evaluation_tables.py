"""Validate one explicitly supplied family of evaluation tables and sources."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__:
    from ._evaluation_evidence import (
        EvidenceError,
        OUTPUT_SCHEMA,
        checkpoint_identity_map,
        implementation_record,
        input_record,
        input_records,
        load_design,
        load_episode_table,
        read_csv,
        reconcile_checkpoint_and_method_tables,
        reconcile_paired_tables,
        validate_build_audit,
        validate_episode_table,
        write_json_exclusive,
    )
else:  # Support invocation by absolute script path from any working directory.
    sys.path.insert(0, str(Path(__file__).parent))
    from _evaluation_evidence import (  # type: ignore[no-redef]
        EvidenceError,
        OUTPUT_SCHEMA,
        checkpoint_identity_map,
        implementation_record,
        input_record,
        input_records,
        load_design,
        load_episode_table,
        read_csv,
        reconcile_checkpoint_and_method_tables,
        reconcile_paired_tables,
        validate_build_audit,
        validate_episode_table,
        write_json_exclusive,
    )


def build_validation(
    *,
    episodes_path: str,
    protocol_path: str,
    layout_suite_path: str,
    checkpoint_summary_path: str,
    method_summary_path: str,
    paired_deltas_path: str,
    paired_summary_path: str,
    build_audit_path: str,
    compare_checkpoint_paths: list[str],
    label: str | None,
) -> dict[str, object]:
    design = load_design(protocol_path, layout_suite_path)
    episodes = load_episode_table(episodes_path)
    episode_checks = validate_episode_table(design, episodes)
    schema = str(episode_checks["result_build_schema_version"])
    _, _, summary_reconciliation = reconcile_checkpoint_and_method_tables(
        design,
        episodes,
        checkpoint_summary_path,
        method_summary_path,
        schema,
    )
    paired_reconciliation = reconcile_paired_tables(
        design,
        episodes,
        paired_deltas_path,
        paired_summary_path,
        schema,
    )
    audit_checks = validate_build_audit(design, build_audit_path, schema, episodes)

    reference_map = checkpoint_identity_map(episodes)
    comparisons: list[dict[str, object]] = []
    for comparison_path in compare_checkpoint_paths:
        comparison_table = load_episode_table(comparison_path)
        comparison_map = checkpoint_identity_map(comparison_table)
        if comparison_map != reference_map:
            raise EvidenceError(
                f"Checkpoint identities differ between --episodes and --compare-checkpoints-with {comparison_path!r}."
            )
        comparisons.append(
            {
                "checkpoint_identity_count": len(comparison_map),
                "exact_match": True,
                "input": input_record(comparison_path),
            }
        )

    paths = {
        "build_audit": build_audit_path,
        "checkpoint_summary": checkpoint_summary_path,
        "episodes": episodes_path,
        "layout_suite": layout_suite_path,
        "method_summary": method_summary_path,
        "paired_deltas": paired_deltas_path,
        "paired_summary": paired_summary_path,
        "protocol": protocol_path,
    }
    result: dict[str, object] = {
        "build_audit_checks": audit_checks,
        "cross_table_checkpoint_comparisons": comparisons,
        "design": design.record(),
        "episode_checks": episode_checks,
        "implementation": implementation_record(__file__),
        "inputs": input_records(paths),
        "output_schema": OUTPUT_SCHEMA,
        "reconciliation": {
            **summary_reconciliation,
            **paired_reconciliation,
        },
        "script": "validate_evaluation_tables",
        "status": "PASS",
        "table_shapes": {
            "checkpoint_summary": list(read_csv(checkpoint_summary_path).shape),
            "episodes": list(episodes.shape),
            "method_summary": list(read_csv(method_summary_path).shape),
            "paired_deltas": list(read_csv(paired_deltas_path).shape),
            "paired_summary": list(read_csv(paired_summary_path).shape),
        },
    }
    if label is not None:
        result["label"] = label
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate explicitly supplied episode, protocol, layout, summary, paired, "
            "and build-audit files without locating evidence implicitly."
        )
    )
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--layout-suite", required=True)
    parser.add_argument("--checkpoint-summary", required=True)
    parser.add_argument("--method-summary", required=True)
    parser.add_argument("--paired-deltas", required=True)
    parser.add_argument("--paired-summary", required=True)
    parser.add_argument("--build-audit", required=True)
    parser.add_argument(
        "--compare-checkpoints-with",
        action="append",
        default=[],
        metavar="EPISODES_CSV",
        help="Optional episode CSV whose method/seed checkpoint map must match exactly; repeatable.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--label")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = build_validation(
            episodes_path=args.episodes,
            protocol_path=args.protocol,
            layout_suite_path=args.layout_suite,
            checkpoint_summary_path=args.checkpoint_summary,
            method_summary_path=args.method_summary,
            paired_deltas_path=args.paired_deltas,
            paired_summary_path=args.paired_summary,
            build_audit_path=args.build_audit,
            compare_checkpoint_paths=args.compare_checkpoints_with,
            label=args.label,
        )
        write_json_exclusive(args.output, result)
    except EvidenceError as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
