"""Load and validate the geometry definitions used by publication assets."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


FIXED_SUITE_ID = "fixed_training_geometry_v1"
TRANSFER_SUITE_ID = "core_navigation_layouts_v1"
EXPECTED_FAMILIES = {
    "control": 4,
    "single": 6,
    "double": 8,
    "triple": 6,
}
X_LIMITS = (-0.45, 4.45)
Y_LIMITS = (-1.22, 1.22)


def _point(value: object, *, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain two coordinates")
    point = (float(value[0]), float(value[1]))
    if not all(math.isfinite(coordinate) for coordinate in point):
        raise ValueError(f"{label} contains a non-finite coordinate")
    return point


def _geometry_signature(layout: dict[str, Any]) -> tuple[object, ...]:
    obstacles = tuple(
        sorted(
            (
                float(obstacle["center"][0]),
                float(obstacle["center"][1]),
                float(obstacle["radius"]),
            )
            for obstacle in layout["obstacles"]
        )
    )
    return (
        tuple(float(value) for value in layout["start"]),
        float(layout["theta"]),
        tuple(float(value) for value in layout["goal"]),
        obstacles,
    )


def load_layout_suite(
    path: Path,
    *,
    expected_suite_id: str,
    expected_layout_count: int,
) -> dict[str, Any]:
    """Load one committed layout suite and enforce its frozen design shape."""

    with path.open("r", encoding="utf-8") as stream:
        suite = json.load(stream)
    if not isinstance(suite, dict):
        raise ValueError(f"Layout JSON root must be an object: {path}")
    if suite.get("schema_version") != "navigation_layout_suite_v1":
        raise ValueError(f"Unexpected schema in {path}")
    if suite.get("suite_id") != expected_suite_id:
        raise ValueError(
            f"Expected suite {expected_suite_id!r}, found {suite.get('suite_id')!r}"
        )
    if int(suite.get("max_obstacles", -1)) != 3:
        raise ValueError("Expected max_obstacles = 3")
    if not math.isclose(float(suite.get("agent_radius", -1)), 0.10):
        raise ValueError("Expected agent_radius = 0.10")
    if not math.isclose(float(suite.get("goal_radius", -1)), 0.25):
        raise ValueError("Expected goal_radius = 0.25")

    layouts = suite.get("layouts")
    if not isinstance(layouts, list) or len(layouts) != expected_layout_count:
        raise ValueError(
            f"Expected {expected_layout_count} layouts in {expected_suite_id}"
        )

    seen_ids: set[str] = set()
    for index, layout in enumerate(layouts):
        if not isinstance(layout, dict):
            raise ValueError(f"Layout {index} is not an object")
        layout_id = layout.get("layout_id")
        if not isinstance(layout_id, str) or not layout_id:
            raise ValueError(f"Layout {index} has no identifier")
        if layout_id in seen_ids:
            raise ValueError(f"Duplicate layout identifier: {layout_id}")
        seen_ids.add(layout_id)

        start = _point(layout.get("start"), label=f"{layout_id}.start")
        goal = _point(layout.get("goal"), label=f"{layout_id}.goal")
        theta = float(layout.get("theta"))
        if start != (0.0, 0.0) or goal != (4.0, 0.0) or theta != 0.0:
            raise ValueError(f"Shared geometry invariants fail for {layout_id}")

        obstacles = layout.get("obstacles")
        if not isinstance(obstacles, list) or len(obstacles) > 3:
            raise ValueError(f"Invalid obstacle collection for {layout_id}")
        for obstacle_index, obstacle in enumerate(obstacles):
            if not isinstance(obstacle, dict):
                raise ValueError(f"Invalid obstacle {obstacle_index} in {layout_id}")
            center = _point(
                obstacle.get("center"),
                label=f"{layout_id}.obstacles[{obstacle_index}].center",
            )
            radius = float(obstacle.get("radius"))
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError(f"Invalid obstacle radius in {layout_id}")
            if not (
                X_LIMITS[0] <= center[0] - radius
                and center[0] + radius <= X_LIMITS[1]
                and Y_LIMITS[0] <= center[1] - radius
                and center[1] + radius <= Y_LIMITS[1]
            ):
                raise ValueError(f"Obstacle footprint is clipped in {layout_id}")

    if len({_geometry_signature(layout) for layout in layouts}) != len(layouts):
        raise ValueError(f"Duplicate task geometry in {expected_suite_id}")
    return suite


def load_study_layouts(layout_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and jointly validate the fixed and transfer suites."""

    fixed = load_layout_suite(
        layout_root / "fixed_training_geometry.json",
        expected_suite_id=FIXED_SUITE_ID,
        expected_layout_count=1,
    )
    transfer = load_layout_suite(
        layout_root / "core_navigation_layouts.json",
        expected_suite_id=TRANSFER_SUITE_ID,
        expected_layout_count=24,
    )

    fixed_layout = fixed["layouts"][0]
    if fixed_layout["layout_id"] != "fixed_training_geometry":
        raise ValueError("Unexpected fixed-geometry layout identifier")
    if len(fixed_layout["obstacles"]) != 3:
        raise ValueError("Fixed training geometry must contain three obstacles")

    family_counts = {
        family: sum(
            str(layout["layout_id"]).startswith(family + "_")
            for layout in transfer["layouts"]
        )
        for family in EXPECTED_FAMILIES
    }
    if family_counts != EXPECTED_FAMILIES:
        raise ValueError(
            f"Unexpected transfer-family counts: {family_counts}; "
            f"expected {EXPECTED_FAMILIES}"
        )
    if _geometry_signature(fixed_layout) in {
        _geometry_signature(layout) for layout in transfer["layouts"]
    }:
        raise ValueError("Fixed training geometry appears in the transfer suite")
    return fixed, transfer
