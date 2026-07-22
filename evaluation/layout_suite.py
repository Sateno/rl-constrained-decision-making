from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LAYOUT_SUITE_SCHEMA_VERSION = "navigation_layout_suite_v1"


#################################################################################
# region Data classes

# One named deterministic environment layout.
@dataclass(frozen=True)
class NavigationLayout:
#{
    layout_id: str
    start: np.ndarray
    theta: float
    goal: np.ndarray
    obstacle_centers: np.ndarray
    obstacle_radii: np.ndarray
    obstacle_mask: np.ndarray

    # Return fresh arrays in the format expected by ConstrainedNavigationEnv.reset(...).
    def reset_options(self) -> dict[str, object]:
    #{
        return {
            "start": self.start.copy(),
            "theta": float(self.theta),
            "goal": self.goal.copy(),
            "obstacle_centers": self.obstacle_centers.copy(),
            "obstacle_radii": self.obstacle_radii.copy(),
            "obstacle_mask": self.obstacle_mask.copy(),
        }

    #} End function reset_options

#} End dataclass NavigationLayout


# One immutable collection of layouts and the geometry contract shared by them.
@dataclass(frozen=True)
class NavigationLayoutSuite:
#{
    schema_version: str
    suite_id: str
    source_path: Path
    sha256: str
    max_obstacles: int
    agent_radius: float
    goal_radius: float
    layouts: tuple[NavigationLayout, ...]

#} End dataclass NavigationLayoutSuite

# end region Data classes


#################################################################################
# region Helpers

# Return a finite two-element vector from one JSON field.
def _as_vector2(value: object, field_name: str) -> np.ndarray:
#{
    vector = np.asarray(value, dtype=np.float64)

    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{field_name} must contain two finite numbers.")

    return vector

#} End function _as_vector2


# Return a finite scalar from one JSON field.
def _as_finite_float(value: object, field_name: str) -> float:
#{
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be numeric.") from error

    if not np.isfinite(scalar):
        raise ValueError(f"{field_name} must be finite.")

    return scalar

#} End function _as_finite_float


# Return the SHA-256 of exact file bytes.
def file_sha256(path: str | Path) -> str:
#{
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()

#} End function file_sha256

# end region Helpers


#################################################################################
# region Interface

# Load and validate one deterministic navigation layout suite from JSON.
def load_navigation_layout_suite(path: str | Path) -> NavigationLayoutSuite:
#{
    source_path = Path(path)

    if not source_path.exists():
        raise FileNotFoundError(f"Layout suite not found: {source_path}")

    raw_bytes = source_path.read_bytes()

    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Layout suite is not valid UTF-8 JSON: {source_path}") from error

    if not isinstance(data, dict):
        raise ValueError("Layout suite root must be a JSON object.")

    schema_version = str(data.get("schema_version", "")).strip()

    if schema_version != LAYOUT_SUITE_SCHEMA_VERSION:
        raise ValueError(
            f"Layout suite schema_version must be {LAYOUT_SUITE_SCHEMA_VERSION!r}."
        )

    suite_id = str(data.get("suite_id", "")).strip()

    if not suite_id:
        raise ValueError("Layout suite must define a nonempty suite_id.")

    raw_max_obstacles = data.get("max_obstacles")

    if isinstance(raw_max_obstacles, bool) or not isinstance(raw_max_obstacles, int):
        raise ValueError("Layout suite max_obstacles must be an integer.")

    max_obstacles = int(raw_max_obstacles)

    if max_obstacles <= 0:
        raise ValueError("Layout suite max_obstacles must be positive.")

    agent_radius = _as_finite_float(data.get("agent_radius"), "agent_radius")
    goal_radius = _as_finite_float(data.get("goal_radius"), "goal_radius")

    if agent_radius < 0.0:
        raise ValueError("Layout suite agent_radius must be nonnegative.")

    if goal_radius <= 0.0:
        raise ValueError("Layout suite goal_radius must be positive.")

    layout_data = data.get("layouts")

    if not isinstance(layout_data, list) or not layout_data:
        raise ValueError("Layout suite must define at least one layout.")

    layouts: list[NavigationLayout] = []
    layout_ids: set[str] = set()

    for layout_index, item in enumerate(layout_data):
    #{
        if not isinstance(item, dict):
            raise ValueError(f"Layout at index {layout_index} must be a JSON object.")

        layout_id = str(item.get("layout_id", "")).strip()

        if not layout_id:
            raise ValueError(f"Layout at index {layout_index} must define a nonempty layout_id.")

        if layout_id in layout_ids:
            raise ValueError(f"Duplicate layout_id: {layout_id}")

        layout_ids.add(layout_id)
        start = _as_vector2(item.get("start"), f"{layout_id}.start")
        theta = _as_finite_float(item.get("theta", 0.0), f"{layout_id}.theta")
        goal = _as_vector2(item.get("goal"), f"{layout_id}.goal")

        if np.linalg.norm(goal - start) <= goal_radius:
            raise ValueError(f"Layout {layout_id} starts within the goal radius.")

        obstacles = item.get("obstacles", [])

        if not isinstance(obstacles, list):
            raise ValueError(f"Layout {layout_id}.obstacles must be a JSON array.")

        if len(obstacles) > max_obstacles:
            raise ValueError(
                f"Layout {layout_id} defines {len(obstacles)} obstacles; "
                f"the suite capacity is {max_obstacles}."
            )

        obstacle_centers = np.zeros((max_obstacles, 2), dtype=np.float64)
        obstacle_radii = np.zeros(max_obstacles, dtype=np.float64)
        obstacle_mask = np.zeros(max_obstacles, dtype=bool)

        for obstacle_index, obstacle in enumerate(obstacles):
        #{
            if not isinstance(obstacle, dict):
                raise ValueError(
                    f"Layout {layout_id} obstacle {obstacle_index} must be a JSON object."
                )

            center = _as_vector2(
                obstacle.get("center"),
                f"{layout_id}.obstacles[{obstacle_index}].center",
            )
            radius = _as_finite_float(
                obstacle.get("radius"),
                f"{layout_id}.obstacles[{obstacle_index}].radius",
            )

            if radius <= 0.0:
                raise ValueError(
                    f"Layout {layout_id} obstacle {obstacle_index} radius must be positive."
                )

            collision_radius = radius + agent_radius

            if np.linalg.norm(start - center) <= collision_radius:
                raise ValueError(
                    f"Layout {layout_id} starts in collision with obstacle {obstacle_index}."
                )

            if np.linalg.norm(goal - center) <= collision_radius:
                raise ValueError(
                    f"Layout {layout_id} goal is in collision with obstacle {obstacle_index}."
                )

            obstacle_centers[obstacle_index] = center
            obstacle_radii[obstacle_index] = radius
            obstacle_mask[obstacle_index] = True

        #} End loop obstacles

        layouts.append(
            NavigationLayout(
                layout_id=layout_id,
                start=start,
                theta=theta,
                goal=goal,
                obstacle_centers=obstacle_centers,
                obstacle_radii=obstacle_radii,
                obstacle_mask=obstacle_mask,
            )
        )

    #} End loop layouts

    return NavigationLayoutSuite(
        schema_version=schema_version,
        suite_id=suite_id,
        source_path=source_path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        max_obstacles=max_obstacles,
        agent_radius=agent_radius,
        goal_radius=goal_radius,
        layouts=tuple(layouts),
    )

#} End function load_navigation_layout_suite

# end region Interface


__all__ = [
    "LAYOUT_SUITE_SCHEMA_VERSION",
    "NavigationLayout",
    "NavigationLayoutSuite",
    "file_sha256",
    "load_navigation_layout_suite",
]
