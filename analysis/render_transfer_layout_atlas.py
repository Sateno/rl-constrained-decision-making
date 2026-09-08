"""Render the 24 prespecified transfer layouts as a four-page vector PDF."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch

from .layout_geometry import EXPECTED_FAMILIES, X_LIMITS, Y_LIMITS, load_study_layouts


COLORS = {
    "ink": "#202124",
    "muted": "#5F6368",
    "grid": "#E3E7EA",
    "centerline": "#C5CBD0",
    "agent": "#2878B5",
    "agent_fill": "#B9D8EB",
    "goal": "#368A43",
    "goal_fill": "#CDE8CE",
    "obstacle": "#A44F1F",
    "obstacle_fill": "#D77A45",
}
FAMILY_TITLES = {
    "control": "Control and noninterference layouts",
    "single": "Single-obstacle layouts",
    "double": "Two-obstacle layouts",
    "triple": "Three-obstacle layouts",
}
FIXED_PDF_DATE = datetime(2026, 9, 4, tzinfo=timezone.utc)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 8.3,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "savefig.facecolor": "white",
    }
)


def _draw_layout(
    axis: plt.Axes,
    layout: dict[str, Any],
    *,
    agent_radius: float,
    goal_radius: float,
) -> None:
    start = tuple(float(value) for value in layout["start"])
    goal = tuple(float(value) for value in layout["goal"])
    theta = float(layout["theta"])
    axis.axhline(
        0,
        color=COLORS["centerline"],
        linewidth=0.75,
        linestyle=(0, (2.2, 2.2)),
        zorder=0,
    )
    for obstacle in layout["obstacles"]:
        axis.add_patch(
            Circle(
                tuple(float(value) for value in obstacle["center"]),
                float(obstacle["radius"]),
                facecolor=COLORS["obstacle_fill"],
                edgecolor=COLORS["obstacle"],
                linewidth=1.05,
                zorder=3,
            )
        )
    axis.add_patch(
        Circle(
            goal,
            goal_radius,
            facecolor=COLORS["goal_fill"],
            edgecolor=COLORS["goal"],
            linewidth=1.15,
            zorder=2,
        )
    )
    axis.plot(
        goal[0],
        goal[1],
        marker="+",
        color=COLORS["goal"],
        markersize=7,
        markeredgewidth=1.15,
        zorder=4,
    )
    axis.add_patch(
        Circle(
            start,
            agent_radius,
            facecolor=COLORS["agent_fill"],
            edgecolor=COLORS["agent"],
            linewidth=1.15,
            zorder=5,
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            start,
            (
                start[0] + 0.37 * math.cos(theta),
                start[1] + 0.37 * math.sin(theta),
            ),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.15,
            color=COLORS["agent"],
            shrinkA=0,
            shrinkB=0,
            zorder=6,
        )
    )
    axis.set_xlim(*X_LIMITS)
    axis.set_ylim(*Y_LIMITS)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks((0, 1, 2, 3, 4))
    axis.set_yticks((-1, 0, 1))
    axis.grid(color=COLORS["grid"], linewidth=0.55, zorder=-1)
    axis.tick_params(length=2.5, width=0.6, color=COLORS["muted"])
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(COLORS["muted"])


def _legend(agent_radius: float, goal_radius: float) -> list[Line2D]:
    return [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=COLORS["agent_fill"], markeredgecolor=COLORS["agent"],
            markersize=7.5, label=f"Start/agent  r={agent_radius:.2f}",
        ),
        Line2D(
            [0], [0], marker=r"$\rightarrow$", linestyle="none",
            color=COLORS["agent"], markersize=11, label=r"Heading  $\theta=0$",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=COLORS["goal_fill"], markeredgecolor=COLORS["goal"],
            markersize=9.5, label=f"Goal  r={goal_radius:.2f}",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=COLORS["obstacle_fill"], markeredgecolor=COLORS["obstacle"],
            markersize=9, label="Obstacle",
        ),
    ]


def render_transfer_layout_atlas(
    transfer_suite: dict[str, Any],
    output_path: Path,
) -> Path:
    """Render all 24 layouts on identical axes and physical panel sizes."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    agent_radius = float(transfer_suite["agent_radius"])
    goal_radius = float(transfer_suite["goal_radius"])
    metadata = {
        "Title": "Prespecified transfer-layout atlas",
        "Author": "Salvador Tenorio",
        "Subject": "All 24 committed transfer layouts on identical axes.",
        "Creator": "analysis.render_transfer_layout_atlas",
        "CreationDate": FIXED_PDF_DATE,
        "ModDate": FIXED_PDF_DATE,
    }

    with PdfPages(output_path, metadata=metadata) as pdf:
        for page_number, family in enumerate(EXPECTED_FAMILIES, start=1):
            layouts = [
                layout
                for layout in transfer_suite["layouts"]
                if str(layout["layout_id"]).startswith(family + "_")
            ]
            expected_count = EXPECTED_FAMILIES[family]
            if len(layouts) != expected_count:
                raise ValueError(f"Unexpected {family} layout count")
            rows = math.ceil(expected_count / 2)
            row_bottoms = {
                2: (0.555, 0.315),
                3: (0.635, 0.405, 0.175),
                4: (0.680, 0.490, 0.300, 0.110),
            }[rows]
            figure = plt.figure(figsize=(8.5, 11))
            axes = [
                figure.add_axes((left, bottom, 0.415, 0.160))
                for bottom in row_bottoms
                for left in (0.075, 0.565)
            ]
            figure.suptitle(
                f"{FAMILY_TITLES[family]} ({expected_count})",
                x=0.075,
                y=0.965,
                ha="left",
                va="top",
                fontsize=12,
                weight="bold",
                color=COLORS["ink"],
            )
            figure.text(
                0.075,
                0.93,
                "Prespecified transfer suite. Every panel uses identical coordinate limits and equal axis scale.",
                ha="left",
                va="top",
                fontsize=8,
                color=COLORS["muted"],
            )
            figure.text(
                0.975,
                0.965,
                f"Atlas page {page_number} of 4",
                ha="right",
                va="top",
                fontsize=7.5,
                color=COLORS["muted"],
            )
            for index, (axis, layout) in enumerate(zip(axes, layouts), start=1):
                _draw_layout(
                    axis,
                    layout,
                    agent_radius=agent_radius,
                    goal_radius=goal_radius,
                )
                axis.set_title(
                    f"{family[0].upper()}{index}. {layout['layout_id']}",
                    loc="left",
                    pad=5,
                    weight="bold",
                    fontsize=7.6,
                    color=COLORS["ink"],
                )
                row_index, column_index = divmod(index - 1, 2)
                if column_index == 0:
                    axis.set_ylabel("y coordinate")
                else:
                    axis.tick_params(labelleft=False)
                if row_index == rows - 1:
                    axis.set_xlabel("x coordinate")
                else:
                    axis.tick_params(labelbottom=False)
            figure.legend(
                handles=_legend(agent_radius, goal_radius),
                loc="lower center",
                bbox_to_anchor=(0.5, 0.032),
                ncol=4,
                frameon=False,
                handletextpad=0.55,
                columnspacing=1.35,
            )
            figure.text(
                0.5,
                0.012,
                "Obstacle, agent, and goal discs use the radii in the committed JSON definitions. Distances are environment-coordinate units.",
                ha="center",
                va="bottom",
                fontsize=7,
                color=COLORS["muted"],
            )
            pdf.savefig(figure)
            plt.close(figure)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _, transfer = load_study_layouts(
        args.repository_root.resolve() / "evaluation" / "layouts"
    )
    render_transfer_layout_atlas(transfer, args.output.resolve())
    print("TRANSFER_LAYOUT_ATLAS_BUILD PASS")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
