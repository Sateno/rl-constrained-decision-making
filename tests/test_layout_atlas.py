"""Retained benchmark documentation is independent of paper production."""

import os
from pathlib import Path
import re
import subprocess
import sys

from analysis.layout_geometry import load_study_layouts


def test_atlas_covers_the_complete_frozen_suite_as_four_vector_pages(tmp_path):
    root = Path(__file__).resolve().parents[1]
    fixed, transfer = load_study_layouts(root / "evaluation/layouts")
    assert len(fixed["layouts"]) == 1
    assert len(transfer["layouts"]) == 24
    path = tmp_path / "atlas.pdf"
    # PDF rendering must run outside the pytest process that loads PyTorch.
    # This follows the existing Windows isolation rule in test_result_aggregation.
    # A native child failure remains a test failure with its diagnostics captured.
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib-config")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis.render_transfer_layout_atlas",
            "--repository-root",
            str(root),
            "--output",
            str(path),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, (
        f"Atlas render subprocess exited {completed.returncode}.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert "TRANSFER_LAYOUT_ATLAS_BUILD PASS" in completed.stdout
    payload = path.read_bytes()
    assert payload.startswith(b"%PDF-")
    assert len(re.findall(rb"/Type\s*/Page\b", payload)) == 4
    assert b"/Subtype /Image" not in payload
