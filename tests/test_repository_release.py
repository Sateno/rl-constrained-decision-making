"""Failure cases for the release gate and its text/binary integrity boundary."""

import subprocess
from zipfile import ZipFile

import pytest

from evaluation.verify_repository_release import (
    EXTERNAL_MANUALS, digest, local_recorded_path, verify_archive_files, verify_base, verify_files, verify_paper,
)


@pytest.fixture
def release(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    data = b"a,b\n1,2\n"
    (tmp_path / "evidence.csv").write_bytes(data)
    return tmp_path, {
        "files": {"evidence.csv": {"text": True, "sha256": digest(data, text=True)}},
        "excluded_paths": ["paper.tex"],
        "paper_evidence_map": {"data/evidence.csv": "evidence.csv"},
    }


def test_text_newlines_are_portable_but_values_and_binary_bytes_are_exact(release):
    root, manifest = release
    (root / "evidence.csv").write_bytes(b"a,b\r\n1,2\r\n")
    assert verify_files(root, manifest, require_tracked=False) == 2
    assert digest(b"a\r\n", text=False) != digest(b"a\n", text=False)
    (root / "evidence.csv").write_text("a,b\n1,3\n")
    with pytest.raises(ValueError, match="content changed"):
        verify_files(root, manifest, require_tracked=False)


def test_missing_and_ignored_evidence_fail(release):
    root, manifest = release
    (root / ".gitignore").write_text("evidence.csv\n")
    with pytest.raises(ValueError, match="ignored by Git"):
        verify_files(root, manifest, require_tracked=False)
    (root / "evidence.csv").unlink()
    with pytest.raises(ValueError, match="Missing release file"):
        verify_files(root, manifest, require_tracked=False)


def test_retired_paper_asset_fails(release):
    root, manifest = release
    (root / "paper.tex").write_text("old manuscript")
    with pytest.raises(ValueError, match="retired paper asset"):
        verify_files(root, manifest, require_tracked=False)


def test_paper_comparison_checks_content_and_ambiguous_members(release):
    root, manifest = release
    path = root / "paper.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("project/data/evidence.csv", "a,b\n1,2\n")
    assert verify_paper(root, manifest, path) == 1
    with ZipFile(path, "w") as archive:
        archive.writestr("data/evidence.csv", "a,b\n1,3\n")
    with pytest.raises(ValueError, match="Paper evidence differs"):
        verify_paper(root, manifest, path)
    with ZipFile(path, "a") as archive:
        archive.writestr("project/data/evidence.csv", "a,b\n1,2\n")
    with pytest.raises(ValueError, match="ambiguous"):
        verify_paper(root, manifest, path)


def test_runtime_comparison_rejects_changed_behavior(release, monkeypatch):
    root, manifest = release
    manifest["protected_against_main"] = ["evidence.csv"]
    monkeypatch.setattr("evaluation.verify_repository_release.git", lambda _root, *args: b"abc123\n" if args[0] == "rev-parse" else b"a,b\r\n1,2\r\n")
    assert verify_base(root, manifest, "main") == "abc123"
    (root / "evidence.csv").write_text("a,b\n1,3\n")
    with pytest.raises(ValueError, match="differs from main"):
        verify_base(root, manifest, "main")


def test_recorded_path_relocation_rejects_escape(tmp_path):
    assert local_recorded_path(tmp_path, r"C:\old\repo\runs\checkpoints\model.pt") == tmp_path / "runs/checkpoints/model.pt"
    with pytest.raises(ValueError, match="escapes"):
        local_recorded_path(tmp_path, "runs/../../secret")


def test_staged_gate_rejects_an_older_staged_copy(release):
    root, manifest = release
    manifest_path = root / "verification/repository_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("{}\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    assert verify_files(root, manifest, require_tracked=True) == 2
    changed = b"a,b\n1,3\n"
    (root / "evidence.csv").write_bytes(changed)
    manifest["files"]["evidence.csv"]["sha256"] = digest(changed, text=True)
    with pytest.raises(ValueError, match="Staged content differs"):
        verify_files(root, manifest, require_tracked=True)


def test_staged_gate_rejects_unrelated_tracked_files(release):
    root, manifest = release
    manifest_path = root / "verification/repository_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("{}\n")
    (root / "unrelated.txt").write_text("not part of the release\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    with pytest.raises(ValueError, match="Unexpected tracked files"):
        verify_files(root, manifest, require_tracked=True)


def pdf_fixture(comment=b"Overleaf build"):
    prefix = b"%PDF-1.5\n% " + comment + b"\n"
    return prefix + b"xref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 >>\nstartxref\n" + str(len(prefix)).encode() + b"\n%%EOF\n"


def test_external_manual_can_be_recompiled_without_weakening_evidence_hashes(release):
    root, manifest = release
    relative = sorted(EXTERNAL_MANUALS)[0]
    path = root / relative
    path.parent.mkdir(parents=True)
    manifest["externally_compiled_manuals"] = [relative]
    path.write_bytes(pdf_fixture())
    assert verify_files(root, manifest, require_tracked=False) == 3
    path.write_bytes(pdf_fixture(b"Different compiler metadata"))
    assert verify_files(root, manifest, require_tracked=False) == 3
    (root / "evidence.csv").write_text("a,b\n1,3\n")
    with pytest.raises(ValueError, match="content changed"):
        verify_files(root, manifest, require_tracked=False)


def test_external_manual_rejects_missing_or_truncated_pdf(release):
    root, manifest = release
    relative = sorted(EXTERNAL_MANUALS)[0]
    path = root / relative
    path.parent.mkdir(parents=True)
    manifest["externally_compiled_manuals"] = [relative]
    with pytest.raises(ValueError, match="Missing compiled manual"):
        verify_files(root, manifest, require_tracked=False)
    path.write_bytes(b"not a PDF")
    with pytest.raises(ValueError, match="PDF header"):
        verify_files(root, manifest, require_tracked=False)
    path.write_bytes(pdf_fixture().replace(b"%%EOF", b""))
    with pytest.raises(ValueError, match="PDF end marker"):
        verify_files(root, manifest, require_tracked=False)
    path.write_bytes(pdf_fixture().replace(b"startxref\n", b"startxref\n999999"))
    with pytest.raises(ValueError, match="cross-reference pointer"):
        verify_files(root, manifest, require_tracked=False)


def test_external_manual_exception_cannot_cover_other_artifacts(release):
    root, manifest = release
    manifest["externally_compiled_manuals"] = ["evidence.csv"]
    with pytest.raises(ValueError, match="Only the three software manuals"):
        verify_files(root, manifest, require_tracked=False)
    manifest["externally_compiled_manuals"] = []
    manifest["excluded_trees"] = ["docs/source"]
    (root / "docs/source").mkdir(parents=True)
    (root / "docs/source/stale.tex").write_text("obsolete local source")
    with pytest.raises(ValueError, match="excluded source/build tree"):
        verify_files(root, manifest, require_tracked=False)


def test_external_manual_staging_requires_the_actual_pdf_bytes(release):
    root, manifest = release
    relative = sorted(EXTERNAL_MANUALS)[0]
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(pdf_fixture())
    manifest["externally_compiled_manuals"] = [relative]
    manifest_path = root / "verification/repository_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("{}\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    assert verify_files(root, manifest, require_tracked=True) == 3
    path.write_bytes(pdf_fixture(b"New Overleaf build"))
    with pytest.raises(ValueError, match="Staged content differs"):
        verify_files(root, manifest, require_tracked=True)


def test_local_archive_is_optional_and_checked_outside_git(release, tmp_path):
    root, manifest = release
    raw = b"preserved checkpoint bytes\x00\r\n"
    records = {"runs/checkpoints/model.pt": {"text": False, "sha256": digest(raw, text=False)}}
    manifest["local_archive_files"] = records
    # A clean checkout needs no runs directory; archive metadata is not a Git requirement.
    assert not (root / "runs").exists()
    assert verify_files(root, manifest, require_tracked=False) == 2
    archive_root = root / "separate_archive"
    checkpoint = archive_root / "runs/checkpoints/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(raw)
    assert verify_archive_files(archive_root, records) == 1
    checkpoint.write_bytes(raw + b"corruption")
    with pytest.raises(ValueError, match="Local archive content changed"):
        verify_archive_files(archive_root, records)
    checkpoint.unlink()
    with pytest.raises(ValueError, match="Missing local archive file"):
        verify_archive_files(archive_root, records)


def test_staged_runs_are_rejected_without_deleting_local_files(release):
    root, manifest = release
    checkpoint = root / "runs/checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"preserve me")
    (root / ".gitignore").write_text("runs/\n")
    assert verify_files(root, manifest, require_tracked=False) == 2
    subprocess.run(["git", "-C", str(root), "add", "-f", "runs/checkpoint.pt"], check=True)
    with pytest.raises(ValueError, match="preserve its local files"):
        verify_files(root, manifest, require_tracked=False)
    assert checkpoint.read_bytes() == b"preserve me"


def test_archive_inventory_cannot_become_a_repository_requirement(release):
    root, manifest = release
    item = {"text": False, "sha256": digest(b"x", text=False)}
    manifest["files"]["runs/checkpoint.pt"] = item
    with pytest.raises(ValueError, match="must not be required repository files"):
        verify_files(root, manifest, require_tracked=False)
    with pytest.raises(ValueError, match="Invalid local archive path"):
        verify_archive_files(root, {"runs/../../outside.pt": item})
