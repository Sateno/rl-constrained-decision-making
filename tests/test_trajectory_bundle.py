"""Synthetic bundle failures check integrity, not the scientific conclusions."""
from pathlib import Path
import csv
import json
import tempfile
import unittest

import numpy as np

from evaluation.verify_trajectory_bundle import (
    DEFAULT_BUNDLE, digest, safe_path, verify_trajectory_bundle,
)


class TrajectoryBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / DEFAULT_BUNDLE
        self.bundle.mkdir(parents=True)
        self.table = self.root / "results/tables/example/evaluation_episode_results.csv"
        self.table.parent.mkdir(parents=True)
        self.release = {"local_archive_files": {}}
        self.manifest = {
            "schema_version": "pap_trajectory_bundle_v1", "dataset_id": "pap_evaluation_v1",
            "trajectory_archive_version": "evaluation_trajectory_v1", "tables": {}, "archives": {},
            "counts": {"archives": 2, "episodes": 2, "pairs": 1},
        }
        self.rows = []
        for mode in ("disabled", "enabled"):
            self.rows.append({
                "method": "method", "policy": "method", "train_seed": "1", "episode": "0",
                "projection_mode": mode, "checkpoint_sha256": "checkpoint",
                "layout_suite_sha256": "layout", "layout_id": "example", "layout_repeat": "0",
                "evaluation_seed": "1000", "evaluation_policy_mode": "deterministic",
                "episode_length": "2", "episode_return": "1.0", "success": "True",
                "collision": "False", "terminated": "True", "truncated": "False",
                "final_distance_to_goal": "0.0", "layout_goal_radius": "0.25",
            })
            data = {
                "policy": np.asarray("method"), "episode": np.asarray(0), "seed": np.asarray(1000),
                "goal": np.array([.2, 0.]), "positions": np.array([[0., 0.], [.1, 0.], [.2, 0.]]),
                "headings": np.zeros(3), "state_step_count": np.arange(3),
                "state_distance_to_goal": np.array([.2, .1, 0.]),
                "state_min_obstacle_clearance": np.full(3, np.nan),
                "obstacle_centers": np.zeros((1, 2)), "obstacle_radii": np.zeros(1),
                "obstacle_mask": np.zeros(1, dtype=bool), "agent_radius": np.asarray(.1),
                "dt": np.asarray(.1), "v_max": np.asarray(1.), "omega_max": np.asarray(2.),
                "state_success": np.array([False, False, True]),
                "state_collision": np.zeros(3, dtype=bool),
                "rewards": np.array([.5, .5]), "terminated": np.array([False, True]),
                "truncated": np.zeros(2, dtype=bool),
                "projection_enabled": np.full(2, mode == "enabled"),
                "projection_intervened": np.zeros(2, dtype=bool),
                "projection_success": np.ones(2, dtype=bool),
                "projection_solver_status": np.asarray(["disabled", "disabled"]),
                "projection_slack_values": np.zeros((2, 1)),
            }
            for field in ("action_raw_normalized", "action_raw_physical", "action_exec_physical", "action_correction_physical"):
                data[field] = np.zeros((2, 2))
            for field in ("projection_correction_norm", "projection_slack_sum", "projection_slack_max", "projection_active_constraint_count"):
                data[field] = np.zeros(2)
            arrays = {"episode_0000_" + key: value for key, value in data.items()}
            arrays.update({
                "trajectory_archive_version": np.asarray("evaluation_trajectory_v1"),
                "episode_keys": np.asarray(["episode_0000"]),
                "run_method": np.asarray("method"), "run_train_seed": np.asarray(1),
                "run_projection_mode": np.asarray(mode), "run_checkpoint_sha256": np.asarray("checkpoint"),
                "run_layout_suite_sha256": np.asarray("layout"), "run_stochastic": np.asarray(False),
                "run_goal_radius": np.asarray(.25),
            })
            path = self.archive_path(mode)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
            self.refresh_archive(mode)
        self.write_table()
        self.save_manifests()

    def archive_path(self, mode):
        return self.bundle / f"example/method/seed1/projection_{mode}_trajectories.npz"

    def refresh_archive(self, mode):
        path = self.archive_path(mode)
        relative = path.relative_to(self.bundle).as_posix()
        source = "runs/evaluation/final/" + relative
        data = path.read_bytes()
        self.manifest["archives"][relative] = {"sha256": digest(data), "size_bytes": len(data),
                                                "episodes": 1, "source_archive": source}
        self.release["local_archive_files"][source] = {"sha256": digest(data), "text": False}

    def write_table(self):
        with self.table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.manifest["tables"]["example"] = {
            "path": self.table.relative_to(self.root).as_posix(),
            "sha256": digest(self.table.read_bytes(), text=True),
        }

    def save_manifests(self):
        (self.bundle / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        path = self.root / "verification/repository_manifest.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(self.release), encoding="utf-8")

    def change_array(self, name, value, mode="enabled"):
        path = self.archive_path(mode)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays[name] = np.asarray(value)
        np.savez_compressed(path, **arrays)
        # Keep fixture provenance consistent to exercise semantic checks beyond hashes.
        self.refresh_archive(mode)
        self.save_manifests()

    def test_complete_bundle_needs_no_local_runs(self):
        self.assertFalse((self.root / "runs").exists())
        self.assertEqual(verify_trajectory_bundle(self.root), self.manifest["counts"])

    def test_table_newline_normalization_is_portable(self):
        data = self.table.read_bytes().replace(b"\r\n", b"\n")
        self.table.write_bytes(data)
        verify_trajectory_bundle(self.root)
        self.table.write_bytes(data.replace(b"\n", b"\r\n"))
        verify_trajectory_bundle(self.root)

    def test_changed_archive_bytes_rejected(self):
        path = self.archive_path("enabled")
        path.write_bytes(path.read_bytes() + b"unexpected")
        with self.assertRaisesRegex(ValueError, "size/hash"):
            verify_trajectory_bundle(self.root)

    def test_changed_table_rejected(self):
        self.table.write_bytes(self.table.read_bytes().replace(b"checkpoint", b"different"))
        with self.assertRaisesRegex(ValueError, "Table hash"):
            verify_trajectory_bundle(self.root)

    def test_original_archive_provenance_is_required(self):
        item = next(iter(self.release["local_archive_files"].values()))
        item["sha256"] = "different"
        self.save_manifests()
        with self.assertRaisesRegex(ValueError, "Frozen archive provenance"):
            verify_trajectory_bundle(self.root)

    def test_missing_or_extra_archive_rejected(self):
        path = self.bundle / "extra.npz"
        path.write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify_trajectory_bundle(self.root)
        path.unlink()
        self.archive_path("enabled").unlink()
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify_trajectory_bundle(self.root)

    def test_paths_cannot_escape_bundle(self):
        for relative in ("../outside", "/outside", r"C:\outside", "a/../outside"):
            with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "Invalid bundle path"):
                safe_path(self.root, relative)

    def test_wrong_seed_or_checkpoint_rejected(self):
        self.change_array("episode_0000_seed", 1001)
        with self.assertRaisesRegex(ValueError, "Evaluation seed"):
            verify_trajectory_bundle(self.root)
        self.change_array("episode_0000_seed", 1000)
        self.change_array("run_checkpoint_sha256", "different")
        with self.assertRaisesRegex(ValueError, "Checkpoint hash"):
            verify_trajectory_bundle(self.root)

    def test_wrong_action_mode_rejected(self):
        self.change_array("run_stochastic", True)
        with self.assertRaisesRegex(ValueError, "Action mode"):
            verify_trajectory_bundle(self.root)

    def test_misaligned_motion_rejected(self):
        self.change_array("episode_0000_positions", np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, "Invalid shape"):
            verify_trajectory_bundle(self.root)

    def test_return_and_terminal_disagreement_rejected(self):
        self.change_array("episode_0000_rewards", [.2, .5])
        with self.assertRaisesRegex(ValueError, "Episode return"):
            verify_trajectory_bundle(self.root)
        self.change_array("episode_0000_rewards", [.5, .5])
        self.change_array("episode_0000_state_success", [False, False, False])
        with self.assertRaisesRegex(ValueError, "Terminal success"):
            verify_trajectory_bundle(self.root)

    def test_off_on_initial_conditions_must_match(self):
        self.change_array("episode_0000_positions", [[.01, 0.], [.1, 0.], [.2, 0.]])
        with self.assertRaisesRegex(ValueError, "Paired initial condition position"):
            verify_trajectory_bundle(self.root)

    def test_counts_and_table_coverage_are_checked(self):
        self.manifest["counts"]["episodes"] = 3
        self.save_manifests()
        with self.assertRaisesRegex(ValueError, "Declared bundle counts"):
            verify_trajectory_bundle(self.root)
        self.rows.pop()
        self.write_table()
        self.save_manifests()
        with self.assertRaisesRegex(ValueError, "Table/archive coverage"):
            verify_trajectory_bundle(self.root)

    def test_unlisted_episode_payload_rejected(self):
        self.change_array("episode_0001_seed", 1000)
        with self.assertRaisesRegex(ValueError, "Unlisted episode"):
            verify_trajectory_bundle(self.root)


if __name__ == "__main__":
    unittest.main()
