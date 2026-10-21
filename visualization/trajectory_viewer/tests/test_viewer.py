"""Local browser server checks using tiny fixtures; no evaluation is rerun."""
from pathlib import Path
import copy
import hashlib
import http.client
import io
import json
import sys
import threading
import time
import unittest
import zipfile
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import viewer
import test_data as fixtures
from data import load_pair


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PairValidationTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.state = viewer.ViewerState(self.fixture.root)
        self.addCleanup(self.state.close)

    def server(self):
        server = viewer.ViewerServer(self.state)
        thread = threading.Thread(target=server.serve_forever,
                                  kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        def stop():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.addCleanup(stop)
        return server

    def request(self, server, method, path, payload=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            actual = dict(headers or {})
            body = None
            if payload is not None:
                body = json.dumps(payload).encode("utf-8")
                actual.setdefault("Content-Type", "application/json")
            conn.request(method, path, body=body, headers=actual)
            response = conn.getresponse()
            content = response.read()
            return response.status, dict(response.getheaders()), content
        finally:
            conn.close()

    def test_json_preserves_saved_geometry_timing_and_pair_identity(self):
        payload = json.loads(self.state.pair_bytes(0))
        pair = load_pair(self.fixture.root, self.fixture.suite,
                         self.fixture.method, 1, 0)
        self.assertEqual(payload["record"]["id"], 0)
        self.assertEqual(payload["record"]["evaluation_seed"], 1000)
        self.assertNotIn("rows", payload["record"])
        self.assertEqual(payload["duration"], .2)
        for actual, expected in zip(payload["pair"], pair):
            for field in viewer.ARRAY_FIELDS:
                self.assertEqual(actual[field], expected[field].tolist(), field)
            self.assertEqual(actual["outcome"], "SUCCESS")
            self.assertEqual(actual["steps"], 2)
            self.assertEqual(actual["dt"], .1)
            self.assertEqual(actual["duration"], .2)
            self.assertEqual(actual["goal_radius"], .25)
        self.assertIs(self.state.pair_bytes(0), self.state.pair_bytes(0))
        with self.assertRaises(ValueError):
            viewer.json_bytes({"bad": float("nan")})

    def test_catalog_omits_missing_archives_and_private_rows(self):
        payload = self.state.catalog_payload()
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["unavailable"], 0)
        self.assertNotIn("rows", payload["records"][0])
        (self.fixture.folder / "projection_enabled_trajectories.npz").unlink()
        missing_state = viewer.ViewerState(self.fixture.root)
        self.addCleanup(missing_state.close)
        missing = missing_state.catalog_payload()
        self.assertEqual(missing["records"], [])
        self.assertEqual(missing["total"], 0)
        self.assertEqual(missing["unavailable"], 1)

    def test_rejects_invalid_pair_ids(self):
        for value in (-1, 1, "0", 0.0, True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.state.pair_bytes(value)
        server = self.server()
        for query in ("", "?id=-1", "?id=1", "?id=x", "?id=0&id=0",
                      "?id=0&path=../../data", "?id="):
            with self.subTest(query=query):
                status, _, body = self.request(server, "GET", "/api/pair" + query)
                self.assertEqual(status, 400, body)
        status, _, body = self.request(server, "GET", "/api/pair?id=0")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["record"]["id"], 0)

    def test_local_host_origin_and_export_token_guards(self):
        server = self.server()
        for headers in ({"Host": "elsewhere.example"},
                        {"Origin": "https://elsewhere.example"}):
            with self.subTest(headers=headers):
                status, _, _ = self.request(server, "GET", "/api/catalog", headers=headers)
                self.assertEqual(status, 403)
        status, headers, _ = self.request(server, "GET", "/api/catalog")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        with mock.patch.object(self.state, "start_export") as start:
            for token in (None, "wrong", "\N{SNOWMAN}"):
                with self.subTest(token=token):
                    status, _, _ = self.request(server, "POST", "/api/export",
                                                {"id": 0, "token": token})
                    self.assertEqual(status, 403)
            start.assert_not_called()
            start.return_value = {"id": "job", "status": "running", "pair_id": 0}
            status, _, body = self.request(server, "POST", "/api/export",
                                          {"id": 0, "token": self.state.token})
            self.assertEqual(status, 202, body)
            start.assert_called_once_with(0)

    def repository_snapshot(self):
        return {str(path.relative_to(self.fixture.root)):
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "directory"
                for path in self.fixture.root.rglob("*")}

    def wait_export(self, job_id):
        deadline = time.monotonic() + 5
        while self.state.job(job_id)["status"] == "running" and time.monotonic() < deadline:
            time.sleep(.01)
        result = self.state.job(job_id)
        self.assertNotEqual(result["status"], "running", "Export worker did not finish")
        return result

    @staticmethod
    def write_fake_export(command, **kwargs):
        folder = Path(command[command.index("--output-dir") + 1])
        (folder / "PAP_Recorded_Trajectories_3D.mp4").write_bytes(b"fixture-video")
        (folder / "selected_episodes.json").write_text(json.dumps({
            "suite": command[command.index("--suite") + 1],
            "method": command[command.index("--method") + 1],
            "train_seed": int(command[command.index("--train-seed") + 1]),
            "episode": int(command[command.index("--episode") + 1]),
        }))
        kwargs["stdout"].write("Fixture renderer completed.\n")
        return mock.Mock(returncode=0)

    def test_missing_ffmpeg_and_renderer_failure_are_reported(self):
        self.state.records[0]["suite"] = "core_layout_transfer"
        snapshot = self.repository_snapshot()
        with mock.patch.object(viewer.shutil, "which", return_value=None):
            self.assertFalse(self.state.catalog_payload()["video_export_available"])
            with self.assertRaisesRegex(ValueError, "FFmpeg"):
                self.state.start_export(0)
        def fail_renderer(command, **kwargs):
            kwargs["stdout"].write("Fixture encoder failure.\n")
            return mock.Mock(returncode=1)
        with mock.patch.object(viewer.shutil, "which", return_value="ffmpeg"), \
             mock.patch.object(viewer.subprocess, "run", side_effect=fail_renderer):
            initial = self.state.start_export(0)
            result = self.wait_export(initial["id"])
        self.assertEqual(result["status"], "error")
        self.assertNotIn("download", result)
        self.assertIn("log_download", result)
        self.assertFalse(any(key.startswith("_") for key in result))
        self.assertNotIn("output_dir", result)
        server = self.server()
        status, headers, body = self.request(server, "GET", result["log_download"])
        self.assertEqual(status, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertIn(b"Fixture encoder failure.", body)
        self.assertEqual(self.repository_snapshot(), snapshot)

    def test_export_zip_contains_selected_pair_and_leaves_repository_unchanged(self):
        selected = copy.deepcopy(self.state.records[0])
        selected.update(suite="core_layout_transfer", method="ppo_train_projection",
                        train_seed=4, episode=8)
        self.state.records.append(selected)
        snapshot = self.repository_snapshot()
        called = []
        def fake_renderer(command, **kwargs):
            called.append((command, kwargs))
            return self.write_fake_export(command, **kwargs)
        with mock.patch.object(viewer.shutil, "which", return_value="ffmpeg"), \
             mock.patch.object(viewer.subprocess, "run", side_effect=fake_renderer):
            initial = self.state.start_export(1)
            done = self.wait_export(initial["id"])
        self.assertEqual(done["status"], "done", done)
        for job in (initial, done):
            self.assertFalse(any(key.startswith("_") for key in job))
            self.assertNotIn("output_dir", job)
        self.assertEqual(len(called), 1)
        command, kwargs = called[0]
        for option, value in (("--suite", "core_layout_transfer"),
                              ("--method", "ppo_train_projection"),
                              ("--train-seed", "4"), ("--episode", "8")):
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertEqual(done["pair_id"], 1)
        folder = Path(command[command.index("--output-dir") + 1])
        self.assertNotIn(self.fixture.root, folder.parents)
        self.assertEqual(kwargs["cwd"], str(folder))
        self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(Path(kwargs["env"]["MPLCONFIGDIR"]).parent, folder)
        self.assertTrue(done["filename"].endswith(".zip"))
        server = self.server()
        status, headers, body = self.request(server, "GET", done["download"])
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/zip")
        self.assertIn(done["filename"], headers["Content-Disposition"])
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            self.assertEqual(set(archive.namelist()), {
                "PAP_Recorded_Trajectories_3D.mp4", "selected_episodes.json", "export.log"})
            self.assertEqual(archive.read("PAP_Recorded_Trajectories_3D.mp4"), b"fixture-video")
            self.assertIn(b"Fixture renderer completed.", archive.read("export.log"))
            manifest = json.loads(archive.read("selected_episodes.json"))
            self.assertEqual(manifest, {"suite": "core_layout_transfer",
                "method": "ppo_train_projection", "train_seed": 4, "episode": 8})
        self.assertEqual(self.repository_snapshot(), snapshot)
        self.state.close()
        self.assertFalse(folder.exists())
        self.assertEqual(self.repository_snapshot(), snapshot)

    def test_close_waits_for_active_export_then_removes_temporary_files(self):
        self.state.records[0]["suite"] = "core_layout_transfer"
        snapshot = self.repository_snapshot()
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        folders = []
        def blocking_renderer(command, **kwargs):
            folders.append(Path(command[command.index("--output-dir") + 1]))
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("Test did not release fake renderer")
            return self.write_fake_export(command, **kwargs)
        with mock.patch.object(viewer.shutil, "which", return_value="ffmpeg"), \
             mock.patch.object(viewer.subprocess, "run", side_effect=blocking_renderer):
            self.state.start_export(0)
            self.assertTrue(entered.wait(timeout=5))
            def close():
                self.state.close()
                closed.set()
            thread = threading.Thread(target=close, daemon=True)
            thread.start()
            try:
                self.assertFalse(closed.wait(timeout=.05))
                self.assertTrue(folders[0].exists())
            finally:
                release.set()
                thread.join(timeout=5)
            self.assertTrue(closed.is_set())
        self.assertFalse(folders[0].exists())
        self.assertEqual(self.repository_snapshot(), snapshot)


if __name__ == "__main__":
    unittest.main()
