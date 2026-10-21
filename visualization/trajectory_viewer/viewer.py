"""Open the offline PAP trajectory explorer in your browser.

Only saved evaluation arrays are read. No model, environment or QP is run.
The HTTP server binds exclusively to this computer's loopback interface.
"""
from collections import OrderedDict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import hmac
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
from urllib.parse import parse_qs, urlsplit
import webbrowser
import zipfile

from data import catalog, load_pair, outcome

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RECORD_FIELDS = ("suite", "method", "train_seed", "episode", "layout_id",
                 "evaluation_seed", "execution", "off", "on", "available")
ARRAY_FIELDS = ("positions", "headings", "goal", "obstacle_centers",
                "obstacle_radii", "obstacle_mask", "projection_intervened",
                "projection_slack_max")


def json_bytes(value):
    # Non-finite scientific values must fail visibly, never silently become null.
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")


class ViewerState:
    """Read-only recording access and separately managed video exports."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        records = catalog(self.root)
        self.records = [r for r in records if r["available"]]
        self.missing = len(records) - len(self.records)
        self.token = secrets.token_urlsafe(32)
        self.cache = OrderedDict()
        self.cache_lock = threading.Lock()
        self.jobs = {}
        self.jobs_lock = threading.Lock()
        self._export_temp = None
        self._workers = []
        self._closing = False

    def public_record(self, index):
        record = self.record(index)
        return {"id": index, **{key: record[key] for key in RECORD_FIELDS}}

    def record(self, index):
        if type(index) is not int or not 0 <= index < len(self.records):
            raise ValueError("Unknown trajectory pair. Select a pair from the list.")
        return self.records[index]

    def catalog_payload(self):
        return {"records": [self.public_record(i) for i in range(len(self.records))],
                "total": len(self.records), "unavailable": self.missing,
                "token": self.token, "video_export_available": bool(shutil.which("ffmpeg"))}

    def pair_bytes(self, index):
        record = self.record(index)
        with self.cache_lock:
            if index in self.cache:
                self.cache.move_to_end(index)
                return self.cache[index]
        pair = load_pair(self.root, record["suite"], record["method"],
                         record["train_seed"], record["episode"], record=record)
        payload = []
        for data in pair:
            steps = len(data["positions"]) - 1
            item = {key: data[key].tolist() for key in ARRAY_FIELDS}
            item.update(mode=data["mode"], dt=float(data["dt"]),
                        agent_radius=float(data["agent_radius"]),
                        goal_radius=float(data["goal_radius"]),
                        stochastic=bool(data["stochastic"]), outcome=outcome(data),
                        steps=steps, duration=steps * float(data["dt"]))
            payload.append(item)
        body = json_bytes({"record": self.public_record(index), "pair": payload,
                           "duration": max(d["duration"] for d in payload)})
        with self.cache_lock:
            self.cache[index] = body
            self.cache.move_to_end(index)
            while len(self.cache) > 4:
                self.cache.popitem(last=False)
        return body

    def start_export(self, index):
        record = self.record(index)
        if not shutil.which("ffmpeg"):
            raise ValueError("MP4 export needs FFmpeg on PATH. Playback works without it. "
                             "Install FFmpeg, then restart the viewer.")
        if record["suite"] not in ("core_layout_transfer", "fixed_training_geometry"):
            raise ValueError("The video renderer does not support this evaluation suite.")
        with self.jobs_lock:
            if self._closing:
                raise ValueError("The viewer is shutting down.")
            if any(j["status"] == "running" for j in self.jobs.values()):
                raise ValueError("An MP4 export is already running. Wait for it to finish.")
            job_id = secrets.token_hex(12)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            if self._export_temp is None:
                self._export_temp = tempfile.TemporaryDirectory(prefix="pap_viewer_exports_")
            folder = Path(self._export_temp.name) / job_id
            folder.mkdir(parents=True, exist_ok=False)
            suite = "transfer" if record["suite"] == "core_layout_transfer" else "fixed"
            method = re.sub(r"[^A-Za-z0-9_-]", "_", record["method"])[:60]
            filename = (f"PAP_{suite}_{method}_seed{record['train_seed']}_ep{record['episode']}_"
                        f"{timestamp}_{job_id[:6]}.zip")
            job = {"id": job_id, "status": "running", "pair_id": index,
                   "filename": filename, "_output_dir": str(folder)}
            self.jobs[job_id] = job
            worker = threading.Thread(target=self._export, args=(job_id, record.copy()),
                                      name="pap-video-export", daemon=False)
            self._workers.append(worker)
            worker.start()
        return self.job(job_id)

    def _export(self, job_id, record):
        folder = Path(self.jobs[job_id]["_output_dir"])
        command = [sys.executable, str(HERE / "render.py"), "--repo-root", str(self.root),
                   "--suite", record["suite"], "--method", record["method"],
                   "--train-seed", str(record["train_seed"]), "--episode", str(record["episode"]),
                   "--output-dir", str(folder)]
        try:
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                               MPLCONFIGDIR=str(folder / "matplotlib_cache"))
            with (folder / "export.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        check=False, cwd=str(folder), env=environment,
                                        start_new_session=(os.name != "nt"),
                                        creationflags=(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                                                       if os.name == "nt" else 0))
            video = folder / "PAP_Recorded_Trajectories_3D.mp4"
            if result.returncode or not video.is_file() or not video.stat().st_size:
                raise RuntimeError("Video rendering failed. Download the export log for details.")
            # Store the already-compressed MP4 without recompressing it. Only these
            # three deliverables enter the ZIP, never caches or other local files.
            with zipfile.ZipFile(folder / self.jobs[job_id]["filename"], "w") as archive:
                for name in (video.name, "selected_episodes.json", "export.log"):
                    archive.write(folder / name, name,
                                  compress_type=zipfile.ZIP_STORED if name == video.name
                                  else zipfile.ZIP_DEFLATED)
            update = {"status": "done", "download": f"/api/download?id={job_id}"}
        except Exception as error:
            update = {"status": "error", "message": str(error)}
            if (folder / "export.log").is_file():
                update["log_download"] = f"/api/export-log?id={job_id}"
        with self.jobs_lock:
            self.jobs[job_id].update(update)

    def job(self, job_id):
        with self.jobs_lock:
            if job_id not in self.jobs:
                raise ValueError("Unknown export.")
            return {key: value for key, value in self.jobs[job_id].items()
                    if not key.startswith("_")}

    def download_path(self, job_id, log=False):
        with self.jobs_lock:
            if job_id not in self.jobs:
                raise ValueError("Unknown export.")
            job = self.jobs[job_id]
            if job["status"] == "running" or (not log and job["status"] != "done"):
                raise ValueError("The export is not ready yet.")
            return Path(job["_output_dir"]) / ("export.log" if log else job["filename"])

    def close(self):
        """Remove only this session's temporary exports after workers finish."""
        with self.jobs_lock:
            self._closing = True
            workers = self._workers.copy()
        for worker in workers:
            worker.join()
        if self._export_temp is not None:
            self._export_temp.cleanup()
            self._export_temp = None


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, state, port=0):
        self.state = state
        super().__init__(("127.0.0.1", port), ViewerHandler)
        self.origin = f"http://127.0.0.1:{self.server_port}"
        self.allowed_hosts = {f"127.0.0.1:{self.server_port}", f"localhost:{self.server_port}"}


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "PAPViewer/2"

    def log_message(self, *_args):
        pass

    def _send(self, status, body=b"", content_type="application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._headers()
        self.end_headers()
        self.wfile.write(body)

    def _headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                         "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                         "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")

    def _local_request(self):
        if self.headers.get("Host") not in self.server.allowed_hosts:
            self._send(403, json_bytes({"error": "Use the local address printed by the viewer."}))
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in self.server.allowed_hosts}:
            self._send(403, json_bytes({"error": "Cross-origin requests are not allowed."}))
            return False
        return True

    @staticmethod
    def _query_id(query):
        values = parse_qs(query, keep_blank_values=True)
        if set(values) != {"id"} or len(values["id"]) != 1:
            raise ValueError("A single pair or export ID is required.")
        return values["id"][0]

    def do_GET(self):
        if not self._local_request():
            return
        route = urlsplit(self.path)
        try:
            if route.path in ("/", "/viewer.html"):
                self._send(200, (HERE / "viewer.html").read_bytes(), "text/html; charset=utf-8")
            elif route.path == "/favicon.ico":
                self._send(204)
            elif route.path == "/api/catalog":
                self._send(200, json_bytes(self.server.state.catalog_payload()))
            elif route.path == "/api/pair":
                index = int(self._query_id(route.query))
                self._send(200, self.server.state.pair_bytes(index))
            elif route.path == "/api/export":
                self._send(200, json_bytes(self.server.state.job(self._query_id(route.query))))
            elif route.path in ("/api/download", "/api/export-log"):
                is_log = route.path == "/api/export-log"
                path = self.server.state.download_path(self._query_id(route.query), log=is_log)
                with path.open("rb") as handle:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8" if is_log
                                     else "application/zip")
                    self.send_header("Content-Length", str(path.stat().st_size))
                    self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
                    self._headers()
                    self.end_headers()
                    shutil.copyfileobj(handle, self.wfile)
            else:
                self._send(404, json_bytes({"error": "Not found."}))
        except (BrokenPipeError, ConnectionResetError):
            pass  # Selecting another pair can cancel the previous request.
        except (ValueError, OSError, KeyError) as error:
            self._send(400, json_bytes({"error": str(error)}))

    def do_POST(self):
        if not self._local_request():
            return
        if self.path != "/api/export":
            self._send(404, json_bytes({"error": "Not found."}))
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096 or self.headers.get_content_type() != "application/json":
                raise ValueError("A small JSON export request is required.")
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict):
                raise ValueError("An export request must be a JSON object.")
            token = request.get("token")
            if not isinstance(token, str) or not hmac.compare_digest(token.encode(), self.server.state.token.encode()):
                self._send(403, json_bytes({"error": "Reload this page before exporting."}))
                return
            result = self.server.state.start_export(request.get("id"))
            self._send(202, json_bytes(result))
        except (ValueError, OSError) as error:
            self._send(400, json_bytes({"error": str(error)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT,
                        help="Defaults to the repository containing visualization/trajectory_viewer.")
    parser.add_argument("--port", type=int, default=0, help="Local port; 0 chooses an available port.")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser.")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if not (HERE / "viewer.html").is_file():
        parser.error("viewer.html is missing. Copy all files from the viewer overlay.")
    try:
        state = ViewerState(args.repo_root)
        server = ViewerServer(state, args.port)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cannot open the trajectory viewer: {error}\n")
    print(f"PAP trajectory explorer: {server.origin}", flush=True)
    print(f"{len(state.records):,} recorded pairs. Keep this terminal open; press Ctrl+C to stop.", flush=True)
    if state.missing:
        print(f"{state.missing:,} pairs are unavailable because trajectory archives are missing.", flush=True)
    if not args.no_browser:
        timer = threading.Timer(0.25, webbrowser.open, args=(server.origin,))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nViewer stopped. Removing temporary exports after any active renderer finishes.", flush=True)
    finally:
        server.server_close()
        state.close()


if __name__ == "__main__":
    main()
