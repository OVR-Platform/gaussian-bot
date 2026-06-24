"""Tiny stdlib dashboard server."""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from gaussian_robot.config import load_config, save_config
from gaussian_robot.events import SessionEvent, SessionStartEvent, StepEvent
from gaussian_robot.render.camera import Pose
from gaussian_robot.session import animate_forward, build_session, load_preview, render_walk_movie
from gaussian_robot.ui.page import PAGE_HTML
from gaussian_robot.ui.serialize import event_to_message
from gaussian_robot.vlm.server import VLLMServerProcess, VLLMStatus

_VLLM = VLLMServerProcess()
_step_gate = threading.Event()
_step_mode = False

# Full per-walk pose trajectories (position + rotation) captured live from the
# event stream, so the dashboard can replay an interpolated fly-through of any walk.
_WALKS: dict[str, list[Pose]] = {}


def serve_dashboard(host: str, port: int, *, start_vllm: bool = False) -> None:
    """Serve the dashboard until interrupted."""
    if start_vllm:
        _VLLM.start(load_config())
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "gaussian-robot-ui/0.1"

    def do_HEAD(self) -> None:  # noqa: N802
        if self.path == "/":
            body = PAGE_HTML.encode()
            self._send_headers("text/html; charset=utf-8", len(body))
            return
        if self.path == "/api/config":
            body = json.dumps(load_config().model_dump(mode="json")).encode()
            self._send_headers("application/json; charset=utf-8", len(body))
            return
        if self.path == "/api/vllm/status":
            body = json.dumps(_status_payload(_VLLM.status(load_config()))).encode()
            self._send_headers("application/json; charset=utf-8", len(body))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:  # noqa: N802
        routes = {
            "/": lambda: self._send_bytes(PAGE_HTML.encode(), "text/html; charset=utf-8"),
            "/api/config": lambda: self._send_json(load_config().model_dump(mode="json")),
            "/api/vllm/status": lambda: self._send_json(
                _status_payload(_VLLM.status(load_config()))
            ),
            "/api/load": self._load_scene,
            "/api/animate-forward": self._animate_forward,
            "/api/walks": self._list_walks,
        }
        path = urlparse(self.path).path
        handler = routes.get(path)
        if handler is not None:
            handler()
        elif path == "/api/walk-movie":
            self._walk_movie(parse_qs(urlparse(self.path).query))
        elif self.path in ("/api/run", "/api/run/stepping"):
            self._stream_run(step_mode=(self.path == "/api/run/stepping"))
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _list_walks(self) -> None:
        self._send_json({"walks": [{"walk_id": w, "frames": len(p)} for w, p in _WALKS.items()]})

    def _walk_movie(self, query: dict[str, list[str]]) -> None:
        walk_id = (query.get("walk") or [""])[0]
        poses = _WALKS.get(walk_id)
        if not poses:
            self._send_json(
                {"ok": False, "error": f"no frames recorded for {walk_id!r}", "frames": []},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        try:
            per = int((query.get("per") or ["8"])[0])
            result = render_walk_movie(load_config(), poses, per_segment=max(1, per))
            result["walk_id"] = walk_id
            self._send_json(result)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/step/next":
            _step_gate.set()
            self._send_json({"ok": True})
            return
        if self.path == "/api/vllm/start":
            self._start_vllm()
            return
        if self.path == "/api/vllm/stop":
            self._send_json(_status_payload(_VLLM.stop()))
            return
        if self.path != "/api/config":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            config = load_config().overrides(data)
            config = type(config).model_validate(config.model_dump(mode="python"))
            path = save_config(config)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json({"ok": True, "path": str(path), "config": config.model_dump(mode="json")})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _load_scene(self) -> None:
        try:
            result = load_preview(load_config())
            self._send_json(result)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _animate_forward(self) -> None:
        try:
            result = animate_forward(load_config())
            self._send_json(result)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _stream_run(self, *, step_mode: bool) -> None:
        global _step_mode  # noqa: PLW0603
        _step_mode = step_mode
        if step_mode:
            _step_gate.clear()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def emit(event: SessionEvent) -> None:
            # Capture full poses (with rotation) for interpolated walk replay.
            if isinstance(event, SessionStartEvent):
                _WALKS.clear()
            elif isinstance(event, StepEvent):
                _WALKS.setdefault(event.walk_id, []).append(event.pose)
            self._write_sse(event_to_message(event))
            if _step_mode and isinstance(event, StepEvent):
                _step_gate.clear()
                _step_gate.wait()

        try:
            config = load_config()
            if config.start_vllm:
                _VLLM.start(config)
            explorer, seeds, coverage = build_session(config)
            explorer.event_sink = emit
            if step_mode:
                _step_gate.set()
            explorer.run_session(seeds, coverage, requested_seeds=config.num_seeds)
        except BrokenPipeError:
            return
        except Exception as exc:  # noqa: BLE001
            try:
                self._write_sse({"type": "error", "message": str(exc)})
            except BrokenPipeError:
                return
        finally:
            _step_mode = False

    def _start_vllm(self) -> None:
        try:
            status = _VLLM.start(load_config())
        except RuntimeError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if not status.running:
            self._send_json(
                {
                    "ok": False,
                    "error": f"vLLM exited during startup with code {status.returncode}",
                    **_status_payload(status),
                },
                status=HTTPStatus.BAD_REQUEST,
            )
            return
        self._send_json({"ok": status.ready, **_status_payload(status)})

    def _write_sse(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message, separators=(",", ":"))
        self.wfile.write(f"data: {payload}\n\n".encode())
        self.wfile.flush()

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(
            json.dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_headers(
        self,
        content_type: str,
        content_length: int,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.end_headers()


def _status_payload(status: VLLMStatus) -> dict[str, Any]:
    return {
        "running": status.running,
        "pid": status.pid,
        "command": status.command,
        "returncode": status.returncode,
        "ready": status.ready,
        "log_path": status.log_path,
        "log_tail": status.log_tail,
    }
