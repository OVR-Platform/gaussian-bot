"""Tiny stdlib dashboard server."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pydantic import ValidationError

from gaussian_robot.config import load_config, save_config
from gaussian_robot.events import SessionEvent
from gaussian_robot.session import build_session
from gaussian_robot.ui.page import PAGE_HTML
from gaussian_robot.ui.serialize import event_to_message
from gaussian_robot.vlm.server import VLLMServerProcess, VLLMStatus

_VLLM = VLLMServerProcess()


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
            body = json.dumps(_status_payload(_VLLM.status())).encode()
            self._send_headers("application/json; charset=utf-8", len(body))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._send_bytes(PAGE_HTML.encode(), "text/html; charset=utf-8")
            return
        if self.path == "/api/config":
            self._send_json(load_config().model_dump(mode="json"))
            return
        if self.path == "/api/vllm/status":
            self._send_json(_status_payload(_VLLM.status()))
            return
        if self.path == "/api/run":
            self._stream_run()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
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

    def _stream_run(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def emit(event: SessionEvent) -> None:
            self._write_sse(event_to_message(event))

        try:
            config = load_config()
            if config.start_vllm:
                _VLLM.start(config)
            explorer, seeds, coverage = build_session(config)
            explorer.event_sink = emit
            explorer.run_session(seeds, coverage)
        except BrokenPipeError:
            return
        except Exception as exc:  # noqa: BLE001
            try:
                self._write_sse({"type": "error", "message": str(exc)})
            except BrokenPipeError:
                return

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
        self._send_json({"ok": True, **_status_payload(status)})

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
        "log_path": status.log_path,
        "log_tail": status.log_tail,
    }
