"""Manage a local vLLM OpenAI-compatible server process."""

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from shutil import which

from gaussian_robot.config import RunConfig

_VLLM_INSTALL_HINT = (
    "Run `uv sync --extra vlm --extra vllm` on this machine, then restart the dashboard."
)
_VLLM_LOG_PATH = Path("data/vllm.log")


@dataclass(frozen=True)
class VLLMStatus:
    """Current vLLM process state."""

    running: bool
    pid: int | None
    command: list[str]
    returncode: int | None = None
    ready: bool = False
    log_path: str = str(_VLLM_LOG_PATH)
    log_tail: str = ""


class VLLMServerProcess:
    """Owns one child ``vllm serve`` process."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._command: list[str] = []

    def start(self, config: RunConfig) -> VLLMStatus:
        process = self._process
        if process is not None and process.poll() is None:
            return self.status()

        command = vllm_command(config)
        _VLLM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_file = _VLLM_LOG_PATH.open("ab")
        try:
            self._process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        except FileNotFoundError as exc:
            raise RuntimeError(f"vllm executable not found. {_VLLM_INSTALL_HINT}") from exc
        finally:
            log_file.close()
        self._command = command
        time.sleep(1.0)
        return self.status(config)

    def stop(self) -> VLLMStatus:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        return self.status()

    def status(self, config: RunConfig | None = None) -> VLLMStatus:
        process = self._process
        if process is None:
            return VLLMStatus(running=False, pid=None, command=self._command, log_tail=_tail_log())
        returncode = process.poll()
        running = returncode is None
        return VLLMStatus(
            running=running,
            pid=process.pid,
            command=self._command,
            returncode=returncode,
            ready=running and config is not None and _server_ready(config),
            log_tail=_tail_log(),
        )


def vllm_command(config: RunConfig) -> list[str]:
    """Build the vLLM OpenAI-compatible server command."""
    extra_args = _vllm_extra_args(config)
    return [
        _vllm_executable(),
        "serve",
        config.vlm_model,
        "--host",
        config.vllm_host,
        "--port",
        str(config.vllm_port),
        *extra_args,
    ]


def _vllm_extra_args(config: RunConfig) -> list[str]:
    args = list(config.vllm_extra_args)
    if _needs_transformers_model_impl(config.vlm_model) and "--model-impl" not in args:
        args.extend(["--model-impl", "transformers"])
    return args


def _needs_transformers_model_impl(model: str) -> bool:
    normalized = model.lower().replace("_", "").replace("-", "")
    return "qwen/qwen3.5" in normalized or "qwen3.5" in normalized


def _vllm_executable(candidates: Sequence[Path] | None = None) -> str:
    candidate_paths = candidates or [Path(".venv/bin/vllm")]
    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)
    found = which("vllm")
    return found or "vllm"


def _tail_log(path: Path = _VLLM_LOG_PATH, *, max_bytes: int = 4000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        return fh.read().decode("utf-8", errors="replace")


def _server_ready(config: RunConfig, *, timeout: float = 0.5) -> bool:
    host = "127.0.0.1" if config.vllm_host in {"0.0.0.0", "::"} else config.vllm_host
    url = f"http://{host}:{config.vllm_port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = int(response.status)
            return 200 <= status < 500
    except (OSError, urllib.error.URLError):
        return False
