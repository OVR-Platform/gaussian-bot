"""Manage a local vLLM OpenAI-compatible server process."""

from __future__ import annotations

import subprocess
import time
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
        return self.status()

    def stop(self) -> VLLMStatus:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        return self.status()

    def status(self) -> VLLMStatus:
        process = self._process
        if process is None:
            return VLLMStatus(running=False, pid=None, command=self._command, log_tail=_tail_log())
        returncode = process.poll()
        return VLLMStatus(
            running=returncode is None,
            pid=process.pid,
            command=self._command,
            returncode=returncode,
            log_tail=_tail_log(),
        )


def vllm_command(config: RunConfig) -> list[str]:
    """Build the vLLM OpenAI-compatible server command."""
    return [
        _vllm_executable(),
        "serve",
        config.vlm_model,
        "--host",
        config.vllm_host,
        "--port",
        str(config.vllm_port),
        *config.vllm_extra_args,
    ]


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
