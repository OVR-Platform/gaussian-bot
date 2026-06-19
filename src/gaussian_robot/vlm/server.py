"""Manage a local vLLM OpenAI-compatible server process."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from gaussian_robot.config import RunConfig


@dataclass(frozen=True)
class VLLMStatus:
    """Current vLLM process state."""

    running: bool
    pid: int | None
    command: list[str]
    returncode: int | None = None


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
        log_path = Path("data/vllm.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("ab")
        try:
            self._process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "vllm executable not found. Install vLLM in this environment before launching it."
            ) from exc
        finally:
            log_file.close()
        self._command = command
        return self.status()

    def stop(self) -> VLLMStatus:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        return self.status()

    def status(self) -> VLLMStatus:
        process = self._process
        if process is None:
            return VLLMStatus(running=False, pid=None, command=self._command)
        returncode = process.poll()
        return VLLMStatus(
            running=returncode is None,
            pid=process.pid,
            command=self._command,
            returncode=returncode,
        )


def vllm_command(config: RunConfig) -> list[str]:
    """Build the vLLM OpenAI-compatible server command."""
    return [
        "vllm",
        "serve",
        config.vlm_model,
        "--host",
        config.vllm_host,
        "--port",
        str(config.vllm_port),
        *config.vllm_extra_args,
    ]
