"""Tests for vLLM server command wiring."""

from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from gaussian_robot.config import RunConfig
from gaussian_robot.vlm.server import (
    VLLMServerProcess,
    _server_ready,
    _vllm_executable,
    vllm_command,
)


def test_vllm_command_uses_configured_model_host_and_port() -> None:
    config = RunConfig(
        vlm_model="Qwen/Qwen3.5-9B",
        vllm_host="0.0.0.0",
        vllm_port=8000,
        vllm_extra_args=["--dtype", "auto"],
    )

    assert vllm_command(config) == [
        _vllm_executable(),
        "serve",
        "Qwen/Qwen3.5-9B",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "auto",
    ]


def test_vllm_executable_prefers_project_venv(tmp_path: Path) -> None:
    vllm = tmp_path / "vllm"
    vllm.touch()

    assert _vllm_executable([vllm]) == str(vllm)


def test_vllm_start_reports_immediate_exit(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    fake = tmp_path / "vllm"
    fake.write_text("#!/bin/sh\necho startup failed\nexit 42\n")
    fake.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))

    status = VLLMServerProcess().start(RunConfig())

    assert not status.running
    assert not status.ready
    assert status.returncode == 42
    assert "startup failed" in status.log_tail


def test_vllm_ready_is_false_when_models_endpoint_is_unreachable() -> None:
    config = RunConfig(vllm_host="0.0.0.0", vllm_port=9)

    assert not _server_ready(config, timeout=0.01)
