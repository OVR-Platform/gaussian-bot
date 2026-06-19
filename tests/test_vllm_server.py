"""Tests for vLLM server command wiring."""

from __future__ import annotations

from pathlib import Path

from gaussian_robot.config import RunConfig
from gaussian_robot.vlm.server import _vllm_executable, vllm_command


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
