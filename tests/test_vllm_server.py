"""Tests for vLLM server command wiring."""

from __future__ import annotations

from gaussian_robot.config import RunConfig
from gaussian_robot.vlm.server import vllm_command


def test_vllm_command_uses_configured_model_host_and_port() -> None:
    config = RunConfig(
        vlm_model="Qwen/Qwen3.5-9B",
        vllm_host="0.0.0.0",
        vllm_port=8000,
        vllm_extra_args=["--dtype", "auto"],
    )

    assert vllm_command(config) == [
        "vllm",
        "serve",
        "Qwen/Qwen3.5-9B",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "auto",
    ]
