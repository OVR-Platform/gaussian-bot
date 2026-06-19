"""Run configuration for a session (ADR: config/IO uses pydantic).

Captures everything the dashboard's config panel edits: the scene (PLY) path,
the vLLM endpoint, and the exploration knobs. Persisted as JSON so the GPU
machine can pick up the same settings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_PATH = Path("data/ui_config.json")


class RunConfig(BaseModel):
    """All inputs needed to build and run an exploration session."""

    ply_path: str | None = Field(default=None, description="Path to the .ply/.splat scene.")
    vlm_base_url: str = Field(
        default="http://localhost:8000/v1", description="OpenAI-compatible vLLM endpoint."
    )
    vlm_model: str = Field(default="Qwen/Qwen3.5-9B", description="Model id served by vLLM.")
    use_real_vlm: bool = Field(
        default=False, description="If true, call the vLLM endpoint; else use the demo VLM."
    )
    start_vllm: bool = Field(
        default=False, description="If true, launch vLLM from the dashboard server."
    )
    vllm_host: str = Field(default="0.0.0.0", description="Bind host for the vLLM server.")
    vllm_port: int = Field(default=8000, ge=1, le=65535)
    vllm_extra_args: list[str] = Field(default_factory=list)

    up_axis: str = Field(default="y")
    bounds_min: tuple[float, float, float] = Field(default=(0.0, 0.0, 0.0))
    bounds_max: tuple[float, float, float] = Field(default=(10.0, 10.0, 10.0))

    action_step_fraction: float = Field(
        default=0.03, description="Step as a fraction of AABB diagonal."
    )
    coverage_radius: float | None = Field(
        default=None, description="Coverage radius; None -> derived from the step length."
    )
    max_steps: int = Field(default=40, ge=1)
    pose_budget: int = Field(default=200, ge=1)
    num_seeds: int = Field(default=5, ge=1)
    map_size: int = Field(default=512, ge=64)

    @field_validator("up_axis")
    @classmethod
    def _check_axis(cls, v: str) -> str:
        if v.lower() not in {"x", "y", "z"}:
            raise ValueError("up_axis must be one of x/y/z")
        return v.lower()

    def overrides(self, data: dict[str, Any]) -> RunConfig:
        """Return a copy with non-None fields from ``data`` applied."""
        update = {k: v for k, v in data.items() if v is not None}
        return self.model_copy(update=update)


def load_config(path: Path | None = None) -> RunConfig:
    """Load a :class:`RunConfig` from JSON, or defaults if missing/unreadable."""
    p = path or DEFAULT_CONFIG_PATH
    if not p.exists():
        return RunConfig()
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return RunConfig()
    return RunConfig.model_validate(raw)


def save_config(config: RunConfig, path: Path | None = None) -> Path:
    """Persist ``config`` as JSON and return the path written."""
    p = path or DEFAULT_CONFIG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config.model_dump(mode="json"), indent=2))
    return p
