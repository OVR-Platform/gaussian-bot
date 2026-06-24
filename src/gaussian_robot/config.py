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
    poses_path: str | None = Field(
        default=None,
        description=(
            "Path to the capture poses the splat was reconstructed from "
            "(3DGS cameras.json, COLMAP images.bin/.txt, or a directory). "
            "Seeds start from these known-good viewpoints. If unset, auto-discovered "
            "next to ply_path."
        ),
    )
    use_capture_pose_seeds: bool = Field(
        default=True,
        description="Seed walks from capture camera poses when available (falls back to density).",
    )
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

    cuda_device: str = Field(
        default="cuda:0", description="CUDA device for the splat renderer, e.g. cuda:0, cuda:1."
    )

    up_axis: str = Field(
        default="auto",
        description=(
            "World up axis: 'auto' detects it from the capture poses, or set it "
            "explicitly (x/y/z, optionally signed e.g. -y)."
        ),
    )
    bounds_min: tuple[float, float, float] = Field(default=(0.0, 0.0, 0.0))
    bounds_max: tuple[float, float, float] = Field(default=(10.0, 10.0, 10.0))

    action_step_fraction: float = Field(
        default=0.015,
        description="Step as a fraction of AABB diagonal. Smaller = finer, more deliberate motion.",
    )
    coverage_radius: float | None = Field(
        default=None, description="Coverage radius; None -> derived from the step length."
    )
    max_steps: int = Field(
        default=120, ge=1, description="Max steps per walk. Higher = longer, deeper walks."
    )
    actions_per_query: int = Field(
        default=4,
        ge=1,
        description="Action chunking: max actions the VLM plans per query. The plan runs "
        "step-by-step and is re-queried when exhausted or interrupted (blocked / degenerate / "
        "describe). Higher = fewer VLM calls/tokens, less reactive. 1 = decide every step.",
    )
    pose_budget: int = Field(default=400, ge=1)
    pose_target: int = Field(
        default=30,
        ge=1,
        description="Target number of fill-in poses to mark (informational goal, not a stop).",
    )
    num_seeds: int = Field(
        default=3,
        ge=1,
        description="Number of walk seeds. Fewer = budget spent exploring deeply, not restarting.",
    )
    coverage_3d: bool = Field(
        default=True,
        description="Build a Tier-3 3D coverage grid (voxel occupancy + capture-camera "
        "frustum raycast) to find occupied-but-unseen regions (roofs/floors/behind-buildings) "
        "and aim the aerial survey at them. Falls back to tallest-geometry when off/unavailable.",
    )
    aerial_survey: bool = Field(
        default=True,
        description="Add one extra walk that starts high above the tallest geometry looking "
        "down, to survey rooftops/upper structure that ground-level walks never reach.",
    )
    terrain_follow: bool = Field(
        default=True,
        description="Keep the camera at a constant eye-height above local ground on "
        "non-flat scenes (uses a one-time gaussian height field). Disable for flat scenes.",
    )
    live_tween_frames: int = Field(
        default=0,
        ge=0,
        description="Interpolated RGB frames rendered between views to smooth the live "
        "dashboard motion. 0 disables (default: each frame is an extra GPU render per step, "
        "the biggest per-step cost after the VLM). The on-demand walk-replay fly-through "
        "interpolates independently, so smoothing is kept where it matters. Raise for a "
        "smoother live view at the cost of speed.",
    )
    map_size: int = Field(default=512, ge=64)
    map_span: float | None = Field(
        default=None,
        description=(
            "World units across the body-fixed map. None -> a local window of "
            "~10 steps, so each step is clearly visible (instead of the whole scene)."
        ),
    )
    task_prompt: str = Field(
        default="", description="Optional task for the robot, e.g. 'find the office door'."
    )

    use_depth_estimator: bool = Field(
        default=False,
        description="If true, run Depth Anything 3 on each render for the depth panel.",
    )
    depth_model: str = Field(
        default="depth-anything/DA3-BASE",
        description="HuggingFace model id for Depth Anything 3.",
    )

    vlm_temperature: float = Field(default=1.0, description="Sampling temperature.")
    vlm_top_p: float = Field(default=0.95, description="Nucleus sampling top-p.")
    vlm_top_k: int = Field(default=20, ge=0, description="Top-k sampling.")
    vlm_min_p: float = Field(default=0.0, ge=0.0, description="Min-p sampling threshold.")
    vlm_presence_penalty: float = Field(default=0.0, description="Presence penalty.")
    vlm_repetition_penalty: float = Field(default=1.0, description="Repetition penalty.")
    vlm_enable_thinking: bool = Field(
        default=False, description="If true, enable 'thinking' mode in the chat template."
    )
    vlm_max_history_turns: int = Field(
        default=3,
        ge=0,
        description="Max conversation turns kept for multi-turn VLM history. 0 = stateless.",
    )

    @field_validator("up_axis")
    @classmethod
    def _check_axis(cls, v: str) -> str:
        from gaussian_robot.render.camera import parse_up_axis  # noqa: PLC0415

        s = v.strip().lower()
        if s == "auto":
            return s  # resolved from the capture poses at session-build time
        parse_up_axis(s)  # raises ValueError for anything but (optionally signed) x/y/z
        return s

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
