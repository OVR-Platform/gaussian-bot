"""Build a runnable session from a :class:`RunConfig`.

This is the single wiring point: it selects backends from the config (demo
fakes now; a real Qwen vLLM client when enabled; a real gsplat renderer when it
lands), builds the :class:`Explorer`, and spreads seed poses inside the scene.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from gaussian_robot.backends.demo import FakeRenderer, ScriptedDemoVLM
from gaussian_robot.config import RunConfig
from gaussian_robot.metrics.coverage import CoverageState
from gaussian_robot.nav.action import ActionSpace
from gaussian_robot.nav.explorer import Explorer
from gaussian_robot.nav.observation import ObservationBuilder
from gaussian_robot.nav.stop import (
    BoundsGuard,
    CoveragePlateau,
    CoverageTarget,
    PoseBudget,
    SeedExhaustion,
    SessionStopPolicy,
    StopPolicy,
)
from gaussian_robot.render.camera import Pose
from gaussian_robot.splat.scene import SceneBounds, SplatScene
from gaussian_robot.vlm.client import VLMClient

_UP_INDEX = {"x": 0, "y": 1, "z": 2}
_FLOOR_AXES = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}


def look_at(origin: np.ndarray, target: np.ndarray, up_axis: str) -> np.ndarray:
    """World->camera rotation (OpenCV) whose camera sits at ``origin`` looking at ``target``."""
    up = np.zeros(3)
    up[_UP_INDEX[up_axis]] = 1.0
    forward = np.asarray(target, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    forward -= up * float(forward @ up)
    fn = float(np.linalg.norm(forward))
    if fn < 1e-9:
        forward = np.zeros(3)
        forward[_FLOOR_AXES[up_axis][1]] = 1.0
    else:
        forward = forward / fn
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(right, forward)
    return np.stack([right, down, forward], axis=0)


def generate_seeds(config: RunConfig) -> list[Pose]:
    """Spread ``num_seeds`` interior poses, each aimed at the scene centre."""
    bmin = np.array(config.bounds_min, dtype=np.float64)
    bmax = np.array(config.bounds_max, dtype=np.float64)
    up_idx = _UP_INDEX[config.up_axis]
    a, b = _FLOOR_AXES[config.up_axis]
    center = (bmin + bmax) / 2.0

    side = max(1, math.ceil(math.sqrt(config.num_seeds)))
    seeds: list[Pose] = []
    for i in range(side):
        for j in range(side):
            if len(seeds) >= config.num_seeds:
                break
            fa = (i + 1) / (side + 1)
            fb = (j + 1) / (side + 1)
            pos = center.copy()
            pos[a] = bmin[a] + fa * (bmax[a] - bmin[a])
            pos[b] = bmin[b] + fb * (bmax[b] - bmin[b])
            pos[up_idx] = center[up_idx]
            seeds.append(Pose(position=pos, rotation=look_at(pos, center, config.up_axis)))
    return seeds


def build_vlm(config: RunConfig) -> VLMClient:
    """Select the demo VLM or the real Qwen vLLM client (lazy import)."""
    if config.use_real_vlm:
        from gaussian_robot.vlm.qwen import QwenVLMClient  # noqa: PLC0415

        return QwenVLMClient(base_url=config.vlm_base_url, model=config.vlm_model)
    return ScriptedDemoVLM()


def build_session(config: RunConfig) -> tuple[Explorer, list[Pose], CoverageState]:
    """Construct an :class:`Explorer`, seeds and a fresh coverage state from config.

    The renderer is always the demo fake for now (gsplat is not yet wired); the
    VLM is the demo script unless ``config.use_real_vlm`` is set.
    """
    bmin = np.array(config.bounds_min, dtype=np.float64)
    bmax = np.array(config.bounds_max, dtype=np.float64)
    diag = float(np.linalg.norm(bmax - bmin))
    if diag <= 0:
        raise ValueError("bounds_max must exceed bounds_min")

    space = ActionSpace(step=config.action_step_fraction * diag)
    coverage_radius = config.coverage_radius or (space.step * 2.0)

    scene = SplatScene(
        path=Path(config.ply_path) if config.ply_path else Path("data/scene.ply"),
        bounds=SceneBounds(min=bmin, max=bmax),
        up_axis=config.up_axis,
    )
    renderer = FakeRenderer()
    vlm = build_vlm(config)
    builder = ObservationBuilder(
        renderer=renderer, up_axis=config.up_axis, map_size=config.map_size
    )

    walk_policies: list[StopPolicy] = [
        CoveragePlateau(novelty_delta=space.step, window=5),
        BoundsGuard(),
    ]
    session_policies: list[SessionStopPolicy] = [
        PoseBudget(max_poses=config.pose_budget),
        CoverageTarget(radius=coverage_radius, tau=0.8),
        SeedExhaustion(),
    ]
    explorer = Explorer(
        scene=scene,
        renderer=renderer,
        vlm=vlm,
        observation_builder=builder,
        action_space=space,
        coverage_radius=coverage_radius,
        walk_policies=walk_policies,
        session_policies=session_policies,
        max_steps=config.max_steps,
    )
    coverage = CoverageState.empty(config.up_axis, bmin, bmax)
    return explorer, generate_seeds(config), coverage
