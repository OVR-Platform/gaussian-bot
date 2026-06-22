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
from gaussian_robot.render.base import Renderer
from gaussian_robot.render.camera import Pose
from gaussian_robot.splat.scene import SceneBounds, SplatScene
from gaussian_robot.vlm.client import VLMClient

_UP_INDEX = {"x": 0, "y": 1, "z": 2}
_FLOOR_AXES = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}

_cached_renderer: tuple[str, object] | None = None


def _floor_centroid(renderer: Renderer) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    """Return (x, z) density-weighted centroid and the density bounds, or None."""
    cloud = getattr(renderer, "cloud", None)
    if cloud is None:
        return None
    grid = getattr(cloud, "density_grid", None)
    dbounds = getattr(cloud, "density_bounds", None)
    if grid is None or dbounds is None:
        return None
    lo, hi = dbounds
    gs = grid.shape[0]
    total = grid.sum()
    if total <= 0:
        return None
    rows = np.arange(gs).reshape(-1, 1)
    cols = np.arange(gs).reshape(1, -1)
    cx = float((grid * rows).sum() / total)
    cz = float((grid * cols).sum() / total)
    x = lo[0] + (cx + 0.5) / gs * (hi[0] - lo[0])
    z = lo[2] + (cz + 0.5) / gs * (hi[2] - lo[2])
    return x, z, lo, hi


def _best_origin(renderer: Renderer, up_axis: str, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    """Pick a camera origin with good coverage via density centroid + test renders."""
    from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose  # noqa: PLC0415

    intrinsics = CameraIntrinsics(fx=400, fy=400, cx=256, cy=256, width=512, height=512)
    fc = _floor_centroid(renderer)
    if fc is None:
        if hasattr(renderer, "cloud") and hasattr(renderer.cloud, "full_bounds"):
            fb = renderer.cloud.full_bounds
            return (fb.min + fb.max) / 2.0
        return (bmin + bmax) / 2.0

    x, z, lo, hi = fc
    up_idx = _UP_INDEX[up_axis]
    cloud = renderer.cloud  # type: ignore[union-attr]
    means = cloud.means
    gs = cloud.density_grid.shape[0]
    cell_w = (hi[0] - lo[0]) / gs * 2
    cell_h = (hi[2] - lo[2]) / gs * 2
    mask = (
        (means[:, 0] > x - cell_w)
        & (means[:, 0] < x + cell_w)
        & (means[:, 2] > z - cell_h)
        & (means[:, 2] < z + cell_h)
    )
    nearby_up = means[mask, up_idx]

    candidates: list[np.ndarray] = []
    base = np.array([(lo[i] + hi[i]) / 2.0 for i in range(3)])
    base[0], base[2] = x, z
    if nearby_up.numel() > 0:
        for pct in (0.20, 0.35, 0.50, 0.65, 0.80):
            c = base.copy()
            c[up_idx] = float(nearby_up.quantile(pct).cpu())
            candidates.append(c)
    else:
        candidates.append(base)

    best, best_depth = candidates[0], 0.0
    for c in candidates:
        rot = look_at(c, c, up_axis)
        result = renderer.render(Camera(pose=Pose(position=c, rotation=rot), intrinsics=intrinsics))
        if result.depth is None:
            continue
        finite = result.depth[np.isfinite(result.depth)]
        med = float(np.median(finite)) if finite.size > 0 else 0.0
        if med > best_depth:
            best, best_depth = c, med

    return best


def _load_renderer(
    ply_path: str, *, device: str = "cuda:0"
) -> tuple[Renderer, np.ndarray, np.ndarray]:
    """Load the best available renderer for a PLY file (gsplat > point cloud).

    Returns (renderer, bounds_min, bounds_max). Caches the renderer so
    repeated calls with the same path reuse the GPU tensors.
    """
    global _cached_renderer  # noqa: PLW0603
    if _cached_renderer is not None and _cached_renderer[0] == ply_path:
        r = _cached_renderer[1]
        bounds = r.cloud.bounds  # type: ignore[attr-defined]
        return r, bounds.min, bounds.max  # type: ignore[return-value]

    try:
        from gaussian_robot.backends.gsplat_renderer import GsplatRenderer  # noqa: PLC0415

        gsplat_r = GsplatRenderer.from_path(ply_path, device=device)
        _cached_renderer = (ply_path, gsplat_r)
        return gsplat_r, gsplat_r.cloud.bounds.min, gsplat_r.cloud.bounds.max
    except (ImportError, ValueError):
        pass

    from gaussian_robot.backends.ply_point import PLYPointRenderer  # noqa: PLC0415

    ply_r = PLYPointRenderer.from_path(ply_path)
    _cached_renderer = (ply_path, ply_r)
    return ply_r, ply_r.cloud.bounds.min, ply_r.cloud.bounds.max


def load_preview(config: RunConfig) -> dict[str, object]:
    """Load the scene and render a single preview frame.

    Returns a dict with ``rgb`` (JPEG data URL), ``bounds_min``, ``bounds_max``,
    and renderer info.
    """
    from gaussian_robot.vlm.qwen import jpeg_data_url  # noqa: PLC0415

    if not config.ply_path:
        raise ValueError("no ply_path configured")
    renderer, bmin, bmax = _load_renderer(config.ply_path, device=config.cuda_device)

    from gaussian_robot.render.camera import Camera, CameraIntrinsics  # noqa: PLC0415

    cam_pos = _best_origin(renderer, config.up_axis, bmin, bmax).copy()
    rot = look_at(cam_pos, cam_pos, config.up_axis)
    pose = Pose(position=cam_pos, rotation=rot)
    intrinsics = CameraIntrinsics(fx=400, fy=400, cx=256, cy=256, width=512, height=512)
    camera = Camera(pose=pose, intrinsics=intrinsics)

    from gaussian_robot.nav.observation import depth_to_uint8  # noqa: PLC0415

    result = renderer.render(camera)
    depth_panel = depth_to_uint8(result.depth)
    return {
        "rgb": jpeg_data_url(result.rgb),
        "depth": jpeg_data_url(depth_panel),
        "bounds_min": bmin.tolist(),
        "bounds_max": bmax.tolist(),
        "renderer": type(renderer).__name__,
        "ok": True,
    }


def animate_forward(config: RunConfig, n_frames: int = 20, n_steps: int = 4) -> dict[str, object]:
    """Render an animation of ``n_steps`` forward steps, interpolated into ``n_frames`` per step."""
    from gaussian_robot.nav.action import Action, ActionSpace, apply_action  # noqa: PLC0415
    from gaussian_robot.render.camera import Camera, CameraIntrinsics  # noqa: PLC0415
    from gaussian_robot.vlm.qwen import jpeg_data_url  # noqa: PLC0415

    if not config.ply_path:
        raise ValueError("no ply_path configured")
    renderer, bmin, bmax = _load_renderer(config.ply_path, device=config.cuda_device)
    diag = float(np.linalg.norm(bmax - bmin))
    space = ActionSpace(step=config.action_step_fraction * diag)
    intrinsics = CameraIntrinsics(fx=400, fy=400, cx=256, cy=256, width=512, height=512)

    origin = _best_origin(renderer, config.up_axis, bmin, bmax)

    pose = Pose(
        position=origin.copy(), rotation=look_at(origin, origin, config.up_axis)
    )
    frames: list[str] = []

    for _step_i in range(n_steps):
        next_pose = apply_action(pose, Action.FORWARD, space, config.up_axis)
        for f in range(n_frames):
            t = f / max(n_frames - 1, 1)
            pos = pose.position * (1 - t) + next_pose.position * t
            cam = Camera(pose=Pose(position=pos, rotation=pose.rotation), intrinsics=intrinsics)
            result = renderer.render(cam)
            frames.append(jpeg_data_url(result.rgb))
        pose = next_pose

    return {"frames": frames, "ok": True, "step_size": space.step, "n_steps": n_steps}


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


def generate_seeds(
    config: RunConfig,
    bounds_min: np.ndarray | None = None,
    bounds_max: np.ndarray | None = None,
    origin: np.ndarray | None = None,
) -> list[Pose]:
    """Generate seed pose candidates spread across the scene.

    Produces a grid of positions inside the bounds (like the original), plus
    extra candidates at ``origin`` facing different directions. The caller
    should validate these with a test render and drop degenerate ones.
    """
    a, b = _FLOOR_AXES[config.up_axis]
    bmin = np.array(config.bounds_min if bounds_min is None else bounds_min, dtype=np.float64)
    bmax = np.array(config.bounds_max if bounds_max is None else bounds_max, dtype=np.float64)
    safe_origin = (bmin + bmax) / 2.0 if origin is None else origin.copy()

    # All seeds from the known-good origin, facing different directions.
    # The exploration walks will move the robot to other positions.
    n_candidates = max(config.num_seeds * 2, 8)
    candidates: list[Pose] = []
    for i in range(n_candidates):
        angle = 2.0 * math.pi * i / n_candidates
        target = safe_origin.copy()
        target[a] += math.cos(angle)
        target[b] += math.sin(angle)
        candidates.append(
            Pose(position=safe_origin.copy(), rotation=look_at(safe_origin, target, config.up_axis))
        )

    return candidates


def build_vlm(config: RunConfig) -> VLMClient:
    """Select the demo VLM or the real Qwen vLLM client (lazy import)."""
    if config.use_real_vlm:
        from gaussian_robot.vlm.qwen import QwenVLMClient  # noqa: PLC0415

        return QwenVLMClient(
            base_url=config.vlm_base_url,
            model=config.vlm_model,
            temperature=config.vlm_temperature,
            top_p=config.vlm_top_p,
            top_k=config.vlm_top_k,
            min_p=config.vlm_min_p,
            presence_penalty=config.vlm_presence_penalty,
            repetition_penalty=config.vlm_repetition_penalty,
            enable_thinking=config.vlm_enable_thinking,
            max_history_turns=config.vlm_max_history_turns,
        )
    return ScriptedDemoVLM()


def build_session(config: RunConfig) -> tuple[Explorer, list[Pose], CoverageState]:
    """Construct an :class:`Explorer`, seeds and a fresh coverage state from config.

    The renderer is always the demo fake for now (gsplat is not yet wired); the
    VLM is the demo script unless ``config.use_real_vlm`` is set.
    """
    bmin = np.array(config.bounds_min, dtype=np.float64)
    bmax = np.array(config.bounds_max, dtype=np.float64)
    renderer: Renderer = FakeRenderer()
    if config.ply_path:
        renderer, bmin, bmax = _load_renderer(config.ply_path, device=config.cuda_device)
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
    vlm = build_vlm(config)
    builder = ObservationBuilder(
        renderer=renderer,
        up_axis=config.up_axis,
        map_size=config.map_size,
        task=config.task_prompt,
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
    seed_origin = _best_origin(renderer, config.up_axis, bmin, bmax)

    from gaussian_robot.render.camera import Camera, CameraIntrinsics  # noqa: PLC0415

    intrinsics = CameraIntrinsics(fx=400, fy=400, cx=256, cy=256, width=512, height=512)
    candidates = generate_seeds(config, bmin, bmax, origin=seed_origin)
    seeds: list[Pose] = []
    for s in candidates:
        if len(seeds) >= config.num_seeds:
            break
        result = renderer.render(Camera(pose=s, intrinsics=intrinsics))
        if result.depth is None:
            continue
        finite_frac = float(np.isfinite(result.depth).mean())
        near = result.depth[np.isfinite(result.depth)]
        median_depth = float(np.median(near)) if near.size > 0 else 0.0
        if finite_frac < 0.5 or median_depth < space.step:
            continue
        seeds.append(s)
    if not seeds:
        seeds = [candidates[0]]

    coverage = CoverageState.empty(config.up_axis, bmin, bmax)
    return explorer, seeds, coverage
