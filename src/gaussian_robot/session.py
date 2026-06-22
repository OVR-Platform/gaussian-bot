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
    QualityTarget,
    SeedExhaustion,
    SessionStopPolicy,
    StopPolicy,
    StuckGuard,
)
from gaussian_robot.render.base import Renderer, RenderResult
from gaussian_robot.render.camera import CameraIntrinsics, Pose
from gaussian_robot.splat.scene import SceneBounds, SplatScene
from gaussian_robot.vlm.client import VLMClient

_UP_INDEX = {"x": 0, "y": 1, "z": 2}
_FLOOR_AXES = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}

# Intrinsics shared by every single-frame preview/animation render.
_PREVIEW_INTRINSICS = CameraIntrinsics(fx=400, fy=400, cx=256, cy=256, width=512, height=512)

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


def _best_origin(
    renderer: Renderer, up_axis: str, bmin: np.ndarray, bmax: np.ndarray
) -> np.ndarray:
    """Pick a camera origin with good coverage via density centroid + test renders."""
    from gaussian_robot.render.camera import Camera, Pose  # noqa: PLC0415

    intrinsics = _PREVIEW_INTRINSICS
    fc = _floor_centroid(renderer)
    if fc is None:
        if hasattr(renderer, "cloud") and hasattr(renderer.cloud, "full_bounds"):
            fb = renderer.cloud.full_bounds
            return np.asarray((fb.min + fb.max) / 2.0)
        return np.asarray((bmin + bmax) / 2.0)

    x, z, lo, hi = fc
    up_idx = _UP_INDEX[up_axis]
    cloud = renderer.cloud  # type: ignore[attr-defined]
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

    best, best_score = candidates[0], -1.0
    for c in candidates:
        rot = look_at(c, c, up_axis)
        result = renderer.render(Camera(pose=Pose(position=c, rotation=rot), intrinsics=intrinsics))
        if result.depth is None:
            continue
        finite = result.depth[np.isfinite(result.depth)]
        med = float(np.median(finite)) if finite.size > 0 else 0.0
        alpha_mean = float(result.alpha.mean()) if result.alpha is not None else 0.5
        # Score combines depth (scene visible) with alpha (scene reconstructed).
        score = med * alpha_mean
        if score > best_score:
            best, best_score = c, score

    return best


def _load_renderer(
    ply_path: str, *, device: str = "cuda:0"
) -> tuple[Renderer, np.ndarray, np.ndarray]:
    """Load the best available renderer for a PLY file (gsplat > point cloud).

    Returns (renderer, bounds_min, bounds_max). Caches the renderer so
    repeated calls with the same path reuse the GPU tensors.
    """
    global _cached_renderer  # noqa: PLW0603
    if _cached_renderer is None or _cached_renderer[0] != ply_path:
        try:
            from gaussian_robot.backends.gsplat_renderer import GsplatRenderer  # noqa: PLC0415

            r: Renderer = GsplatRenderer.from_path(ply_path, device=device)
        except (ImportError, ValueError):
            from gaussian_robot.backends.ply_point import PLYPointRenderer  # noqa: PLC0415

            r = PLYPointRenderer.from_path(ply_path)
        _cached_renderer = (ply_path, r)

    r = _cached_renderer[1]  # type: ignore[assignment]
    bounds = r.cloud.bounds  # type: ignore[attr-defined]
    return r, bounds.min, bounds.max


def load_preview(config: RunConfig) -> dict[str, object]:
    """Load the scene and render a single preview frame.

    Returns a dict with ``rgb`` (JPEG data URL), ``bounds_min``, ``bounds_max``,
    and renderer info.
    """
    from gaussian_robot.vlm.qwen import jpeg_data_url  # noqa: PLC0415

    if not config.ply_path:
        raise ValueError("no ply_path configured")
    renderer, bmin, bmax = _load_renderer(config.ply_path, device=config.cuda_device)

    from gaussian_robot.render.camera import Camera  # noqa: PLC0415

    cam_pos = _best_origin(renderer, config.up_axis, bmin, bmax).copy()
    rot = look_at(cam_pos, cam_pos, config.up_axis)
    pose = Pose(position=cam_pos, rotation=rot)
    camera = Camera(pose=pose, intrinsics=_PREVIEW_INTRINSICS)

    from gaussian_robot.nav.observation import depth_to_uint8  # noqa: PLC0415

    result = renderer.render(camera)
    if config.use_depth_estimator:
        from gaussian_robot.depth.estimator import DA3DepthEstimator  # noqa: PLC0415

        estimator = DA3DepthEstimator(
            model_name=config.depth_model,
            device=config.cuda_device,
        )
        da3_depth = estimator.estimate(result.rgb)
        result = RenderResult(
            rgb=result.rgb,
            camera=result.camera,
            depth=da3_depth,
            alpha=result.alpha,
        )
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
    from gaussian_robot.render.camera import Camera  # noqa: PLC0415
    from gaussian_robot.vlm.qwen import jpeg_data_url  # noqa: PLC0415

    if not config.ply_path:
        raise ValueError("no ply_path configured")
    renderer, bmin, bmax = _load_renderer(config.ply_path, device=config.cuda_device)
    diag = float(np.linalg.norm(bmax - bmin))
    space = ActionSpace(step=config.action_step_fraction * diag)
    intrinsics = _PREVIEW_INTRINSICS

    origin = _best_origin(renderer, config.up_axis, bmin, bmax)

    pose = Pose(position=origin.copy(), rotation=look_at(origin, origin, config.up_axis))
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


def _floor_seed_positions(
    bmin: np.ndarray,
    bmax: np.ndarray,
    up_axis: str,
    n: int,
    height: float = 0.0,
) -> list[np.ndarray]:
    """Spread ``n`` seed positions on a grid across the floor AABB at ``height``."""
    a, b = _FLOOR_AXES[up_axis]
    up_idx = _UP_INDEX[up_axis]
    side = max(1, math.isqrt(n))
    xs = np.linspace(bmin[a], bmax[a], side + 2)[1:-1]
    ys = np.linspace(bmin[b], bmax[b], side + 2)[1:-1]
    positions: list[np.ndarray] = []
    for x in xs:
        for y in ys:
            pos = np.zeros(3)
            pos[a] = float(x)
            pos[b] = float(y)
            pos[up_idx] = height
            positions.append(pos)
    return positions


def _positions_from_density(
    renderer: Renderer,
    up_axis: str,
    n: int,
) -> list[np.ndarray] | None:
    """Sample ``n`` positions weighted by sqrt(density) so seeds spread across the scene
    but prefer reconstructed, navigable areas rather than empty holes.

    Returns ``None`` when the renderer has no density grid.
    """
    cloud = getattr(renderer, "cloud", None)
    if cloud is None:
        return None
    grid = getattr(cloud, "density_grid", None)
    dbounds = getattr(cloud, "density_bounds", None)
    if grid is None or dbounds is None:
        return None

    lo, hi = dbounds
    gs = grid.shape[0]
    weighted = np.sqrt(grid / (grid.max() + 1e-9))
    flat = weighted.ravel()
    flat_sum = flat.sum()
    if flat_sum <= 0:
        return None
    probs = flat / flat_sum

    a, b = _FLOOR_AXES[up_axis]

    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(probs), size=n, replace=True, p=probs)
    positions: list[np.ndarray] = []
    for idx in indices:
        row, col = divmod(int(idx), gs)
        xa = float(lo[a] + (row + 0.5) / gs * (hi[a] - lo[a]))
        xb = float(lo[b] + (col + 0.5) / gs * (hi[b] - lo[b]))
        pos = np.zeros(3)
        pos[a] = xa
        pos[b] = xb
        # Height is filled in by generate_seeds using the validated origin height.
        positions.append(pos)
    return positions


def generate_seeds(
    config: RunConfig,
    bounds_min: np.ndarray | None = None,
    bounds_max: np.ndarray | None = None,
    origin: np.ndarray | None = None,
    renderer: Renderer | None = None,
) -> list[Pose]:
    """Generate seed pose candidates spread across the scene floor.

    When a renderer with a density grid is available, seeds are sampled from
    low-density (sparse) regions so walks start where coverage is most needed.
    Otherwise, seeds are placed on a regular floor grid. A set of candidates
    from the known-good ``origin`` (facing different directions) is always
    appended as a fallback.
    """
    a, b = _FLOOR_AXES[config.up_axis]
    up_idx = _UP_INDEX[config.up_axis]
    bmin = np.array(config.bounds_min if bounds_min is None else bounds_min, dtype=np.float64)
    bmax = np.array(config.bounds_max if bounds_max is None else bounds_max, dtype=np.float64)
    safe_origin = (bmin + bmax) / 2.0 if origin is None else origin.copy()
    # Use the render-validated origin height for all seeds so they start at the
    # same elevation as a known-good camera position rather than mid-bounding-box.
    good_height = float(safe_origin[up_idx])

    n_candidates = max(config.num_seeds * 3, 12)
    frontier: list[np.ndarray] | None = None
    if renderer is not None:
        frontier = _positions_from_density(renderer, config.up_axis, n_candidates)
    if frontier is None:
        frontier = _floor_seed_positions(bmin, bmax, config.up_axis, n_candidates, height=good_height)

    # Stamp the validated height onto density-sampled positions (they have height=0).
    for pos in frontier:
        pos[up_idx] = good_height

    candidates: list[Pose] = []
    for pos in frontier:
        angle = 2.0 * math.pi * len(candidates) / max(n_candidates, 1)
        target = pos.copy()
        target[a] += math.cos(angle)
        target[b] += math.sin(angle)
        candidates.append(Pose(position=pos.copy(), rotation=look_at(pos, target, config.up_axis)))

    # Always include origin-facing candidates as a fallback seed pool
    n_origin = max(config.num_seeds, 4)
    for i in range(n_origin):
        angle = 2.0 * math.pi * i / n_origin
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

    depth_estimator = None
    if config.use_depth_estimator:
        from gaussian_robot.depth.estimator import DA3DepthEstimator  # noqa: PLC0415

        depth_estimator = DA3DepthEstimator(
            model_name=config.depth_model,
            device=config.cuda_device,
        )

    builder = ObservationBuilder(
        renderer=renderer,
        up_axis=config.up_axis,
        map_size=config.map_size,
        task=config.task_prompt,
        depth_estimator=depth_estimator,
    )

    walk_policies: list[StopPolicy] = [
        CoveragePlateau(novelty_delta=space.step, window=5),
        BoundsGuard(),
        StuckGuard(step=space.step),
    ]
    session_policies: list[SessionStopPolicy] = [
        PoseBudget(max_poses=config.pose_budget),
        CoverageTarget(radius=coverage_radius, tau=0.8),
        QualityTarget(radius=coverage_radius, tau=0.7),
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

    from gaussian_robot.render.camera import Camera  # noqa: PLC0415

    intrinsics = _PREVIEW_INTRINSICS
    candidates = generate_seeds(config, bmin, bmax, origin=seed_origin, renderer=renderer)
    # Sort candidates by render quality so the best-looking positions are tried first.
    def _seed_score(pose: Pose) -> float:
        r = renderer.render(Camera(pose=pose, intrinsics=intrinsics))
        if r.depth is None:
            return -1.0
        finite = r.depth[np.isfinite(r.depth)]
        med = float(np.median(finite)) if finite.size > 0 else 0.0
        alpha = float(r.alpha.mean()) if r.alpha is not None else 0.5
        return med * alpha

    scored = sorted(candidates, key=_seed_score, reverse=True)

    seeds: list[Pose] = []
    for s in scored:
        if len(seeds) >= config.num_seeds:
            break
        result = renderer.render(Camera(pose=s, intrinsics=intrinsics))
        if result.depth is None:
            continue
        finite_frac = float(np.isfinite(result.depth).mean())
        near = result.depth[np.isfinite(result.depth)]
        median_depth = float(np.median(near)) if near.size > 0 else 0.0
        alpha_mean = float(result.alpha.mean()) if result.alpha is not None else 1.0
        if finite_frac < 0.5 or median_depth < space.step or alpha_mean < 0.15:
            continue
        seeds.append(s)
    if not seeds:
        seeds = [scored[0]]

    coverage = CoverageState.empty(config.up_axis, bmin, bmax)
    return explorer, seeds, coverage
