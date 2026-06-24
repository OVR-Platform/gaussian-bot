"""Build a runnable session from a :class:`RunConfig`.

This is the single wiring point: it selects backends from the config (demo
fakes now; a real Qwen vLLM client when enabled; a real gsplat renderer when it
lands), builds the :class:`Explorer`, and spreads seed poses inside the scene.

Seed lifecycle (the source of truth for "where do walks start?"):

    capture poses / density grid / floor grid
        │  generate_seed_candidates()  -> list[SeedPose] (tagged with `kind`)
        ▼
    candidate SeedPoses (spread + origin-facing fallbacks)
        │  validate_seed_poses()       -> drops void/blurry views, keeps `kind`
        ▼
    seed poses  ──run_session()──▶  one Walk each (walk_id "walk{i}")
                                     into a shared CoverageState

A *seed* is the pose a walk starts from; a *walk* is the episode and is
identified by ``walk_id`` (not "seed_id"). ``SeedPose.kind`` records provenance
("capture" = a real reconstruction camera; otherwise a synthesised guess) so the
UI can show whether a walk started from a real viewpoint or a fallback.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

from gaussian_robot.backends.demo import FakeRenderer, ScriptedDemoVLM
from gaussian_robot.config import RunConfig
from gaussian_robot.filters.pose_filters import FilteredPose, filter_poses
from gaussian_robot.metrics.coverage import CoverageState, PoseSample
from gaussian_robot.nav.action import ActionSpace
from gaussian_robot.nav.explorer import Explorer, SeedPose, WalkResult
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
from gaussian_robot.render.camera import CameraIntrinsics, Pose, axis_index, up_vector
from gaussian_robot.splat.scene import SceneBounds, SplatScene
from gaussian_robot.vlm.client import VLMClient

_FLOOR_PLANE = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

# Fixed RNG seed for reproducible density-weighted candidate sampling. This is a
# numpy PRNG seed — unrelated to the SeedPoses that walks start from.
_RNG_SEED = 42


def _floor_axes(up_axis: str) -> tuple[int, int]:
    """The two world axes spanning the floor plane (orthogonal to up)."""
    return _FLOOR_PLANE[axis_index(up_axis)]


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
    up_idx = axis_index(up_axis)
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


def _load_capture_poses(config: RunConfig) -> list[Pose]:
    """Load the splat's capture poses from config or by discovery (empty on failure).

    Independent of ``use_capture_pose_seeds``: the poses are also used to auto-detect
    the up axis, which we want even when capture-pose *seeding* is disabled.
    """
    from gaussian_robot.splat.capture_poses import (  # noqa: PLC0415
        discover_capture_poses,
        load_capture_poses,
    )

    source: str | Path | None = config.poses_path
    if source is None and config.ply_path:
        source = discover_capture_poses(config.ply_path)
    if source is None:
        return []
    try:
        return load_capture_poses(source)
    except (OSError, ValueError, KeyError, struct.error):
        return []


def _resolve_up_axis(config: RunConfig, capture_poses: list[Pose]) -> str:
    """Resolve ``config.up_axis``, auto-detecting from capture poses when set to 'auto'."""
    if config.up_axis != "auto":
        return config.up_axis
    from gaussian_robot.splat.capture_poses import infer_up_axis  # noqa: PLC0415

    return infer_up_axis(capture_poses) or "y"


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

    up_axis = _resolve_up_axis(config, _load_capture_poses(config))
    cam_pos = _best_origin(renderer, up_axis, bmin, bmax).copy()
    rot = look_at(cam_pos, cam_pos, up_axis)
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

    up_axis = _resolve_up_axis(config, _load_capture_poses(config))
    origin = _best_origin(renderer, up_axis, bmin, bmax)

    pose = Pose(position=origin.copy(), rotation=look_at(origin, origin, up_axis))
    frames: list[str] = []

    for _step_i in range(n_steps):
        next_pose = apply_action(pose, Action.FORWARD, space, up_axis)
        for f in range(n_frames):
            t = f / max(n_frames - 1, 1)
            pos = pose.position * (1 - t) + next_pose.position * t
            cam = Camera(pose=Pose(position=pos, rotation=pose.rotation), intrinsics=intrinsics)
            result = renderer.render(cam)
            frames.append(jpeg_data_url(result.rgb))
        pose = next_pose

    return {"frames": frames, "ok": True, "step_size": space.step, "n_steps": n_steps}


def _rotation_geodesic(r0: np.ndarray, r1: np.ndarray, t: float) -> np.ndarray:
    """Interpolate two world->camera rotations along the SO(3) geodesic (slerp).

    Uses the same world-frame composition as ``apply_action`` (``R = Rw @ R0``):
    the relative world rotation ``r1 @ r0.T`` is reduced to axis-angle and applied
    by a fraction ``t``. Robust for the small per-step rotations a walk makes.
    """
    from gaussian_robot.nav.action import _rotation_about_axis  # noqa: PLC0415

    rel = r1 @ r0.T
    cos = float(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cos)
    if theta < 1e-6:
        return r0.copy()
    axis = np.array(
        [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]], dtype=np.float64
    ) / (2.0 * math.sin(theta))
    interpolated: np.ndarray = _rotation_about_axis(axis, t * theta) @ r0
    return interpolated


def interpolate_walk_poses(poses: list[Pose], per_segment: int) -> list[Pose]:
    """Densify a walk trajectory: ``per_segment`` interpolated poses per checkpoint pair.

    Consecutive identical checkpoints (a blocked/stop step that didn't move) add no
    frames. Position is linearly interpolated; orientation slerped.
    """
    if len(poses) < 2 or per_segment < 1:
        return list(poses)
    out: list[Pose] = []
    for a, b in zip(poses[:-1], poses[1:], strict=False):
        same = bool(np.allclose(a.position, b.position) and np.allclose(a.rotation, b.rotation))
        steps = 1 if same else per_segment
        for k in range(steps):
            t = k / per_segment
            pos = a.position * (1 - t) + b.position * t
            rot = a.rotation if same else _rotation_geodesic(a.rotation, b.rotation, t)
            out.append(Pose(position=pos, rotation=rot))
    out.append(poses[-1])
    return out


def render_walk_movie(
    config: RunConfig,
    poses: list[Pose],
    *,
    per_segment: int = 8,
    max_frames: int = 300,
) -> dict[str, object]:
    """Render a smooth fly-through of a walk by interpolating between its poses.

    The walk's checkpoint poses are densified with up to ``per_segment`` frames per
    segment (reduced so the total never exceeds ``max_frames``), then each is
    rendered to an RGB frame. Returns JPEG data URLs ready for the dashboard player.
    """
    from gaussian_robot.render.camera import Camera  # noqa: PLC0415
    from gaussian_robot.vlm.qwen import jpeg_data_url  # noqa: PLC0415

    if not config.ply_path:
        raise ValueError("no ply_path configured")
    if not poses:
        return {"ok": False, "error": "no frames for this walk yet", "frames": []}
    renderer, _, _ = _load_renderer(config.ply_path, device=config.cuda_device)

    segments = max(1, len(poses) - 1)
    per = max(1, min(per_segment, max(1, max_frames // segments)))
    frame_poses = interpolate_walk_poses(poses, per)
    if len(frame_poses) > max_frames:  # long walk: subsample evenly to the cap
        idx = np.linspace(0, len(frame_poses) - 1, max_frames).astype(int)
        frame_poses = [frame_poses[i] for i in idx]

    frames = [
        jpeg_data_url(renderer.render(Camera(pose=p, intrinsics=_PREVIEW_INTRINSICS)).rgb)
        for p in frame_poses
    ]
    return {"ok": True, "frames": frames, "n_frames": len(frames), "checkpoints": len(poses)}


def look_at(origin: np.ndarray, target: np.ndarray, up_axis: str) -> np.ndarray:
    """World->camera rotation (OpenCV) whose camera sits at ``origin`` looking at ``target``."""
    up = up_vector(up_axis)
    forward = np.asarray(target, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    forward -= up * float(forward @ up)
    fn = float(np.linalg.norm(forward))
    if fn < 1e-9:
        forward = np.zeros(3)
        forward[_floor_axes(up_axis)[1]] = 1.0
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
    a, b = _floor_axes(up_axis)
    up_idx = axis_index(up_axis)
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

    a, b = _floor_axes(up_axis)

    rng = np.random.default_rng(_RNG_SEED)
    indices = rng.choice(len(probs), size=n, replace=True, p=probs)
    positions: list[np.ndarray] = []
    for idx in indices:
        row, col = divmod(int(idx), gs)
        xa = float(lo[a] + (row + 0.5) / gs * (hi[a] - lo[a]))
        xb = float(lo[b] + (col + 0.5) / gs * (hi[b] - lo[b]))
        pos = np.zeros(3)
        pos[a] = xa
        pos[b] = xb
        # Height is filled in by generate_seed_candidates using the validated origin height.
        positions.append(pos)
    return positions


def _spread_capture_poses(poses: list[Pose], up_axis: str, n: int) -> list[Pose]:
    """Pick up to ``n`` capture poses spread across the floor via farthest-point.

    Capture poses cluster (rig bursts, slow walks), so a naive prefix would seed
    many near-identical viewpoints. Farthest-point selection on floor-plane
    position keeps the chosen seeds spatially diverse while every one remains a
    real, known-good viewpoint.
    """
    from gaussian_robot.filters.pose_filters import farthest_point_select  # noqa: PLC0415
    from gaussian_robot.metrics.coverage import floor_xy  # noqa: PLC0415

    if not poses:
        return []
    points = np.array([floor_xy(p.position, up_axis)[0] for p in poses], dtype=np.float64)
    # r_keep=0 so selection only stops at the budget: we want exactly the most
    # spread-out ``n`` poses regardless of absolute spacing.
    idx = farthest_point_select(points, r_keep=0.0, budget=min(n, len(poses)))
    return [poses[i] for i in idx]


def _origin_fallback_seeds(
    safe_origin: np.ndarray, a: int, b: int, ua: str, n: int
) -> list[SeedPose]:
    """Synthesised seeds at the validated origin facing out in ``n`` directions."""
    out: list[SeedPose] = []
    for i in range(n):
        angle = 2.0 * math.pi * i / n
        target = safe_origin.copy()
        target[a] += math.cos(angle)
        target[b] += math.sin(angle)
        out.append(
            SeedPose(
                pose=Pose(position=safe_origin.copy(), rotation=look_at(safe_origin, target, ua)),
                kind="origin_fallback",
            )
        )
    return out


def generate_seed_candidates(
    config: RunConfig,
    bounds_min: np.ndarray | None = None,
    bounds_max: np.ndarray | None = None,
    origin: np.ndarray | None = None,
    renderer: Renderer | None = None,
    capture_poses: list[Pose] | None = None,
    up_axis: str | None = None,
) -> list[SeedPose]:
    """Generate candidate :class:`SeedPose`\\ s spread across the scene floor.

    Each candidate is tagged with its provenance (``SeedPose.kind``):

    - ``capture`` — when ``capture_poses`` (the cameras the splat was
      reconstructed from) are available, they are the candidate pool: every one
      is a known-good viewpoint, spread via farthest-point selection, keeping its
      original orientation.
    - ``density`` / ``grid`` — otherwise candidates are sampled from the
      reconstructed-density grid, or failing that placed on a regular floor grid.
    - ``origin_fallback`` — a ring of origin-facing candidates is always appended
      so validation still has options if the preferred candidates render poorly.
    """
    ua = up_axis if up_axis is not None else config.up_axis
    if ua == "auto":  # defensive: callers normally pass an already-resolved axis
        ua = "y"
    a, b = _floor_axes(ua)
    up_idx = axis_index(ua)
    bmin = np.array(config.bounds_min if bounds_min is None else bounds_min, dtype=np.float64)
    bmax = np.array(config.bounds_max if bounds_max is None else bounds_max, dtype=np.float64)
    safe_origin = (bmin + bmax) / 2.0 if origin is None else origin.copy()
    # Use the render-validated origin height for all seeds so they start at the
    # same elevation as a known-good camera position rather than mid-bounding-box.
    good_height = float(safe_origin[up_idx])

    # Over-sample candidates: after the sharpness floor drops the blurriest ~half,
    # enough well-spread real views remain to pick num_seeds sharp ones from.
    n_candidates = max(config.num_seeds * 6, 24)
    n_origin = max(config.num_seeds, 4)

    if capture_poses:
        candidates = [
            SeedPose(pose=p, kind="capture")
            for p in _spread_capture_poses(capture_poses, ua, n_candidates)
        ]
        candidates += _origin_fallback_seeds(safe_origin, a, b, ua, n_origin)
        return candidates

    frontier_kind = "density"
    frontier: list[np.ndarray] | None = None
    if renderer is not None:
        frontier = _positions_from_density(renderer, ua, n_candidates)
    if frontier is None:
        frontier = _floor_seed_positions(bmin, bmax, ua, n_candidates, height=good_height)
        frontier_kind = "grid"

    # Stamp the validated height onto density-sampled positions (they have height=0).
    for pos in frontier:
        pos[up_idx] = good_height

    candidates = []
    for j, pos in enumerate(frontier):
        angle = 2.0 * math.pi * j / max(n_candidates, 1)
        target = pos.copy()
        target[a] += math.cos(angle)
        target[b] += math.sin(angle)
        candidates.append(
            SeedPose(
                pose=Pose(position=pos.copy(), rotation=look_at(pos, target, ua)),
                kind=frontier_kind,
            )
        )

    candidates += _origin_fallback_seeds(safe_origin, a, b, ua, n_origin)
    return candidates


def _sharpness(rgb: np.ndarray) -> float:
    """High-frequency content of a render (mean squared image gradient).

    Crisp, well-reconstructed views score high; blurry/smeared regions where the
    splat lacks training views score low. Cheap, dependency-free, and scale-free
    enough to compare candidate seed views of the same scene.
    """
    g = rgb.astype(np.float64).mean(axis=2)
    gx = np.diff(g, axis=1)
    gy = np.diff(g, axis=0)
    return float((gx * gx).mean() + (gy * gy).mean())


def validate_seed_poses(
    renderer: Renderer,
    candidates: list[SeedPose],
    *,
    num_seeds: int,
    step: float,
    strict: bool,
) -> list[SeedPose]:
    """Keep the best valid ``num_seeds`` candidates (preserving their ``kind``).

    A candidate is rejected when it looks into the void (mostly-infinite depth or
    near-zero alpha). When ``strict`` (guessed density/grid seeds), candidates are
    re-ranked by render quality (median depth x alpha) so the best guesses are
    tried first, and an extra median-depth floor rejects views buried in geometry.

    When not ``strict`` (capture-pose seeds), the incoming farthest-point order is
    preserved for spatial diversity, but views whose sharpness is below the median
    of the candidate pool are skipped — so a real but blurry/under-reconstructed
    capture view doesn't become a seed while sharper, equally-spread ones exist.
    """
    from gaussian_robot.render.camera import Camera  # noqa: PLC0415

    intrinsics = _PREVIEW_INTRINSICS

    # Render every candidate once and cache its metrics (avoids double-rendering).
    scored: list[tuple[SeedPose, float, float, float, float]] = []
    for s in candidates:
        r = renderer.render(Camera(pose=s.pose, intrinsics=intrinsics))
        if r.depth is None:
            scored.append((s, 0.0, 0.0, 0.0, 0.0))
            continue
        finite_mask = np.isfinite(r.depth)
        finite_frac = float(finite_mask.mean())
        near = r.depth[finite_mask]
        median_depth = float(np.median(near)) if near.size > 0 else 0.0
        alpha_mean = float(r.alpha.mean()) if r.alpha is not None else 1.0
        sharp = _sharpness(r.rgb)
        scored.append((s, finite_frac, median_depth, alpha_mean, sharp))

    if strict:
        ordered = sorted(scored, key=lambda it: it[2] * it[3], reverse=True)
        sharp_floor = 0.0
    else:
        ordered = scored  # preserve farthest-point spread
        valid_sharps = [sh for (_, ff, _, a, sh) in scored if ff >= 0.5 and a >= 0.15]
        sharp_floor = float(np.median(valid_sharps)) if valid_sharps else 0.0

    seeds: list[SeedPose] = []
    for s, finite_frac, median_depth, alpha_mean, sharp in ordered:
        if len(seeds) >= num_seeds:
            break
        if finite_frac < 0.5 or alpha_mean < 0.15:
            continue
        if strict and median_depth < step:
            continue
        if not strict and sharp < sharp_floor:
            continue  # skip the blurriest real views; a sharper spread one remains
        seeds.append(s)
    return seeds or [ordered[0][0]]


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


def assemble_deliverable(
    results: list[WalkResult],
    *,
    up_axis: str,
    r_keep: float,
    budget: int,
    min_confidence: float = 0.0,
) -> list[FilteredPose]:
    """Build the deliverable pose set (ADR-0008), preferring VLM-marked viewpoints.

    The deliverable is the set of *new* camera poses to densify the splat. When the
    VLM has marked fill-in viewpoints (``WalkResult.marks``), those are the primary
    source — the agent has explicitly proposed them. Only when no poses were marked
    do we fall back to the union of all visited trajectory poses. Either way the
    chosen poses pass the standard quality/novelty/budget filter.
    """
    marks = [PoseSample(pose=p, walk_id=r.walk_id) for r in results for p in r.marks]
    samples = marks or [
        PoseSample(pose=s.pose, walk_id=r.walk_id) for r in results for s in r.steps
    ]
    return filter_poses(
        samples, up_axis=up_axis, r_keep=r_keep, budget=budget, min_confidence=min_confidence
    )


def build_session(config: RunConfig) -> tuple[Explorer, list[SeedPose], CoverageState]:
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

    capture_poses = _load_capture_poses(config)
    up_axis = _resolve_up_axis(config, capture_poses)

    scene = SplatScene(
        path=Path(config.ply_path) if config.ply_path else Path("data/scene.ply"),
        bounds=SceneBounds(min=bmin, max=bmax),
        up_axis=up_axis,
    )
    vlm = build_vlm(config)

    depth_estimator = None
    if config.use_depth_estimator:
        from gaussian_robot.depth.estimator import DA3DepthEstimator  # noqa: PLC0415

        depth_estimator = DA3DepthEstimator(
            model_name=config.depth_model,
            device=config.cuda_device,
        )

    # Default the map to a local window (~10 steps across) so motion is visible;
    # the whole-scene view made every step an invisible sub-pixel nudge.
    map_span = config.map_span if config.map_span is not None else space.step * 10.0
    builder = ObservationBuilder(
        renderer=renderer,
        up_axis=up_axis,
        map_size=config.map_size,
        map_span=map_span,
        task=config.task_prompt,
        depth_estimator=depth_estimator,
    )

    walk_policies: list[StopPolicy] = [
        # window=8: tolerate a longer unproductive stretch before ending a walk,
        # so deliberate fine-grained exploration isn't cut short prematurely.
        CoveragePlateau(novelty_delta=space.step, window=8),
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
        mark_target=config.pose_target,
    )
    seed_origin = _best_origin(renderer, up_axis, bmin, bmax)

    seed_pool = capture_poses if config.use_capture_pose_seeds else []
    candidates = generate_seed_candidates(
        config,
        bmin,
        bmax,
        origin=seed_origin,
        renderer=renderer,
        capture_poses=seed_pool,
        up_axis=up_axis,
    )
    # Capture poses are known-good viewpoints, so validation can be lenient (see
    # validate_seed_poses); guessed density/grid candidates need the strict checks.
    seeds = validate_seed_poses(
        renderer, candidates, num_seeds=config.num_seeds, step=space.step, strict=not seed_pool
    )

    coverage = CoverageState.empty(up_axis, bmin, bmax)
    return explorer, seeds, coverage
