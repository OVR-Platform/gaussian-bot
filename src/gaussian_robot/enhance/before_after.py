"""Before/after fly-through: the SAME robot path rendered through two clouds, side by side.

Given the trajectory the navigator walked (``ExploreFillReport.trajectory``), this renders each
pose from the ORIGINAL cloud and from the ENHANCED cloud and lays them next to a top-down map of
the path — so a human can see exactly what the gap-fill changed along the route the robot took.

Rendering is two-pass (all "before" frames, free the cloud, then all "after" frames) so only one
cloud is resident at a time — the peak VRAM is one splat, not two. Frames are written to an
animated GIF with Pillow (no extra dependency), matching ``experiments/taskgen/make_movie.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gaussian_robot.metrics.coverage import floor_xy
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose
from gaussian_robot.session import interpolate_walk_poses


def plan_movie_poses(
    trajectory: list[Pose], *, per_segment: int = 4, max_frames: int = 240
) -> list[Pose]:
    """Densify ``trajectory`` (slerped interpolation) and cap to ``max_frames`` evenly.

    Pure (no GPU): factored out so the frame plan can be unit-tested without a renderer.
    """
    poses = interpolate_walk_poses(trajectory, per_segment) if len(trajectory) >= 2 else list(trajectory)
    if len(poses) > max_frames:
        idx = np.linspace(0, len(poses) - 1, max_frames).astype(int)
        poses = [poses[i] for i in idx]
    return poses


def map_extent(points: np.ndarray, *, margin: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, span)`` floor-plane bounds covering ``points`` (N, 2) with a margin.

    Pure helper for the top-down panel; ``span`` is clamped away from zero.
    """
    if points.shape[0] == 0:
        return np.zeros(2), np.ones(2)
    lo = points.min(0) - margin
    hi = points.max(0) + margin
    span = np.maximum(hi - lo, 1e-3)
    return lo, span


def _render_pass(ply_path: str | Path, poses: list[Pose], intr: CameraIntrinsics, device: str) -> list[np.ndarray]:
    """Render every pose from one cloud, return RGB arrays, then release the cloud + CUDA cache."""
    from gaussian_robot.backends.gsplat_renderer import GsplatRenderer  # noqa: PLC0415

    renderer = GsplatRenderer.from_path(str(ply_path), device=device)
    frames = [renderer.render(Camera(pose=p, intrinsics=intr)).rgb for p in poses]
    del renderer
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return frames


def render_before_after_gif(
    trajectory: list[Pose],
    orig_ply: str | Path,
    enhanced_ply: str | Path,
    out_gif: str | Path,
    *,
    device: str = "cuda:0",
    up_axis: str = "y",
    marks: list[Pose] | None = None,
    panel: int = 480,
    per_segment: int = 4,
    max_frames: int = 240,
    duration_ms: int = 120,
) -> dict[str, object]:
    """Render the before/after path movie to ``out_gif``; returns a small summary dict.

    Each frame is ``[BEFORE | AFTER | top-down map]``. The map draws the marked "poses to improve"
    (red), the path walked so far (cyan) and the current pose + heading (green).
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    poses = plan_movie_poses(trajectory, per_segment=per_segment, max_frames=max_frames)
    if len(poses) < 2:
        return {"ok": False, "error": "trajectory too short for a movie", "n_frames": len(poses)}

    s = panel
    intr = CameraIntrinsics(fx=s / 2.0, fy=s / 2.0, cx=s / 2.0, cy=s / 2.0, width=s, height=s)

    # Two-pass render: one cloud resident at a time (peak VRAM = a single splat).
    before = _render_pass(orig_ply, poses, intr, device)
    after = _render_pass(enhanced_ply, poses, intr, device)

    path_f = floor_xy(np.array([p.position for p in poses], dtype=np.float64), up_axis)
    mark_f = (
        floor_xy(np.array([m.position for m in marks], dtype=np.float64), up_axis)
        if marks
        else np.empty((0, 2), dtype=np.float64)
    )
    lo, span = map_extent(np.vstack([path_f, mark_f]) if mark_f.shape[0] else path_f)

    def to_px(p: np.ndarray) -> tuple[int, int]:
        u = (p[0] - lo[0]) / span[0] * (s - 20) + 10
        v = (1 - (p[1] - lo[1]) / span[1]) * (s - 20) + 10
        return int(u), int(v)

    frames: list[Image.Image] = []
    for k in range(len(poses)):
        left = Image.fromarray(before[k]).resize((s, s))
        right = Image.fromarray(after[k]).resize((s, s))

        mp = Image.new("RGB", (s, s), (24, 24, 28))
        d = ImageDraw.Draw(mp)
        for mf in mark_f:  # marked poses-to-improve (the gaps the robot flagged)
            mx, my = to_px(mf)
            d.ellipse([mx - 3, my - 3, mx + 3, my + 3], outline=(220, 80, 80))
        if k > 0:  # path walked so far
            d.line([to_px(q) for q in path_f[: k + 1]], fill=(60, 200, 230), width=2)
        cx, cy = to_px(path_f[k])  # current pose + heading
        fwd = poses[k].rotation[2, :]
        ff = floor_xy(fwd, up_axis)[0]
        nf = float(np.linalg.norm(ff))
        if nf > 1e-6:
            hx, hy = to_px(path_f[k] + ff / nf * span[0] * 0.06)
            d.line([cx, cy, hx, hy], fill=(80, 240, 120), width=2)
        d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(80, 240, 120))
        d.text((8, 8), f"frame {k}/{len(poses) - 1}", fill=(230, 230, 230))

        combo = Image.new("RGB", (3 * s, s), (0, 0, 0))
        combo.paste(left, (0, 0))
        combo.paste(right, (s, 0))
        combo.paste(mp, (2 * s, 0))
        cap = ImageDraw.Draw(combo)
        cap.text((8, s - 18), "BEFORE", fill=(255, 220, 120))
        cap.text((s + 8, s - 18), "AFTER", fill=(120, 255, 180))
        frames.append(combo)

    out = Path(out_gif)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return {"ok": True, "out_gif": str(out), "n_frames": len(frames), "n_marks": int(mark_f.shape[0])}
