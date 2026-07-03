"""Headless episode recording + replay movie for goal-conditioned navigation (ADR-0012).

Two pieces the ``navigate`` CLI (and any headless runner) hangs off the existing event
stream instead of inventing new hooks:

- :class:`EpisodeRecorder` — an :data:`~gaussian_robot.events.EventSink` that captures the
  trajectory (positions + viewing directions), executed actions and grab/drop locations
  from :class:`~gaussian_robot.events.StepEvent` / :class:`~gaussian_robot.events.CarryEvent`.
- :func:`render_episode_gif` — replays a recorded trajectory through the splat and writes a
  ``[robot view | top-down trail+goal]`` animated GIF. This generalises
  ``experiments/taskgen/make_movie.py`` (which is welded to one scene) into the package;
  the renderer is injected per ADR-0001 (pass any :class:`~gaussian_robot.render.base.Renderer`)
  or loaded lazily from ``ply_path``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gaussian_robot.events import CarryEvent, SessionEvent, StepEvent
from gaussian_robot.metrics.coverage import floor_xy, viewing_direction

if TYPE_CHECKING:
    from PIL import Image

    from gaussian_robot.render.base import Renderer


@dataclass
class EpisodeRecord:
    """Everything a navigate episode produced, JSON-serialisable."""

    instruction: str
    up_axis: str
    trajectory: list[list[float]] = field(default_factory=list)  # (T, 3) eye positions
    forwards: list[list[float]] = field(default_factory=list)  # (T, 3) viewing directions
    actions: list[str] = field(default_factory=list)  # executed verb per step
    grabs: list[list[float]] = field(default_factory=list)  # floor positions of grabs
    drops: list[list[float]] = field(default_factory=list)
    target: list[float] | None = None  # (3,) goal position, when one was given
    goal_eps: float | None = None
    stop_reason: str = ""
    success: bool = False
    success_source: str = ""  # "geometric" | "vlm_declared" | ""
    steps: int = 0

    def write_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(self), indent=1))
        return out


class EpisodeRecorder:
    """Event sink capturing a walk into an :class:`EpisodeRecord`."""

    def __init__(self, instruction: str, up_axis: str) -> None:
        self.record = EpisodeRecord(instruction=instruction, up_axis=up_axis)

    def __call__(self, event: SessionEvent) -> None:
        if isinstance(event, StepEvent):
            self.record.trajectory.append([float(x) for x in event.pose.position])
            self.record.forwards.append([float(x) for x in viewing_direction(event.pose.rotation)])
            self.record.actions.append(event.action.value)
        elif isinstance(event, CarryEvent):
            dest = self.record.grabs if event.kind == "grab" else self.record.drops
            dest.append([float(x) for x in event.floor])


def finalize_outcome(record: EpisodeRecord, stop_reason: str) -> EpisodeRecord:
    """Stamp ``stop_reason`` → (success, provenance) per ADR-0012.

    ``goal_reached`` is a *measured* success. ``task_complete`` (the VLM's own stop) counts
    as success ONLY when no geometric target was available — and is labelled
    ``vlm_declared`` so downstream consumers know it is unverified. With a target present,
    a VLM stop outside ``goal_eps`` is a failure (the geometric policy would have fired
    first on a true arrival).
    """
    record.stop_reason = stop_reason
    record.steps = len(record.trajectory)
    if stop_reason == "goal_reached":
        record.success = True
        record.success_source = "geometric"
    elif stop_reason == "task_complete":
        if record.target is None:
            record.success = True
            record.success_source = "vlm_declared"
        else:
            record.success = False
            record.success_source = "geometric"
    else:
        record.success = False
        record.success_source = "geometric" if record.target is not None else "vlm_declared"
    return record


def _draw_map_panel(
    record: EpisodeRecord,
    k: int,
    *,
    traj_f: np.ndarray,
    fwds: np.ndarray,
    tgt_f: np.ndarray | None,
    lo: np.ndarray,
    span: np.ndarray,
    s: int,
) -> Image.Image:
    """One top-down map frame: goal (+reach radius), trail, grab/drop marks, pose/heading."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    def to_px(p: np.ndarray) -> tuple[int, int]:
        u = (p[0] - lo[0]) / span[0] * (s - 20) + 10
        v = (1 - (p[1] - lo[1]) / span[1]) * (s - 20) + 10
        return int(u), int(v)

    mp = Image.new("RGB", (s, s), (24, 24, 28))
    d = ImageDraw.Draw(mp)
    if tgt_f is not None:
        tx, ty = to_px(tgt_f)
        if record.goal_eps:
            rr = max(2, int(record.goal_eps / span[0] * (s - 20)))
            d.ellipse([tx - rr, ty - rr, tx + rr, ty + rr], outline=(200, 60, 60))
        d.line([tx - 7, ty - 7, tx + 7, ty + 7], fill=(230, 70, 70), width=2)
        d.line([tx - 7, ty + 7, tx + 7, ty - 7], fill=(230, 70, 70), width=2)
    if k > 0:
        d.line([to_px(q) for q in traj_f[: k + 1]], fill=(60, 200, 230), width=2)
    for pts, colour in ((record.grabs, (80, 240, 120)), (record.drops, (230, 150, 50))):
        for gx in pts:
            px = to_px(np.asarray(gx))
            d.line([px[0] - 6, px[1], px[0] + 6, px[1]], fill=colour, width=3)
            d.line([px[0], px[1] - 6, px[0], px[1] + 6], fill=colour, width=3)
    cx, cy = to_px(traj_f[k])
    ff = floor_xy(fwds[k], record.up_axis)[0]
    nf = float(np.linalg.norm(ff))
    if nf > 1e-6:
        hx, hy = to_px(traj_f[k] + ff / nf * span[0] * 0.06)
        d.line([cx, cy, hx, hy], fill=(80, 240, 120), width=2)
    d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(80, 240, 120))
    label = f"step {k}/{traj_f.shape[0] - 1}"
    if tgt_f is not None:
        label += f"  dist {float(np.linalg.norm(traj_f[k] - tgt_f)):.1f}m"
    d.text((8, 8), label, fill=(230, 230, 230))
    return mp


def render_episode_gif(
    record: EpisodeRecord,
    out_gif: str | Path,
    *,
    ply_path: str | Path | None = None,
    renderer: Renderer | None = None,
    device: str = "cuda:0",
    panel: int = 480,
    duration_ms: int = 350,
    max_frames: int = 240,
) -> dict[str, object]:
    """Replay the trajectory through the splat → ``[robot view | top-down map]`` GIF.

    Pass either a live ``renderer`` (reused, e.g. the session's) or ``ply_path`` to load
    one lazily. The map draws the goal (+ reach radius when ``goal_eps`` is set), the
    trail so far, grab/drop marks, and the current pose/heading.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose  # noqa: PLC0415
    from gaussian_robot.session import look_at  # noqa: PLC0415

    traj = np.asarray(record.trajectory, dtype=np.float64)
    fwds = np.asarray(record.forwards, dtype=np.float64)
    if traj.shape[0] < 2:
        return {"ok": False, "error": "trajectory too short for a movie", "n_frames": 0}
    if traj.shape[0] > max_frames:
        idx = np.linspace(0, traj.shape[0] - 1, max_frames).astype(int)
        traj, fwds = traj[idx], fwds[idx]

    if renderer is None:
        if ply_path is None:
            raise ValueError("render_episode_gif needs a renderer or a ply_path")
        from gaussian_robot.backends.gsplat_renderer import GsplatRenderer  # noqa: PLC0415

        renderer = GsplatRenderer.from_path(str(ply_path), device=device)

    s = panel
    intr = CameraIntrinsics(fx=s / 2.0, fy=s / 2.0, cx=s / 2.0, cy=s / 2.0, width=s, height=s)
    ua = record.up_axis

    traj_f = floor_xy(traj, ua)
    tgt_f = floor_xy(np.asarray(record.target), ua)[0] if record.target is not None else None
    allpts = np.vstack([traj_f] + ([tgt_f[None]] if tgt_f is not None else []))
    lo, hi = allpts.min(0) - 1.0, allpts.max(0) + 1.0
    span = np.maximum(hi - lo, 1e-3)

    frames: list[Image.Image] = []
    for k in range(traj.shape[0]):
        pose = Pose(position=traj[k], rotation=look_at(traj[k], traj[k] + fwds[k], ua))
        rgb = renderer.render(Camera(pose=pose, intrinsics=intr)).rgb
        left = Image.fromarray(rgb).resize((s, s))
        mp = _draw_map_panel(
            record, k, traj_f=traj_f, fwds=fwds, tgt_f=tgt_f, lo=lo, span=span, s=s
        )
        combo = Image.new("RGB", (2 * s, s), (0, 0, 0))
        combo.paste(left, (0, 0))
        combo.paste(mp, (s, 0))
        cap = ImageDraw.Draw(combo)
        cap.text((8, s - 18), record.instruction[:70], fill=(255, 220, 120))
        frames.append(combo)

    out = Path(out_gif)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return {"ok": True, "out_gif": str(out), "n_frames": len(frames)}
