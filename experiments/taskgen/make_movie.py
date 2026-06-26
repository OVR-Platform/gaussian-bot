"""Render a run_record into an animated GIF: robot RGB view | top-down path map.

Reconstructs each camera from the saved eye position + forward (look_at), renders the splat
view, and draws a top-down panel (trajectory, target + reach radius, current pose/heading,
other objects). Lets a human watch what the agent saw and where it went.

Run:  uv run python experiments/taskgen/make_movie.py [out.gif]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from gaussian_robot.backends.gsplat_renderer import GsplatRenderer
from gaussian_robot.metrics.coverage import floor_xy
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose
from gaussian_robot.session import look_at

HERE = Path(__file__).parent
PLY = "/mnt/archive/datasets/ufficio360-35a39133-e1f2-4426-86d4-a3d7a00614ee-PIC/gaussian_pointcloud_30000_original.ply"
DEVICE = "cuda:1"
S = 480  # panel size
FX = FY = S / 2  # 90 deg fov
INTR = CameraIntrinsics(fx=FX, fy=FY, cx=S / 2, cy=S / 2, width=S, height=S)
EPS_REACH = 1.2
UP = "-y"


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "run.gif"
    run = json.loads((HERE / "run_record.json").read_text())
    tasks = {t["task_id"]: t for t in json.loads((HERE / "tasks.json").read_text())["tasks"]}
    graph = json.loads((HERE / "scene_graph.json").read_text())
    objs = graph["objects"]
    task = tasks[run["task_id"]]
    target = np.asarray(next(o for o in objs if o["id"] == task["target"])["center"])
    tgt_f = floor_xy(target, UP)[0]

    traj = np.asarray(run["trajectory"], dtype=np.float64)
    fwds = np.asarray(run["forwards"], dtype=np.float64)
    traj_f = floor_xy(traj, UP)  # (T,2)
    obj_f = np.array([floor_xy(np.asarray(o["center"]), UP)[0] for o in objs])

    # map extent from everything we draw, with margin
    allpts = np.vstack([traj_f, obj_f, tgt_f[None]])
    lo, hi = allpts.min(0) - 1.0, allpts.max(0) + 1.0
    span = np.maximum(hi - lo, 1e-3)

    def to_px(p: np.ndarray) -> tuple[int, int]:
        u = (p[0] - lo[0]) / span[0] * (S - 20) + 10
        v = (1 - (p[1] - lo[1]) / span[1]) * (S - 20) + 10
        return int(u), int(v)

    print(f"loading splat on {DEVICE} ...", flush=True)
    renderer = GsplatRenderer.from_path(PLY, device=DEVICE)

    frames: list[Image.Image] = []
    for k in range(len(traj)):
        p, f = traj[k], fwds[k]
        pose = Pose(position=p, rotation=look_at(p, p + f, UP))
        rgb = renderer.render(Camera(pose=pose, intrinsics=INTR)).rgb
        left = Image.fromarray(rgb).resize((S, S))

        mp = Image.new("RGB", (S, S), (24, 24, 28))
        d = ImageDraw.Draw(mp)
        for of in obj_f:  # all objects, faint
            x, y = to_px(of)
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(90, 90, 100))
        tx, ty = to_px(tgt_f)  # target + reach radius
        rr = int(EPS_REACH / span[0] * (S - 20))
        d.ellipse([tx - rr, ty - rr, tx + rr, ty + rr], outline=(200, 60, 60))
        d.line([tx - 7, ty - 7, tx + 7, ty + 7], fill=(230, 70, 70), width=2)
        d.line([tx - 7, ty + 7, tx + 7, ty - 7], fill=(230, 70, 70), width=2)
        wps = run.get("waypoints", [])  # planned route (yellow), if any
        if wps:
            pts = [to_px(np.asarray(w)) for w in wps]
            if len(pts) > 1:
                d.line(pts, fill=(220, 200, 60), width=1)
            for px in pts:
                d.ellipse([px[0] - 2, px[1] - 2, px[0] + 2, px[1] + 2], outline=(220, 200, 60))
        if k > 0:  # actual path so far
            d.line([to_px(q) for q in traj_f[: k + 1]], fill=(60, 200, 230), width=2)
        cx, cy = to_px(traj_f[k])  # current pose + heading
        ff = floor_xy(f, UP)[0]
        nf = np.linalg.norm(ff)
        if nf > 1e-6:
            hx, hy = to_px(traj_f[k] + ff / nf * span[0] * 0.06)
            d.line([cx, cy, hx, hy], fill=(80, 240, 120), width=2)
        d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(80, 240, 120))
        dist = float(np.linalg.norm(traj_f[k] - tgt_f))
        d.text((8, 8), f"step {k}/{len(traj) - 1}  dist {dist:.1f}m", fill=(230, 230, 230))

        combo = Image.new("RGB", (2 * S, S), (0, 0, 0))
        combo.paste(left, (0, 0))
        combo.paste(mp, (S, 0))
        frames.append(combo)

    frames[0].save(out, save_all=True, append_images=frames[1:], duration=350, loop=0)
    print(f"wrote {out} ({len(frames)} frames)  task: {task['instruction']}")


if __name__ == "__main__":
    main()
