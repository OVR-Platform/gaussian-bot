"""Montage reel of the successful demos: a grid of animated top-down mini-maps (no GPU).

Reads demos/*.json, lays out every SUCCESSFUL episode in a grid, and animates each trajectory
(path, target, grab/drop, robot) so you can see the whole dataset at a glance.

Run:  uv run python experiments/taskgen/make_reel.py [out.gif]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from gaussian_robot.metrics.coverage import floor_xy

HERE = Path(__file__).parent
DEMOS = HERE / "demos"
CELL = 240
PAD = 12
UP = "-y"
FRAMES = 48  # subsample each trajectory to this many frames


def _f(p) -> np.ndarray:
    return floor_xy(np.asarray(p, dtype=np.float64), UP)[0]


def _draw_cell(demo: dict, frame_t: float) -> Image.Image:
    run = demo["run"]
    traj = np.array(run["trajectory"], dtype=np.float64)
    tf = _f(demo["target_center"])
    pts = [_f(p) for p in run["trajectory"]] + [tf]
    if demo.get("destination_center") is not None:
        pts.append(_f(demo["destination_center"]))
    pts += [np.asarray(w) for w in run.get("waypoints", [])]
    arr = np.array(pts)
    lo, hi = arr.min(0) - 0.5, arr.max(0) + 0.5
    span = np.maximum(hi - lo, 1e-3)

    def tx(p):
        u = PAD + (p[0] - lo[0]) / span[0] * (CELL - 2 * PAD)
        v = PAD + (1 - (p[1] - lo[1]) / span[1]) * (CELL - 2 * PAD)
        return int(u), int(v)

    im = Image.new("RGB", (CELL, CELL), (22, 22, 26))
    d = ImageDraw.Draw(im)
    for w in run.get("waypoints", []):  # planned path (faint yellow)
        px = tx(np.asarray(w))
        d.ellipse([px[0] - 2, px[1] - 2, px[0] + 2, px[1] + 2], outline=(150, 140, 60))
    tp = tx(tf)  # target
    d.line([tp[0] - 6, tp[1] - 6, tp[0] + 6, tp[1] + 6], fill=(230, 70, 70), width=2)
    d.line([tp[0] - 6, tp[1] + 6, tp[0] + 6, tp[1] - 6], fill=(230, 70, 70), width=2)
    k = max(1, int(frame_t * (len(traj) - 1)))
    floor_pts = [tx(_f(p)) for p in run["trajectory"][: k + 1]]
    if len(floor_pts) > 1:
        d.line(floor_pts, fill=(60, 200, 230), width=2)
    for g in run.get("grabs", []):
        px = tx(np.asarray(g)); d.line([px[0]-5,px[1],px[0]+5,px[1]],fill=(90,240,120),width=3)
        d.line([px[0], px[1]-5, px[0], px[1]+5], fill=(90, 240, 120), width=3)
    for dr in run.get("drops", []):
        px = tx(np.asarray(dr)); d.line([px[0]-5,px[1],px[0]+5,px[1]],fill=(230,150,50),width=3)
        d.line([px[0], px[1]-5, px[0], px[1]+5], fill=(230, 150, 50), width=3)
    cur = floor_pts[-1] if floor_pts else tp
    d.ellipse([cur[0] - 4, cur[1] - 4, cur[0] + 4, cur[1] + 4], fill=(90, 240, 120))
    label = demo["instruction"][:42]
    d.text((6, 4), f"{demo['type']}", fill=(180, 180, 190))
    d.text((6, CELL - 14), label, fill=(210, 210, 215))
    return im


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "reel.gif"
    man = json.loads((DEMOS / "manifest.json").read_text())["demos"]
    # exclude tasks flagged invalid (e.g. target is a mislabelled surface, not an object)
    invalid = {t["task_id"] for t in json.loads((HERE / "tasks.json").read_text())["tasks"]
               if t.get("valid") is False}
    demos = [json.loads((DEMOS / f"{m['task_id'].replace('/', '_')}.json").read_text())
             for m in man if m["success"] and m["task_id"] not in invalid]
    if invalid:
        print(f"excluding invalid tasks (mislabelled targets): {sorted(invalid)}")
    if not demos:
        print("no successful demos yet")
        return
    n = len(demos)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    W, H = cols * CELL, rows * CELL
    frames = []
    for fi in range(FRAMES):
        t = fi / (FRAMES - 1)
        sheet = Image.new("RGB", (W, H), (0, 0, 0))
        for j, demo in enumerate(demos):
            cell = _draw_cell(demo, t)
            sheet.paste(cell, ((j % cols) * CELL, (j // cols) * CELL))
        frames.append(sheet)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=120, loop=0)
    print(f"wrote {out}: {n} successful demos, {cols}x{rows} grid")


if __name__ == "__main__":
    main()
