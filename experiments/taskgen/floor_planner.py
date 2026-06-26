"""Floor occupancy + A* path planning so navigation routes AROUND obstacles (e.g. a column).

Greedy bearing-following gets trapped when an obstacle sits on the straight line to the target
(forward blocked -> back up -> re-aim -> forward into it again). Here we build a body-height
occupancy grid from the gaussians (cells with geometry between ~knee and ~head height are
obstacles; bare floor is free), inflate by the robot radius, and A* a path; the steering cue
then points at the next waypoint instead of the target, so the agent walks around.
"""

from __future__ import annotations

import heapq

import numpy as np

from gaussian_robot.metrics.coverage import floor_xy
from gaussian_robot.render.camera import up_vector


def build_occupancy(
    means: np.ndarray, up_axis: str, *, grid: int = 72, band=(0.4, 2.2), occ_min: int = 6,
    inflate: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Body-height occupancy over the floor. Returns (occ[grid,grid] bool, lo[2], hi[2])."""
    up = up_vector(up_axis)
    h = means @ up
    f = floor_xy(means, up_axis)  # (N,2)
    lo, hi = f.min(0), f.max(0)
    span = np.maximum(hi - lo, 1e-3)
    ground = float(np.percentile(h, 3))
    band_m = (h >= ground + band[0]) & (h <= ground + band[1])  # geometry at body height
    fb = f[band_m]
    ix = np.clip(((fb[:, 0] - lo[0]) / span[0] * grid).astype(int), 0, grid - 1)
    iz = np.clip(((fb[:, 1] - lo[1]) / span[1] * grid).astype(int), 0, grid - 1)
    cnt = np.zeros((grid, grid), dtype=np.int64)
    np.add.at(cnt, (ix, iz), 1)
    occ = cnt >= occ_min
    for _ in range(inflate):  # dilate by robot radius
        o = occ.copy()
        o[:-1] |= occ[1:]; o[1:] |= occ[:-1]
        o[:, :-1] |= occ[:, 1:]; o[:, 1:] |= occ[:, :-1]
        occ = o
    return occ, lo, hi


def _cell(p: np.ndarray, lo: np.ndarray, hi: np.ndarray, g: int) -> tuple[int, int]:
    span = np.maximum(hi - lo, 1e-3)
    return (int(np.clip((p[0] - lo[0]) / span[0] * g, 0, g - 1)),
            int(np.clip((p[1] - lo[1]) / span[1] * g, 0, g - 1)))


def _nearest_free(occ: np.ndarray, c: tuple[int, int]) -> tuple[int, int]:
    if not occ[c]:
        return c
    g = occ.shape[0]
    for r in range(1, g):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                i, j = c[0] + di, c[1] + dj
                if 0 <= i < g and 0 <= j < g and not occ[i, j]:
                    return (i, j)
    return c


def plan(means: np.ndarray, up_axis: str, start_xz: np.ndarray, goal_xz: np.ndarray,
         **kw) -> list[np.ndarray]:
    """A* floor path from start to goal. Returns world (x,z) waypoints (empty if none)."""
    occ, lo, hi = build_occupancy(means, up_axis, **kw)
    g = occ.shape[0]
    span = np.maximum(hi - lo, 1e-3)
    start = _nearest_free(occ, _cell(start_xz, lo, hi, g))
    goal = _nearest_free(occ, _cell(goal_xz, lo, hi, g))
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    openq = [(0.0, start)]
    came: dict = {}
    gcost = {start: 0.0}
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == goal:
            break
        for di, dj in nbrs:
            nx = (cur[0] + di, cur[1] + dj)
            if not (0 <= nx[0] < g and 0 <= nx[1] < g) or occ[nx]:
                continue
            ng = gcost[cur] + np.hypot(di, dj)
            if ng < gcost.get(nx, 1e18):
                gcost[nx] = ng
                came[nx] = cur
                heapq.heappush(openq, (ng + np.hypot(goal[0] - nx[0], goal[1] - nx[1]), nx))
    if goal not in came and goal != start:
        return []
    path = [goal]
    while path[-1] != start:
        path.append(came[path[-1]])
    path.reverse()
    # cell -> world centre, then drop colinear points to keep only turn waypoints
    def w(c: tuple[int, int]) -> np.ndarray:
        return lo + (np.array(c) + 0.5) / g * span
    pts = [w(c) for c in path]
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = out[-1], pts[i], pts[i + 1]
        d1, d2 = b - a, c - b
        if abs(d1[0] * d2[1] - d1[1] * d2[0]) > 1e-6:  # direction change -> keep
            out.append(b)
    out.append(pts[-1])
    return out
