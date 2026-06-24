"""Coarse ground-height field for terrain-following on non-flat scenes.

A walk locks its camera at the seed's height; on a sloped scene (riverbank, valley)
that sinks the camera underground or floats it. :class:`HeightField` estimates the
ground level across the floor once (a low percentile of gaussian heights per cell),
so the explorer can keep the camera at a constant eye-height above local ground.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaussian_robot.render.camera import up_vector


@dataclass(frozen=True)
class HeightField:
    """Per-cell ground height (signed up-axis coordinate) over the floor (x, z) plane.

    ``grid[ix, iz]`` is the estimated ground height in the cell, or ``NaN`` where no
    gaussians were seen. Heights are ``position @ up`` so the sign matches the up axis
    (e.g. up = ``-y``).
    """

    grid: np.ndarray  # (G, G), NaN where unknown
    lo: np.ndarray  # (3,) floor bounds min
    hi: np.ndarray  # (3,) floor bounds max
    up_axis: str

    def ground(self, x: float, z: float) -> float | None:
        """Ground height (signed up-coord) at world ``(x, z)``, or ``None`` if unknown."""
        g = self.grid.shape[0]
        wx, wz = self.hi[0] - self.lo[0], self.hi[2] - self.lo[2]
        if wx <= 0 or wz <= 0:
            return None
        ix = int(np.clip((x - self.lo[0]) / wx * g, 0, g - 1))
        iz = int(np.clip((z - self.lo[2]) / wz * g, 0, g - 1))
        h = float(self.grid[ix, iz])
        return None if np.isnan(h) else h


def build_height_field(
    means: np.ndarray,
    up_axis: str,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    grid_size: int = 48,
    ground_q: float = 0.2,
) -> HeightField:
    """Build a ground-height field from gaussian ``means`` (``(N, 3)`` world points).

    Each floor cell's ground is the ``ground_q`` quantile of the heights
    (``means @ up``) of the gaussians binned into it — a low percentile picks the
    floor rather than walls/canopy above it, while being robust to a few stray
    below-ground floaters.
    """
    up = up_vector(up_axis)
    g = grid_size
    grid = np.full(g * g, np.nan, dtype=np.float64)
    wx, wz = float(hi[0] - lo[0]), float(hi[2] - lo[2])
    if means.shape[0] == 0 or wx <= 0 or wz <= 0:
        return HeightField(grid=grid.reshape(g, g), lo=lo, hi=hi, up_axis=up_axis)

    heights = means @ up
    ix = np.clip(((means[:, 0] - lo[0]) / wx * g).astype(np.int64), 0, g - 1)
    iz = np.clip(((means[:, 2] - lo[2]) / wz * g).astype(np.int64), 0, g - 1)
    flat = ix * g + iz

    order = np.argsort(flat, kind="stable")
    flat_s, h_s = flat[order], heights[order]
    uniq, starts = np.unique(flat_s, return_index=True)
    ends = np.append(starts[1:], len(flat_s))
    for cell, s, e in zip(uniq, starts, ends, strict=True):
        grid[cell] = float(np.quantile(h_s[s:e], ground_q))
    return HeightField(grid=grid.reshape(g, g), lo=lo, hi=hi, up_axis=up_axis)
