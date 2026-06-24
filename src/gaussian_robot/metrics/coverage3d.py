"""Tier-3 3D coverage (ADR-0007): which reconstructed voxels the cameras actually saw.

A voxel **occupancy** grid is built from the gaussian means (opacity-weighted). A
**seen** grid is built by ray-casting each capture camera's frustum onto the occupancy
and marking the *first* occupied voxel each ray hits — i.e. the surface actually visible
to that camera, so occlusion is handled (a roof behind a wall is not "seen through" it).

``occupied & ~seen`` are the real 3D gaps — roofs, upper floors, behind-building pockets —
that the 2D floor-density frontier is blind to. These drive seeding, the aerial survey,
auto-marking and the deliverable toward genuinely under-observed regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Coverage3D:
    """Voxel occupancy + per-voxel "seen by a capture camera" mask over a 3D grid."""

    occupied: np.ndarray  # (G, G, G) bool
    seen: np.ndarray  # (G, G, G) bool
    lo: np.ndarray  # (3,) grid min corner
    hi: np.ndarray  # (3,) grid max corner

    @property
    def gap_mask(self) -> np.ndarray:
        """Occupied voxels that no camera saw — the 3D coverage gaps."""
        mask: np.ndarray = self.occupied & ~self.seen
        return mask

    def gap_centers(self) -> np.ndarray:
        """World ``(K, 3)`` centres of the gap voxels (empty ``(0, 3)`` if none)."""
        idx = np.argwhere(self.gap_mask)
        if idx.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)
        g = np.array(self.occupied.shape, dtype=np.float64)
        centers: np.ndarray = self.lo + (idx + 0.5) / g * (self.hi - self.lo)
        return centers

    def nearest_gap(self, position: np.ndarray) -> np.ndarray | None:
        """World centre of the nearest gap voxel to ``position`` (3D), or None."""
        centers = self.gap_centers()
        if centers.shape[0] == 0:
            return None
        nearest: np.ndarray = centers[int(np.argmin(((centers - position) ** 2).sum(axis=1)))]
        return nearest


def _voxel_index(points: np.ndarray, lo: np.ndarray, hi: np.ndarray, g: int) -> np.ndarray:
    """Map world ``points`` (..., 3) to integer voxel indices (..., 3) in a ``g^3`` grid."""
    idx: np.ndarray = np.floor((points - lo) / (hi - lo) * g).astype(np.int64)
    return idx


def build_coverage3d(
    means: np.ndarray,
    opacities: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    hfov: np.ndarray,
    vfov: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    grid: int = 32,
    occ_min: float = 0.5,
    rays: int = 18,
    steps: int = 64,
    max_cams: int = 800,
) -> Coverage3D:
    """Build a :class:`Coverage3D` from gaussians and capture cameras.

    ``cam_rot`` are world->camera rotations (rows = right, down, forward; ADR-0002).
    ``hfov``/``vfov`` are per-camera horizontal/vertical fields of view (radians). Cameras
    are sub-sampled to ``max_cams``; each casts a ``rays x rays`` grid over its FOV,
    marching ``steps`` samples and marking the first occupied voxel hit as seen.
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    extent = hi - lo
    occupied: np.ndarray = np.zeros((grid, grid, grid), dtype=bool)
    if means.shape[0] > 0 and np.all(extent > 0):
        edges = [np.linspace(lo[d], hi[d], grid + 1) for d in range(3)]
        hist, _ = np.histogramdd(means, bins=edges, weights=opacities)
        occupied = hist > occ_min

    seen = np.zeros_like(occupied)
    n = cam_pos.shape[0]
    if n == 0 or not occupied.any():
        return Coverage3D(occupied=occupied, seen=seen, lo=lo, hi=hi)

    sel = np.linspace(0, n - 1, min(n, max_cams)).astype(np.int64) if n > max_cams else np.arange(n)
    max_range = float(np.linalg.norm(extent))
    t = np.linspace(max_range / (steps * 2), max_range, steps)  # sample distances along each ray
    su = np.linspace(-1.0, 1.0, rays)
    gx, gy = np.meshgrid(su, su, indexing="xy")
    gx, gy = gx.ravel(), gy.ravel()  # (R,)

    for m in sel:
        right, down, fwd = cam_rot[m]  # world axes of this camera
        sx = gx * np.tan(hfov[m] / 2.0)
        sy = gy * np.tan(vfov[m] / 2.0)
        dirs = fwd[None, :] + sx[:, None] * right[None, :] + sy[:, None] * down[None, :]
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)  # (R, 3)
        pts = cam_pos[m][None, None, :] + t[None, :, None] * dirs[:, None, :]  # (R, steps, 3)
        idx = _voxel_index(pts, lo, hi, grid)
        inb = np.all((idx >= 0) & (idx < grid), axis=-1)  # (R, steps)
        ci = np.clip(idx, 0, grid - 1)
        occ_at = occupied[ci[..., 0], ci[..., 1], ci[..., 2]] & inb  # (R, steps)
        has_hit = occ_at.any(axis=1)
        if not has_hit.any():
            continue
        first = occ_at.argmax(axis=1)  # first occupied step per ray
        rows = np.nonzero(has_hit)[0]
        hit = ci[rows, first[rows]]  # (H, 3) voxel of the first hit
        seen[hit[:, 0], hit[:, 1], hit[:, 2]] = True

    return Coverage3D(occupied=occupied, seen=seen, lo=lo, hi=hi)
