"""Coverage state and exploration metrics (ADR-0007, Tier 1 & 2).

This module is the shared backbone for termination (ADR-0006), the dedup filter
(ADR-0008) and the metrics themselves. It is pure numpy and depends only on
``Pose``.

Two flat metrics are implemented now:

- **Tier 1 — ``floor_coverage``**: fraction of navigable floor cells (within
  radius ``r`` of a sampled pose) that are covered, over a grid spanning the
  scene AABB projected onto the floor plane.
- **Tier 2 — ``pose_space_coverage``**: extends Tier 1 by also binning the
  *viewing direction*, so two poses at the same spot looking the same way do not
  double-count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gaussian_robot.render.camera import Pose

_UP_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_FLOOR_AXES = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}


def _floor_axes_for(up_axis: str) -> tuple[int, int]:
    try:
        return _FLOOR_AXES[up_axis.lower()]
    except KeyError:
        raise ValueError(f"up_axis must be one of x/y/z, got {up_axis!r}") from None


def floor_xy(position: np.ndarray, up_axis: str) -> np.ndarray:
    """Project a ``(3,)`` or ``(N, 3)`` array onto the two floor-plane axes."""
    ax_a, ax_b = _floor_axes_for(up_axis)
    return np.atleast_2d(position)[:, (ax_a, ax_b)]


def viewing_direction(rotation: np.ndarray) -> np.ndarray:
    """World-space viewing direction (unit) for a ``(3, 3)`` world->camera matrix.

    OpenCV convention (ADR-0002): camera +Z is forward, which is the third row of
    a world->camera rotation.
    """
    fwd = rotation[2, :].astype(np.float64)
    n = np.linalg.norm(fwd)
    if n < 1e-12:
        raise ValueError("rotation has degenerate forward direction")
    out: np.ndarray = fwd / n
    return out


@dataclass(frozen=True)
class PoseSample:
    """A sampled pose plus provenance/metadata used by metrics and filtering."""

    pose: Pose
    seed_id: str = ""
    confidence: float = 1.0


@dataclass
class CoverageState:
    """Accumulates sampled poses and answers novelty/coverage queries.

    Attributes
    ----------
    up_axis:
        World axis treated as up (drives the floor-plane projection).
    bounds_min / bounds_max:
        ``(3,)`` scene AABB; used to grid the floor for coverage metrics.
    """

    up_axis: str
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    _samples: list[PoseSample] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.bounds_min.shape != (3,) or self.bounds_max.shape != (3,):
            raise ValueError("bounds must be (3,) arrays")

    @classmethod
    def empty(cls, up_axis: str, bounds_min: np.ndarray, bounds_max: np.ndarray) -> CoverageState:
        return cls(up_axis=up_axis, bounds_min=bounds_min, bounds_max=bounds_max)

    @property
    def samples(self) -> list[PoseSample]:
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, sample: PoseSample) -> None:
        self._samples.append(sample)

    def add_pose(self, pose: Pose, *, seed_id: str = "", confidence: float = 1.0) -> None:
        self._samples.append(PoseSample(pose=pose, seed_id=seed_id, confidence=confidence))

    def positions(self) -> np.ndarray:
        """``(N, 3)`` array of sampled positions (empty ``(0, 3)`` if none)."""
        if not self._samples:
            return np.empty((0, 3), dtype=np.float64)
        return np.array([s.pose.position for s in self._samples], dtype=np.float64)

    def floor_positions(self) -> np.ndarray:
        """``(N, 2)`` sampled positions projected onto the floor plane."""
        return floor_xy(self.positions(), self.up_axis)

    def novelty(self, pose: Pose) -> float:
        """Min floor-plane distance from ``pose`` to any sampled pose.

        Returns ``+inf`` when nothing has been sampled yet (the pose is
        maximally novel).
        """
        return novelty(pose.position, self.floor_positions(), self.up_axis)


def novelty(position: np.ndarray, sampled_floor: np.ndarray, up_axis: str) -> float:
    """Min floor-plane distance from ``position`` to the rows of ``sampled_floor``.

    ``+inf`` when ``sampled_floor`` is empty.
    """
    pos2 = floor_xy(np.asarray(position, dtype=np.float64), up_axis)[0]
    if sampled_floor.size == 0:
        return float("inf")
    diffs: np.ndarray = sampled_floor - pos2
    dists: np.ndarray = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
    return float(dists.min())


def _floor_grid(state: CoverageState, grid_cells: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Build the floor-cell centre grid and sampled floor positions.

    Returns ``(centres, sampled)``, or ``None`` when the AABB footprint is
    degenerate or no poses have been sampled (both mean zero coverage).
    """
    a, b = _floor_axes_for(state.up_axis)
    mins = state.bounds_min[[a, b]].astype(np.float64)
    maxs = state.bounds_max[[a, b]].astype(np.float64)
    if np.any(maxs <= mins):
        return None

    xs = np.linspace(mins[0], maxs[0], grid_cells)
    ys = np.linspace(mins[1], maxs[1], grid_cells)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    centres: np.ndarray = np.stack([gx.ravel(), gy.ravel()], axis=1)

    sampled = state.floor_positions()
    if sampled.size == 0:
        return None
    return centres, sampled


def floor_coverage(state: CoverageState, *, radius: float, grid_cells: int = 64) -> float:
    """Tier-1 metric: fraction of navigable floor cells covered within ``radius``.

    A cell is "navigable" if it lies inside the AABB footprint and "covered" if
    at least one sampled pose is within ``radius`` (floor-plane) of its centre.
    Cells outside the AABB are ignored.
    """
    grid = _floor_grid(state, grid_cells)
    if grid is None:
        return 0.0
    centres, sampled = grid
    r2 = float(radius) ** 2

    covered_mask = np.zeros(centres.shape[0], dtype=bool)
    for s in sampled:
        d = centres - s
        covered_mask |= np.einsum("ij,ij->i", d, d) <= r2
    return float(covered_mask.mean())


def quality_floor_coverage(
    state: CoverageState,
    *,
    radius: float,
    q_min: float = 0.5,
    grid_cells: int = 64,
) -> float:
    """Like floor_coverage but only counting poses with confidence >= q_min.

    A cell is 'well-covered' only if a high-quality observation (render alpha
    >= q_min) was recorded within radius. This ties the metric to actual
    reconstruction quality rather than pure visitation.
    """
    a, b = _floor_axes_for(state.up_axis)
    mins = state.bounds_min[[a, b]].astype(np.float64)
    maxs = state.bounds_max[[a, b]].astype(np.float64)
    if np.any(maxs <= mins):
        return 0.0

    hi_q = [s.pose.position for s in state.samples if s.confidence >= q_min]
    if not hi_q:
        return 0.0

    xs = np.linspace(mins[0], maxs[0], grid_cells)
    ys = np.linspace(mins[1], maxs[1], grid_cells)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    centres = np.stack([gx.ravel(), gy.ravel()], axis=1)

    sampled = floor_xy(np.array(hi_q, dtype=np.float64), state.up_axis)
    r2 = float(radius) ** 2
    covered = np.zeros(centres.shape[0], dtype=bool)
    for s in sampled:
        d = centres - s
        covered |= np.einsum("ij,ij->i", d, d) <= r2
    return float(covered.mean())


def pose_space_coverage(
    state: CoverageState, *, radius: float, dir_bins: int = 8, grid_cells: int = 32
) -> float:
    """Tier-2 metric: coverage over ``(position cell × viewing-direction bin)``.

    Like :func:`floor_coverage` but a covered cell is only counted for the
    *direction bins* its observers face. Returns the fraction of
    ``(cell, bin)`` pairs that are occupied over navigable cells.
    """
    grid = _floor_grid(state, grid_cells)
    if grid is None:
        return 0.0
    centres, sampled = grid
    r2 = float(radius) ** 2

    a, b = _floor_axes_for(state.up_axis)
    dirs = np.array([viewing_direction(s.pose.rotation) for s in state.samples], dtype=np.float64)
    dir_a = dirs[:, a]
    dir_b = dirs[:, b]
    angles = np.arctan2(dir_b, dir_a)
    bins = ((angles + np.pi) / (2 * np.pi) * dir_bins).astype(int) % dir_bins

    occupied = np.zeros((centres.shape[0], dir_bins), dtype=bool)
    for s, bn in zip(sampled, bins, strict=True):
        d = centres - s
        near = np.einsum("ij,ij->i", d, d) <= r2
        occupied[near, bn] = True

    total_pairs = centres.shape[0] * dir_bins
    return float(occupied.sum() / total_pairs)
