"""Filter raw walk trajectories into a clean deliverable pose set (ADR-0008).

Three stages over the global union of samples:

1. **Quality drop** — discard poses whose render confidence is too low.
2. **Novelty dedup** — greedy farthest-point selection on floor-plane positions so
   every kept pose is at least ``r_keep`` from the others.
3. **Budget cap** — stop selecting once ``budget`` poses are kept.

Stages 2 and 3 are unified in a single farthest-point pass: select until either
the next-best pose is within ``r_keep`` of a kept one (dedup satisfied) or the
budget is reached.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaussian_robot.metrics.coverage import PoseSample, floor_xy
from gaussian_robot.render.camera import Pose


@dataclass(frozen=True)
class FilteredPose:
    """A pose that survived filtering, with selection metadata."""

    pose: Pose
    seed_id: str
    novelty: float
    confidence: float


def _pairwise_min_dist(points: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Min Euclidean distance from each row of ``points`` to ``query`` (2D)."""
    d = points - query
    dists: np.ndarray = np.sqrt(np.einsum("ij,ij->i", d, d))
    return dists


def farthest_point_select(points: np.ndarray, *, r_keep: float, budget: int) -> list[int]:
    """Greedy farthest-point selection over ``(N, 2)`` floor-plane ``points``.

    Returns selected indices. Stops when the next-best candidate is closer than
    ``r_keep`` to an already-selected point, when ``budget`` is reached, or when
    all points are selected.
    """
    n = points.shape[0]
    if n == 0:
        return []
    budget = max(1, min(budget, n))
    neg_inf = float("-inf")
    selected: list[int] = [0]
    min_d = _pairwise_min_dist(points, points[0])
    min_d[0] = neg_inf  # never re-select
    while len(selected) < budget:
        i = int(np.argmax(min_d))
        if float(min_d[i]) < r_keep:
            break
        selected.append(i)
        min_d = np.minimum(min_d, _pairwise_min_dist(points, points[i]))
        min_d[i] = neg_inf
    return selected


def filter_poses(
    samples: list[PoseSample],
    *,
    up_axis: str,
    r_keep: float,
    budget: int = 200,
    min_confidence: float = 0.0,
) -> list[FilteredPose]:
    """Apply quality drop → novelty dedup → budget cap to ``samples``.

    Parameters
    ----------
    samples:
        Global union of all walk steps (e.g. from ``CoverageState.samples``).
    up_axis:
        World up axis, for the floor-plane projection used in dedup.
    r_keep:
        Minimum floor-plane spacing between kept poses (the coverage radius).
    budget:
        Maximum number of poses to keep.
    min_confidence:
        Quality threshold; samples below it are dropped before dedup.
    """
    kept = [s for s in samples if s.confidence >= min_confidence]
    if not kept:
        return []

    points = np.array([floor_xy(s.pose.position, up_axis)[0] for s in kept], dtype=np.float64)
    selected = farthest_point_select(points, r_keep=r_keep, budget=budget)

    out: list[FilteredPose] = []
    selected_points = points[selected]
    for rank, idx in enumerate(selected):
        sample = kept[idx]
        mask = np.ones(len(selected), dtype=bool)
        mask[rank] = False
        if mask.any():
            novelty = float(_pairwise_min_dist(selected_points[mask], points[idx]).min())
        else:
            novelty = float("inf")
        out.append(
            FilteredPose(
                pose=sample.pose,
                seed_id=sample.seed_id,
                novelty=novelty,
                confidence=sample.confidence,
            )
        )
    return out
