"""Tests for the dedup/filter pipeline (ADR-0008)."""

from __future__ import annotations

import numpy as np

from gaussian_robot.filters.pose_filters import farthest_point_select, filter_poses
from gaussian_robot.metrics.coverage import PoseSample
from gaussian_robot.render.camera import Pose


def _sample(x: float, z: float, *, confidence: float = 1.0, walk: str = "w") -> PoseSample:
    return PoseSample(
        pose=Pose(position=np.array([x, 0.0, z])), walk_id=walk, confidence=confidence
    )


def test_quality_drop_removes_low_confidence() -> None:
    samples = [_sample(0.0, 0.0, confidence=0.1), _sample(5.0, 5.0, confidence=1.0)]
    out = filter_poses(samples, up_axis="y", r_keep=0.5, budget=10, min_confidence=0.5)
    assert len(out) == 1
    assert np.allclose(out[0].pose.position, [5.0, 0.0, 5.0])


def test_novelty_dedup_collapses_cluster() -> None:
    samples = [_sample(x, 0.0) for x in (0.0, 0.01, 0.02, 10.0)]
    out = filter_poses(samples, up_axis="y", r_keep=1.0, budget=10)
    positions = sorted(np.linalg.norm(o.pose.position) for o in out)
    assert len(out) == 2  # one near origin cluster + the far one
    assert positions[-1] > 9.0


def test_budget_cap_limits_output() -> None:
    samples = [_sample(float(x), float(z)) for x in range(10) for z in range(10)]
    out = filter_poses(samples, up_axis="y", r_keep=0.0, budget=5)
    assert len(out) == 5


def test_farthest_point_select_empty() -> None:
    assert farthest_point_select(np.empty((0, 2)), r_keep=1.0, budget=5) == []


def test_filtered_pose_has_finite_novelty_except_single() -> None:
    samples = [_sample(0.0, 0.0), _sample(5.0, 5.0)]
    out = filter_poses(samples, up_axis="y", r_keep=0.0, budget=10)
    assert len(out) == 2
    novelties = sorted(o.novelty for o in out)
    assert novelties[1] > 0  # the second-selected has a real nearest-neighbour
