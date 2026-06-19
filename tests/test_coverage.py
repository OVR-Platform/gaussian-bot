"""Tests for coverage state and Tier-1/Tier-2 metrics (ADR-0007)."""

from __future__ import annotations

import numpy as np

from gaussian_robot.metrics.coverage import (
    CoverageState,
    floor_coverage,
    novelty,
    pose_space_coverage,
)
from gaussian_robot.render.camera import Pose


def _state(up_axis: str = "y") -> CoverageState:
    return CoverageState.empty(
        up_axis=up_axis,
        bounds_min=np.zeros(3),
        bounds_max=np.array([10.0, 10.0, 10.0]),
    )


def _pose(x: float, z: float) -> Pose:
    return Pose(position=np.array([x, 0.0, z]))


def test_novelty_inf_with_no_samples() -> None:
    assert novelty(np.zeros(3), np.empty((0, 2)), "y") == float("inf")


def test_novelty_is_min_floor_distance() -> None:
    sampled = np.array([[0.0, 0.0], [2.0, 0.0]])
    assert np.isclose(novelty(np.array([1.0, 0.0, 0.0]), sampled, "y"), 1.0)


def test_coverage_state_novelty_after_add() -> None:
    state = _state()
    assert state.novelty(_pose(1.0, 1.0)) == float("inf")
    state.add_pose(_pose(0.0, 0.0))
    assert np.isclose(state.novelty(_pose(3.0, 4.0)), 5.0)  # floor distance 3-4-5


def test_floor_coverage_grows_with_samples() -> None:
    state = _state()
    assert floor_coverage(state, radius=1.0) == 0.0
    for x in np.linspace(0.5, 9.5, 10):
        for z in np.linspace(0.5, 9.5, 10):
            state.add_pose(_pose(float(x), float(z)))
    cov = floor_coverage(state, radius=1.5)
    assert cov > 0.9


def test_pose_space_coverage_distinguishes_directions() -> None:
    state = _state()
    pos = _pose(5.0, 5.0)
    looking_z = Pose(position=pos.position, rotation=np.eye(3))  # +Z forward
    state.add_pose(looking_z)
    cov_one_dir = pose_space_coverage(state, radius=2.0, dir_bins=8)

    state2 = _state()
    for _ in range(4):
        state2.add_pose(looking_z)
    cov_same = pose_space_coverage(state2, radius=2.0, dir_bins=8)
    # Repeated identical direction does not increase pose-space coverage.
    assert np.isclose(cov_one_dir, cov_same)


def test_pose_space_coverage_rises_with_diverse_directions() -> None:
    state = _state()
    pos = np.zeros(3)
    for angle_deg in (0, 90, 180, 270):
        a = np.deg2rad(angle_deg)
        rot = np.eye(3)
        rot[2, :] = [0.0, 0.0, 1.0] if angle_deg == 0 else [np.sin(a), 0.0, np.cos(a)]
        state.add_pose(Pose(position=pos, rotation=rot))
    diverse = pose_space_coverage(state, radius=2.0, dir_bins=8)
    assert diverse > 0.0
