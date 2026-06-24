"""Interpolated walk replay: pose densification + rotation slerp."""

from __future__ import annotations

import numpy as np

from gaussian_robot.nav.action import _rotation_about_axis
from gaussian_robot.render.camera import Pose
from gaussian_robot.session import _rotation_geodesic, interpolate_walk_poses


def test_interpolate_densifies_moving_segments() -> None:
    a = Pose(position=np.zeros(3))
    b = Pose(position=np.array([0.0, 0.0, 4.0]))
    out = interpolate_walk_poses([a, b], per_segment=4)
    assert len(out) == 5  # 4 per segment + the final endpoint
    assert np.allclose(out[0].position, [0.0, 0.0, 0.0])
    assert np.allclose(out[2].position, [0.0, 0.0, 2.0])  # t=0.5 midpoint
    assert np.allclose(out[-1].position, [0.0, 0.0, 4.0])


def test_interpolate_skips_static_segments() -> None:
    a = Pose(position=np.zeros(3))
    out = interpolate_walk_poses([a, a, a], per_segment=8)  # blocked/stop: no motion
    assert len(out) == 3  # one frame per static segment + final endpoint


def test_rotation_geodesic_endpoints_and_midpoint() -> None:
    r0 = np.eye(3)
    r1 = _rotation_about_axis(np.array([0.0, 1.0, 0.0]), np.deg2rad(30.0))
    assert np.allclose(_rotation_geodesic(r0, r1, 0.0), r0)
    assert np.allclose(_rotation_geodesic(r0, r1, 1.0), r1)
    expected_half = _rotation_about_axis(np.array([0.0, 1.0, 0.0]), np.deg2rad(15.0)) @ r0
    assert np.allclose(_rotation_geodesic(r0, r1, 0.5), expected_half)
