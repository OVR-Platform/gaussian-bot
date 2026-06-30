"""Pure (CPU, no-GPU) helpers of the before/after path movie.

The frame plan (densify + cap) and map extent are deterministic plumbing; they are factored out
of ``render_before_after_gif`` so they can be checked without loading a renderer.
"""

from __future__ import annotations

import numpy as np

from gaussian_robot.enhance.before_after import map_extent, plan_movie_poses
from gaussian_robot.render.camera import Pose


def _pose(x: float, z: float) -> Pose:
    return Pose(position=np.array([x, 0.0, z], dtype=np.float64), rotation=np.eye(3))


def test_plan_movie_poses_interpolates_and_caps() -> None:
    traj = [_pose(0.0, 0.0), _pose(1.0, 0.0), _pose(2.0, 0.0)]
    dense = plan_movie_poses(traj, per_segment=4, max_frames=240)
    assert len(dense) > len(traj)  # interpolation added frames
    # endpoints preserved
    assert np.allclose(dense[0].position, traj[0].position)
    assert np.allclose(dense[-1].position, traj[-1].position)
    # the cap is honoured
    capped = plan_movie_poses(traj, per_segment=50, max_frames=10)
    assert len(capped) == 10


def test_plan_movie_poses_short_trajectory() -> None:
    assert plan_movie_poses([], per_segment=4) == []
    one = [_pose(0.0, 0.0)]
    assert plan_movie_poses(one, per_segment=4) == one


def test_map_extent_covers_points_with_margin() -> None:
    pts = np.array([[0.0, 0.0], [4.0, 2.0]], dtype=np.float64)
    lo, span = map_extent(pts, margin=1.0)
    assert np.allclose(lo, [-1.0, -1.0])
    assert np.allclose(span, [6.0, 4.0])
    # empty -> safe unit box (never zero span)
    lo0, span0 = map_extent(np.empty((0, 2)))
    assert np.all(span0 > 0)
