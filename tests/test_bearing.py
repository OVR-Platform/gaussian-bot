"""Steering correctness: following the bearing cue must reduce the angle to the target.

A latent bug: the floor 'right' was guessed from forward as [fwd_b, -fwd_a], wrong-handed for a
negative up axis (e.g. -y), which inverted the left/right cue so the robot/teacher turned the
WRONG way. The operational invariant that must hold (any up axis): turning toward the reported
side brings the target closer to dead-ahead.
"""

from __future__ import annotations

import numpy as np

from gaussian_robot.nav.action import Action, ActionSpace, apply_action
from gaussian_robot.nav.observation import ObservationBuilder
from gaussian_robot.render.camera import Pose


def _bearing(b: ObservationBuilder, pose: Pose, target: np.ndarray) -> float:
    from gaussian_robot.metrics.coverage import floor_xy  # noqa: PLC0415

    info = b._gap_bearing(
        pose, floor_xy(pose.position, b.up_axis)[0], floor_xy(target, b.up_axis)[0]
    )
    assert info is not None
    return info[1]


def _follows_to_target(up_axis: str, target: np.ndarray) -> bool:
    b = ObservationBuilder(renderer=None, up_axis=up_axis)  # type: ignore[arg-type]
    space = ActionSpace(step=1.0)
    pose = Pose(position=np.zeros(3), rotation=np.eye(3))
    b0 = _bearing(b, pose, target)
    # State-line rule: side = right if bearing > 0 else left; turn that side.
    turn = Action.TURN_RIGHT if b0 > 0 else Action.TURN_LEFT
    b1 = _bearing(b, apply_action(pose, turn, space, up_axis), target)
    return abs(b1) < abs(b0)


def test_steering_reduces_angle_both_sides_minus_y() -> None:
    assert _follows_to_target("-y", np.array([2.0, 0.0, 1.0]))  # target off to one side
    assert _follows_to_target("-y", np.array([-2.0, 0.0, 1.0]))  # and the other


def test_steering_reduces_angle_plus_y() -> None:
    assert _follows_to_target("y", np.array([2.0, 0.0, 1.0]))
    assert _follows_to_target("y", np.array([-2.0, 0.0, 1.0]))
