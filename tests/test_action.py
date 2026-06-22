"""Tests for the egocentric action executor (ADR-0002, ADR-0004)."""

from __future__ import annotations

import numpy as np

from gaussian_robot.nav.action import Action, ActionSpace, apply_action
from gaussian_robot.render.camera import Pose

_SPACE = ActionSpace(step=1.0, delta_rot=np.deg2rad(30.0))


def test_action_verbs_match_adr() -> None:
    assert set(Action.verbs()) == {
        "forward",
        "back",
        "turn_left",
        "turn_right",
        "look_up",
        "look_down",
        "move_up",
        "move_down",
        "stop",
    }


def test_action_space_from_bounds_scales_with_diagonal() -> None:
    space = ActionSpace.from_bounds(np.zeros(3), np.array([10.0, 0.0, 0.0]))
    assert np.isclose(space.step, 0.3)  # 3% of diagonal length 10


def test_stop_is_noop() -> None:
    pose = Pose(position=np.array([1.0, 2.0, 3.0]))
    assert apply_action(pose, Action.STOP, _SPACE) == pose


def test_forward_moves_along_horizontal_heading() -> None:
    pose = Pose(position=np.zeros(3))  # looks +Z (OpenCV identity)
    moved = apply_action(pose, Action.FORWARD, _SPACE, up_axis="y")
    assert np.allclose(moved.position, [0.0, 0.0, 1.0])
    assert np.allclose(moved.rotation, pose.rotation)


def test_back_moves_opposite() -> None:
    pose = Pose(position=np.zeros(3))
    moved = apply_action(pose, Action.BACK, _SPACE, up_axis="y")
    assert np.allclose(moved.position, [0.0, 0.0, -1.0])


def test_turn_left_yaws_around_up_axis() -> None:
    pose = Pose(position=np.zeros(3))
    turned = apply_action(pose, Action.TURN_LEFT, _SPACE, up_axis="y")
    new_forward = turned.forward()
    assert abs(new_forward[1]) < 1e-9  # still horizontal
    # Turn left rotates forward (+Z) toward camera-left (-X): forward -> [-sin, 0, cos].
    assert np.allclose(new_forward, [-np.sin(np.deg2rad(30.0)), 0.0, np.cos(np.deg2rad(30.0))])


def test_forward_stays_level_when_looking_up() -> None:
    pose = Pose(position=np.zeros(3))
    looking_up = apply_action(pose, Action.LOOK_UP, _SPACE, up_axis="y")
    assert looking_up.forward()[1] > 0.4  # genuinely pitched up toward +Y
    moved = apply_action(looking_up, Action.FORWARD, _SPACE, up_axis="y")
    assert abs(moved.position[1]) < 1e-9  # translation stays on floor plane


def test_move_up_translates_along_up_axis() -> None:
    pose = Pose(position=np.zeros(3))
    moved = apply_action(pose, Action.MOVE_UP, _SPACE, up_axis="y")
    assert np.allclose(moved.position, [0.0, 1.0, 0.0])
    assert np.allclose(moved.rotation, pose.rotation)


def test_move_down_translates_opposite_up_axis() -> None:
    pose = Pose(position=np.array([0.0, 5.0, 0.0]))
    moved = apply_action(pose, Action.MOVE_DOWN, _SPACE, up_axis="y")
    assert np.allclose(moved.position, [0.0, 4.0, 0.0])
    assert np.allclose(moved.rotation, pose.rotation)
