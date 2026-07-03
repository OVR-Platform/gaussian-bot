"""GoalReached: the instruction-tied geometric walk terminator (ADR-0012)."""

from __future__ import annotations

import numpy as np

from gaussian_robot.nav.action import Action
from gaussian_robot.nav.stop import (
    GoalReached,
    SessionContext,
    TaskComplete,
    TaskStop,
    WalkContext,
    walk_stop_reason,
)
from gaussian_robot.render.camera import Pose


def _ctx(pos: tuple[float, float, float], action: Action = Action.FORWARD) -> WalkContext:
    return WalkContext(
        step=1, action=action, novelty=0.1, pose=Pose(position=np.array(pos, dtype=np.float64))
    )


def test_goal_reached_fires_only_within_eps_on_the_floor_plane() -> None:
    p = GoalReached(target=np.array([2.0, 0.0, 0.0]), eps=0.5, up_axis="y")
    p.reset()
    p.update(_ctx((0.0, 0.0, 0.0)))
    assert not p.should_stop()
    # Height (the up axis) must not count: 10m above the goal is still "at" it on the floor.
    p.update(_ctx((2.2, 10.0, 0.0)))
    assert p.should_stop()
    assert p.reason == "goal_reached"


def test_goal_reached_latches_once_reached() -> None:
    p = GoalReached(target=np.zeros(3), eps=1.0, up_axis="y")
    p.reset()
    p.update(_ctx((0.5, 0.0, 0.0)))
    p.update(_ctx((5.0, 0.0, 5.0)))  # walked away afterwards
    assert p.should_stop()
    p.reset()
    assert not p.should_stop()


def test_goal_reached_wins_the_reason_over_a_simultaneous_vlm_stop() -> None:
    goal = GoalReached(target=np.zeros(3), eps=1.0, up_axis="y")
    task = TaskStop()
    for p in (goal, task):
        p.reset()
    ctx = _ctx((0.2, 0.0, 0.2), action=Action.STOP)
    for p in (goal, task):
        p.update(ctx)
    # Composed goal-first (as the navigate CLI does): the measured reason is reported.
    assert walk_stop_reason([goal, task]) == "goal_reached"


def test_task_complete_session_policy_accepts_goal_reached() -> None:
    tc = TaskComplete()
    ctx = SessionContext(state=None, walks_completed=1, total_seeds=1)  # type: ignore[arg-type]
    for reason, expected in (("goal_reached", True), ("task_complete", True), ("stuck", False)):
        ctx.last_walk_reason = reason
        assert tc.should_stop(ctx) is expected
