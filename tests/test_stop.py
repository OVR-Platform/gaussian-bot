"""Tests for termination policies (ADR-0006)."""

from __future__ import annotations

import numpy as np

from gaussian_robot.metrics.coverage import CoverageState
from gaussian_robot.nav.action import Action
from gaussian_robot.nav.stop import (
    BoundsGuard,
    CoveragePlateau,
    CoverageTarget,
    DiminishingReturns,
    PoseBudget,
    SeedExhaustion,
    SessionContext,
    WalkContext,
    session_stop_reason,
    walk_stop_reason,
)
from gaussian_robot.render.camera import Pose


def _ctx(action: Action, novelty: float, *, degenerate: bool = False, step: int = 1) -> WalkContext:
    return WalkContext(
        step=step, action=action, novelty=novelty, pose=Pose(), degenerate=degenerate
    )


def test_coverage_plateau_fires_on_low_novelty_window() -> None:
    plateau = CoveragePlateau(novelty_delta=1.0, window=3)
    for _ in range(2):
        plateau.update(_ctx(Action.FORWARD, novelty=0.1))
        assert not plateau.should_stop()
    plateau.update(_ctx(Action.FORWARD, novelty=0.1))
    assert plateau.should_stop()


def test_coverage_plateau_resets_on_novelty() -> None:
    plateau = CoveragePlateau(novelty_delta=1.0, window=3)
    plateau.update(_ctx(Action.FORWARD, novelty=0.1))
    plateau.update(_ctx(Action.FORWARD, novelty=0.1))
    plateau.update(_ctx(Action.FORWARD, novelty=5.0))  # novel -> reset
    plateau.update(_ctx(Action.FORWARD, novelty=0.1))
    assert not plateau.should_stop()


def test_coverage_plateau_counts_stop_vote() -> None:
    plateau = CoveragePlateau(novelty_delta=100.0, window=2)  # novelty never low
    plateau.update(_ctx(Action.STOP, novelty=999.0))
    plateau.update(_ctx(Action.STOP, novelty=999.0))
    assert plateau.should_stop()


def test_coverage_plateau_stop_alone_not_enough_if_novel() -> None:
    plateau = CoveragePlateau(novelty_delta=100.0, window=3)
    for _ in range(2):
        plateau.update(_ctx(Action.STOP, novelty=999.0))  # stop votes
    assert not plateau.should_stop()  # but a novel step cancels it
    plateau.update(_ctx(Action.FORWARD, novelty=999.0))
    assert not plateau.should_stop()


def test_coverage_plateau_ignores_rotations() -> None:
    # Scanning in place (turns/looks) must not be read as a plateau.
    plateau = CoveragePlateau(novelty_delta=1.0, window=2)
    for action in (Action.TURN_LEFT, Action.LOOK_UP, Action.TURN_RIGHT, Action.LOOK_DOWN):
        plateau.update(_ctx(action, novelty=0.0))
    assert not plateau.should_stop()
    plateau.update(_ctx(Action.FORWARD, novelty=0.0))
    plateau.update(_ctx(Action.FORWARD, novelty=0.0))
    assert plateau.should_stop()  # two unproductive *translations* do trigger it


def test_bounds_guard_fires_on_degenerate() -> None:
    guard = BoundsGuard()
    guard.update(_ctx(Action.FORWARD, novelty=1.0, degenerate=True))
    assert guard.should_stop()


def test_walk_stop_reason_names_firing_policy() -> None:
    guard = BoundsGuard()
    assert walk_stop_reason([guard]) is None
    guard.update(_ctx(Action.FORWARD, novelty=1.0, degenerate=True))
    assert walk_stop_reason([guard]) == "bounds"


def test_session_stop_reason_names_firing_policy() -> None:
    state = CoverageState.empty("y", np.zeros(3), np.array([10.0, 10.0, 10.0]))
    ctx = SessionContext(state=state, walks_completed=5, total_seeds=5)
    assert (
        session_stop_reason([PoseBudget(max_poses=10), SeedExhaustion()], ctx) == "seeds_exhausted"
    )
    assert session_stop_reason([PoseBudget(max_poses=10)], ctx) is None


def test_session_policies() -> None:
    bounds_min = np.zeros(3)
    bounds_max = np.array([10.0, 10.0, 10.0])
    state = CoverageState.empty("y", bounds_min, bounds_max)
    base = SessionContext(state=state, walks_completed=0, total_seeds=5)

    assert not PoseBudget(max_poses=10).should_stop(base)
    assert SeedExhaustion().should_stop(
        SessionContext(state=state, walks_completed=5, total_seeds=5)
    )
    assert not DiminishingReturns(epsilon=0.01).should_stop(base)  # no walks yet
    assert (
        CoverageTarget(radius=1.0, tau=0.5).should_stop(
            SessionContext(
                state=CoverageState.empty("y", bounds_min, bounds_max),
                walks_completed=1,
                total_seeds=5,
            )
        )
        is False
    )  # empty coverage is below target
