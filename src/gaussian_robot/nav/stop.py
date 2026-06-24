"""Termination policies (ADR-0006).

Two levels, each a set of small composable policies:

- **Walk-level** :class:`StopPolicy` — ends one local-control episode. Evaluated
  every step with a :class:`WalkContext`. The VLM's ``STOP`` action is *demoted*
  here: :class:`CoveragePlateau` counts it as one plateau vote, but it cannot
  end a walk on its own.
- **Session-level** :class:`SessionStopPolicy` — ends the whole exploration.
  Evaluated after each walk with a :class:`SessionContext`.

Compose many with ``any(...)`` (OR semantics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from gaussian_robot.metrics.coverage import CoverageState, floor_coverage
from gaussian_robot.nav.action import Action
from gaussian_robot.render.camera import Pose

# Actions that don't translate on the FLOOR plane. Their (near-zero) floor-novelty
# must not be read as a coverage plateau: turns/looks/describe/mark don't move, and
# move_up/move_down change only height (novelty is floor-plane only) — so a vertical
# climb to inspect roofs is neutral, not a plateau vote.
_NON_TRANSLATING = frozenset(
    {
        Action.TURN_LEFT,
        Action.TURN_RIGHT,
        Action.LOOK_UP,
        Action.LOOK_DOWN,
        Action.MOVE_UP,
        Action.MOVE_DOWN,
        Action.DESCRIBE,
        Action.MARK,
    }
)


@dataclass
class WalkContext:
    """Per-step snapshot handed to walk-level stop policies."""

    step: int
    action: Action
    novelty: float
    pose: Pose
    degenerate: bool = False


@runtime_checkable
class StopPolicy(Protocol):
    """Ends a walk. Stateful: call :meth:`reset` before each walk."""

    reason: str

    def reset(self) -> None: ...

    def update(self, ctx: WalkContext) -> None: ...

    def should_stop(self) -> bool: ...


@dataclass
class BoundsGuard:
    """Stops when a render is degenerate (out of bounds / empty)."""

    reason: str = "bounds"
    _triggered: bool = False

    def reset(self) -> None:
        self._triggered = False

    def update(self, ctx: WalkContext) -> None:
        if ctx.degenerate:
            self._triggered = True

    def should_stop(self) -> bool:
        return self._triggered


@dataclass
class CoveragePlateau:
    """Primary, objective walk-terminator.

    Counts consecutive **translation** steps whose novelty is below
    ``novelty_delta`` (plus the demoted VLM ``STOP`` vote, ADR-0006). Pure
    rotations (turn/look) and ``describe`` are *neutral* — they neither count
    nor reset — because scanning in place legitimately produces no novelty and
    must not be mistaken for a plateau. Resets to 0 on a novel translation.
    Stops when the count reaches ``window``.
    """

    novelty_delta: float
    window: int = 5
    reason: str = "coverage_plateau"
    _count: int = 0

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be positive")

    def reset(self) -> None:
        self._count = 0

    def update(self, ctx: WalkContext) -> None:
        if ctx.action is Action.STOP:
            self._count += 1  # demoted vote: counts but cannot stop alone
            return
        if ctx.action in _NON_TRANSLATING:
            return  # scanning in place is neutral, not a plateau
        self._count = self._count + 1 if ctx.novelty < self.novelty_delta else 0

    def should_stop(self) -> bool:
        return self._count >= self.window


@dataclass
class StuckGuard:
    """Walk-level policy: stops when the robot makes no net progress over a window.

    Tracks the bounding-box span of recent positions. If the robot hasn't
    displaced more than ``min_displacement_factor * step`` across the last
    ``window`` steps, it is considered stuck.
    """

    step: float
    window: int = 8
    min_displacement_factor: float = 0.5
    reason: str = "stuck"
    _positions: list[np.ndarray] = field(default_factory=list)
    _triggered: bool = False

    def reset(self) -> None:
        self._positions.clear()
        self._triggered = False

    def update(self, ctx: WalkContext) -> None:
        if ctx.action in (Action.STOP, Action.DESCRIBE, Action.MARK) or ctx.degenerate:
            return
        self._positions.append(ctx.pose.position.copy())
        if len(self._positions) > self.window:
            self._positions.pop(0)
        if len(self._positions) >= self.window:
            arr = np.array(self._positions)
            span = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))
            if span < self.step * self.min_displacement_factor:
                self._triggered = True

    def should_stop(self) -> bool:
        return self._triggered


def any_walk_stop(policies: list[StopPolicy]) -> bool:
    """OR-composition for walk-level policies (excluding step counting)."""
    return any(p.should_stop() for p in policies)


def walk_stop_reason(policies: list[StopPolicy]) -> str | None:
    """The ``reason`` of the first firing walk policy, or ``None`` if none fired."""
    for p in policies:
        if p.should_stop():
            return p.reason
    return None


@dataclass
class SessionContext:
    """Snapshot handed to session-level stop policies after each walk."""

    state: CoverageState
    walks_completed: int
    total_seeds: int
    last_batch_coverage_gain: float = 0.0


@runtime_checkable
class SessionStopPolicy(Protocol):
    reason: str

    def should_stop(self, ctx: SessionContext) -> bool: ...


@dataclass
class PoseBudget:
    """Stop when the deliverable size is reached."""

    max_poses: int = 200
    reason: str = "pose_budget"

    def __post_init__(self) -> None:
        if self.max_poses <= 0:
            raise ValueError("max_poses must be positive")

    def should_stop(self, ctx: SessionContext) -> bool:
        return len(ctx.state) >= self.max_poses


@dataclass
class CoverageTarget:
    """Stop when floor coverage reaches ``tau`` (primary session terminator)."""

    radius: float
    tau: float = 0.8
    grid_cells: int = 64
    reason: str = "coverage_target"

    def __post_init__(self) -> None:
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")

    def should_stop(self, ctx: SessionContext) -> bool:
        return floor_coverage(ctx.state, radius=self.radius, grid_cells=self.grid_cells) >= self.tau


@dataclass
class QualityTarget:
    """Stop when quality-weighted floor coverage reaches ``tau``.

    Like CoverageTarget but only counts observations with render alpha >=
    q_min, so the session ends only when the scene is both visited *and*
    well-reconstructed.
    """

    radius: float
    tau: float = 0.8
    q_min: float = 0.5
    grid_cells: int = 64
    reason: str = "quality_target"

    def __post_init__(self) -> None:
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")

    def should_stop(self, ctx: SessionContext) -> bool:
        from gaussian_robot.metrics.coverage import quality_floor_coverage  # noqa: PLC0415

        return (
            quality_floor_coverage(
                ctx.state,
                radius=self.radius,
                q_min=self.q_min,
                grid_cells=self.grid_cells,
            )
            >= self.tau
        )


@dataclass
class SeedExhaustion:
    """Stop when all seeds have been launched."""

    reason: str = "seeds_exhausted"

    def should_stop(self, ctx: SessionContext) -> bool:
        return ctx.walks_completed >= ctx.total_seeds


@dataclass
class DiminishingReturns:
    """Stop when the last seed batch added < ``epsilon`` coverage."""

    epsilon: float = 0.005
    reason: str = "diminishing_returns"

    def __post_init__(self) -> None:
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")

    def should_stop(self, ctx: SessionContext) -> bool:
        return ctx.walks_completed > 0 and ctx.last_batch_coverage_gain < self.epsilon


def any_session_stop(policies: list[SessionStopPolicy], ctx: SessionContext) -> bool:
    """OR-composition for session-level policies."""
    return any(p.should_stop(ctx) for p in policies)


def session_stop_reason(policies: list[SessionStopPolicy], ctx: SessionContext) -> str | None:
    """The ``reason`` of the first firing session policy, or ``None`` if none fired."""
    for p in policies:
        if p.should_stop(ctx):
            return p.reason
    return None
