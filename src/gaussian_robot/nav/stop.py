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

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from gaussian_robot.metrics.coverage import CoverageState, floor_coverage
from gaussian_robot.nav.action import Action
from gaussian_robot.render.camera import Pose


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

    def reset(self) -> None: ...

    def update(self, ctx: WalkContext) -> None: ...

    def should_stop(self) -> bool: ...


@dataclass
class StepBudget:
    """Hard cap on steps per walk (safety net)."""

    max_steps: int = 40

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

    def reset(self) -> None: ...

    def update(self, ctx: WalkContext) -> None: ...

    def should_stop(self) -> bool:
        return False  # checked via the running step count by the explorer


@dataclass
class BoundsGuard:
    """Stops when a render is degenerate (out of bounds / empty)."""

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

    Counts consecutive steps that are "unproductive": either the current pose's
    novelty is below ``novelty_delta`` **or** the VLM emitted ``STOP`` (the
    demoted vote, ADR-0006). Resets to 0 on a novel step. Stops when the count
    reaches ``window``.
    """

    novelty_delta: float
    window: int = 5
    _count: int = 0

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be positive")

    def reset(self) -> None:
        self._count = 0

    def update(self, ctx: WalkContext) -> None:
        unproductive = ctx.novelty < self.novelty_delta or ctx.action is Action.STOP
        self._count = self._count + 1 if unproductive else 0

    def should_stop(self) -> bool:
        return self._count >= self.window


def any_walk_stop(policies: list[StopPolicy]) -> bool:
    """OR-composition for walk-level policies (excluding step counting)."""
    return any(p.should_stop() for p in policies)


@dataclass
class SessionContext:
    """Snapshot handed to session-level stop policies after each walk."""

    state: CoverageState
    walks_completed: int
    total_seeds: int
    last_batch_coverage_gain: float = 0.0


@runtime_checkable
class SessionStopPolicy(Protocol):
    def should_stop(self, ctx: SessionContext) -> bool: ...


@dataclass
class PoseBudget:
    """Stop when the deliverable size is reached."""

    max_poses: int = 200

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

    def __post_init__(self) -> None:
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")

    def should_stop(self, ctx: SessionContext) -> bool:
        return floor_coverage(ctx.state, radius=self.radius, grid_cells=self.grid_cells) >= self.tau


@dataclass
class SeedExhaustion:
    """Stop when all seeds have been launched."""

    def should_stop(self, ctx: SessionContext) -> bool:
        return ctx.walks_completed >= ctx.total_seeds


@dataclass
class DiminishingReturns:
    """Stop when the last seed batch added < ``epsilon`` coverage."""

    epsilon: float = 0.005

    def __post_init__(self) -> None:
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")

    def should_stop(self, ctx: SessionContext) -> bool:
        return ctx.walks_completed > 0 and ctx.last_batch_coverage_gain < self.epsilon


def any_session_stop(policies: list[SessionStopPolicy], ctx: SessionContext) -> bool:
    """OR-composition for session-level policies."""
    return any(p.should_stop(ctx) for p in policies)
