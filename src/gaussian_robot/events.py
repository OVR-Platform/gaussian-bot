"""Live exploration events emitted by :class:`~gaussian_robot.nav.explorer.Explorer`.

These are core value types (no UI deps) so the dashboard can subscribe without
the core depending on the UI. A :type:`EventSink` is a plain callable that
receives one :data:`SessionEvent` at a time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from gaussian_robot.nav.action import Action
from gaussian_robot.render.camera import Pose
from gaussian_robot.vlm.client import Decision
from gaussian_robot.vlm.observation import Observation


@dataclass(frozen=True)
class SessionStartEvent:
    """Emitted once at the start of a session."""

    bounds_min: np.ndarray
    bounds_max: np.ndarray
    up_axis: str
    total_seeds: int  # seeds actually launched (== len(seed_floor))
    seed_floor: np.ndarray  # (S, 2) floor positions of the seeds walks start from
    seed_kinds: list[str]  # per-seed provenance: "capture" | "origin_fallback" | "density" | "grid"
    requested_seeds: int  # how many were asked for (may exceed total_seeds if some were rejected)


@dataclass(frozen=True)
class StepEvent:
    """Emitted after each step of a walk."""

    walk_id: str
    step: int
    budget: int
    observation: Observation
    decision: Decision
    action: Action
    pose: Pose
    novelty: float
    degenerate: bool
    coverage_floor: float
    coverage_pose_space: float
    sampled_floor: np.ndarray  # (N, 2)
    trail_floor: np.ndarray  # (M, 2)
    blocked: bool = False  # a FORWARD step halted short of an obstacle (no move committed)


@dataclass(frozen=True)
class WalkEndEvent:
    """Emitted when one walk ends, carrying *why* it stopped."""

    walk_id: str
    reason: str  # coverage_plateau | bounds | stuck | step_budget
    steps: int


@dataclass(frozen=True)
class SceneDescribeEvent:
    """Emitted when the VLM describes (or re-describes) the scene."""

    walk_id: str
    step: int
    description: str


@dataclass(frozen=True)
class SessionEndEvent:
    """Emitted once when the session ends."""

    reason: str
    total_steps: int
    total_poses: int


SessionEvent = SessionStartEvent | StepEvent | WalkEndEvent | SceneDescribeEvent | SessionEndEvent
EventSink = Callable[[SessionEvent], None]
