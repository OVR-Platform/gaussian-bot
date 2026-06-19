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
    total_seeds: int


@dataclass(frozen=True)
class StepEvent:
    """Emitted after each step of a walk."""

    seed_id: str
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


@dataclass(frozen=True)
class SessionEndEvent:
    """Emitted once when the session ends."""

    reason: str
    total_steps: int
    total_poses: int


SessionEvent = SessionStartEvent | StepEvent | SessionEndEvent
EventSink = Callable[[SessionEvent], None]
