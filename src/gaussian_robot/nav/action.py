"""Actions and the egocentric controller executor (ADR-0003, ADR-0004).

The VLM emits one :class:`Action` verb per step; the executor applies a
**system-owned** magnitude (:class:`ActionSpace`) to the current :class:`Pose`.
Translation stays on the floor plane; pitch is view-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from gaussian_robot.render.camera import Pose, up_vector


class Action(StrEnum):
    """The discrete egocentric action vocabulary (ADR-0004)."""

    FORWARD = "forward"
    BACK = "back"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    LOOK_UP = "look_up"
    LOOK_DOWN = "look_down"
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    DESCRIBE = "describe"
    STOP = "stop"

    @classmethod
    def verbs(cls) -> list[str]:
        return [a.value for a in cls]


@dataclass(frozen=True)
class ActionSpace:
    """System-owned magnitudes applied to actions (ADR-0004).

    Attributes
    ----------
    step:
        Translation length per ``forward``/``back`` (scene units).
    delta_rot:
        Per-step rotation in **radians** for turn/pitch.
    """

    step: float
    delta_rot: float = np.deg2rad(30.0)

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError("step must be positive")
        if self.delta_rot <= 0:
            raise ValueError("delta_rot must be positive")

    @classmethod
    def from_bounds(
        cls, bounds_min: np.ndarray, bounds_max: np.ndarray, *, step_fraction: float = 0.03
    ) -> ActionSpace:
        """Build an action space whose ``step`` scales with the scene AABB."""
        diagonal = float(np.linalg.norm(bounds_max - bounds_min))
        if diagonal <= 0:
            raise ValueError("scene AABB has zero diagonal")
        return cls(step=diagonal * step_fraction)


def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit ``axis`` by ``angle`` (radians)."""
    axis = axis / np.linalg.norm(axis)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    t = 1.0 - c
    x, y, z = axis
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def apply_action(pose: Pose, action: Action, space: ActionSpace, up_axis: str = "y") -> Pose:
    """Return the pose resulting from applying ``action`` to ``pose``.

    ``STOP`` returns ``pose`` unchanged (its termination semantics live in
    :mod:`gaussian_robot.nav.stop`). Rotation changes are world-frame rotations
    composed as ``R_new = R_world @ R``.
    """
    if action in (Action.STOP, Action.DESCRIBE):
        return pose

    if action in (Action.FORWARD, Action.BACK):
        heading = pose.heading(up_axis)
        sign = 1.0 if action is Action.FORWARD else -1.0
        new_pos: np.ndarray = pose.position + sign * space.step * heading
        return Pose(position=new_pos, rotation=pose.rotation)

    if action in (Action.TURN_LEFT, Action.TURN_RIGHT):
        up = up_vector(up_axis)
        sign = 1.0 if action is Action.TURN_LEFT else -1.0
        r_world = _rotation_about_axis(up, sign * space.delta_rot)
        new_rot: np.ndarray = r_world @ pose.rotation
        return Pose(position=pose.position, rotation=new_rot)

    if action in (Action.MOVE_UP, Action.MOVE_DOWN):
        up = up_vector(up_axis)
        sign = 1.0 if action is Action.MOVE_UP else -1.0
        new_pos = pose.position + sign * space.step * up
        return Pose(position=new_pos, rotation=pose.rotation)

    # LOOK_UP / LOOK_DOWN: rotate about the camera's current right axis.
    right = pose.right()
    sign = 1.0 if action is Action.LOOK_UP else -1.0
    r_world = _rotation_about_axis(right, sign * space.delta_rot)
    new_rot = r_world @ pose.rotation
    return Pose(position=pose.position, rotation=new_rot)
