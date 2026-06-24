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
    MARK = "mark"
    GRAB = "grab"  # task mode: simulated pick-up of the target at the current pose (no motion)
    DROP = "drop"  # task mode: simulated release of the carried target (no motion)
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


def capped_forward_step(
    step: float, clearance: float | None, *, margin_factor: float = 0.3
) -> float:
    """Shorten a forward ``step`` so it stops short of an obstacle at ``clearance``.

    ``clearance`` is the free distance ahead in scene units (e.g. the median
    render depth in the central band). The step is clamped so the camera halts
    ``margin_factor * step`` before the obstacle, never driving into geometry.
    Returns the full ``step`` when ``clearance`` is unknown.
    """
    if clearance is None:
        return step
    margin = margin_factor * step
    return float(np.clip(clearance - margin, 0.0, step))


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


def slerp_rotation(r0: np.ndarray, r1: np.ndarray, t: float) -> np.ndarray:
    """Geodesic interpolation between two world->camera rotations (SO(3) slerp).

    Uses the same world-frame composition as :func:`apply_action` (``R = Rw @ R0``):
    the relative world rotation ``r1 @ r0.T`` is reduced to axis-angle and applied by
    a fraction ``t``. Robust for the small rotations a walk/tween makes.
    """
    rel = r1 @ r0.T
    cos = float(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos))
    if theta < 1e-6:
        return r0.copy()
    axis = np.array(
        [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]], dtype=np.float64
    ) / (2.0 * np.sin(theta))
    out: np.ndarray = _rotation_about_axis(axis, t * theta) @ r0
    return out


def interpolate_pose(a: Pose, b: Pose, t: float) -> Pose:
    """Pose between ``a`` and ``b``: linear position, slerped orientation."""
    return Pose(
        position=a.position * (1 - t) + b.position * t,
        rotation=slerp_rotation(a.rotation, b.rotation, t),
    )


def apply_action(
    pose: Pose,
    action: Action,
    space: ActionSpace,
    up_axis: str = "y",
    *,
    clearance: float | None = None,
) -> Pose:
    """Return the pose resulting from applying ``action`` to ``pose``.

    ``STOP`` returns ``pose`` unchanged (its termination semantics live in
    :mod:`gaussian_robot.nav.stop`). Rotation changes are world-frame rotations
    composed as ``R_new = R_world @ R``.

    ``clearance`` (free distance ahead, scene units) caps a ``FORWARD`` step so
    the camera stops short of obstacles instead of penetrating geometry. It does
    not affect ``BACK`` (rear clearance is unknown).
    """
    if action in (Action.STOP, Action.DESCRIBE, Action.MARK, Action.GRAB, Action.DROP):
        # MARK records the current viewpoint; GRAB/DROP are simulated manipulation that
        # only toggles the carried-payload state. None of these move the robot.
        return pose

    if action in (Action.FORWARD, Action.BACK):
        heading = pose.heading(up_axis)
        sign = 1.0 if action is Action.FORWARD else -1.0
        step = (
            capped_forward_step(space.step, clearance) if action is Action.FORWARD else space.step
        )
        new_pos: np.ndarray = pose.position + sign * step * heading
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
