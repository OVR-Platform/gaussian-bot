"""Robot state.

The :class:`Robot` is a thin, framework-agnostic holder of the agent's pose
inside the scene plus a small, safe motion API. Keeping it separate from the
planner means we can drive it from a VLM policy, a keyboard teleop loop, or a
classic path planner without changing the state model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose
from gaussian_robot.splat.scene import SplatScene

# A reasonable default for rendered views fed to a VLM. VLMs are typically
# trained on ~square-ish, moderately wide images; revisit as needed.
_DEFAULT_INTRINSICS = CameraIntrinsics(
    fx=400.0, fy=400.0, cx=256.0, cy=256.0, width=512, height=512
)


@dataclass
class Robot:
    """A robot grounded in a specific scene.

    Attributes
    ----------
    scene:
        The scene the robot lives in; used to clamp poses to the navigable AABB.
    pose:
        Current pose of the robot's camera in world space.
    intrinsics:
        Intrinsics of the robot's onboard "camera". Used when rendering views.
    """

    scene: SplatScene
    pose: Pose
    intrinsics: CameraIntrinsics = _DEFAULT_INTRINSICS

    def camera(self) -> Camera:
        """Return the full camera (current pose + intrinsics)."""
        return Camera(pose=self.pose, intrinsics=self.intrinsics)

    def clamp_position(self, position: np.ndarray) -> np.ndarray:
        """Clip ``position`` to the scene AABB and return it."""
        return np.clip(
            np.asarray(position, dtype=np.float64), self.scene.bounds.min, self.scene.bounds.max
        )

    def move_to(self, position: np.ndarray, *, clamp: bool = True) -> Pose:
        """Translate the robot to ``position`` (rotation unchanged).

        If ``clamp`` is set and ``position`` leaves the scene AABB, it is
        projected back onto the bounds. Returns the new pose.
        """
        pos = np.asarray(position, dtype=np.float64)
        if pos.shape != (3,):
            raise ValueError(f"position must be (3,), got {pos.shape}")
        if clamp:
            pos = self.clamp_position(pos)
        self.pose = Pose(position=pos, rotation=self.pose.rotation)
        return self.pose

    def move(self, pose: Pose, *, clamp: bool = True) -> Pose:
        """Adopt ``pose`` (full position + rotation).

        Used by the controller executor: rotation is taken from ``pose``,
        position is optionally clamped to the scene AABB. Returns the new pose.
        """
        position = pose.position
        if clamp:
            position = self.clamp_position(position)
        self.pose = Pose(position=position, rotation=pose.rotation)
        return self.pose
