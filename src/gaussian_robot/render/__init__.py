"""Renderer: turn a camera pose into an image (and optional depth).

The :class:`Renderer` protocol is intentionally backend-agnostic so we can swap
between ``gsplat``, a web viewer, or a custom rasterizer without touching the
navigation code.
"""

from gaussian_robot.render.base import Renderer, RenderResult
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose

__all__ = [
    "Camera",
    "CameraIntrinsics",
    "Pose",
    "RenderResult",
    "Renderer",
]
