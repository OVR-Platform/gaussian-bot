"""Renderer protocol and output type.

Anything that can produce a rendered view from a :class:`~gaussian_robot.render.camera.Camera`
inside a scene implements :class:`Renderer`. This is the seam where we will plug
in ``gsplat``, a web viewer bridge, or a custom rasterizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from gaussian_robot.render.camera import Camera


@dataclass(frozen=True)
class RenderResult:
    """A rendered view from inside the scene.

    Attributes
    ----------
    rgb:
        ``(H, W, 3)`` uint8 image, sRGB.
    depth:
        Optional ``(H, W)`` float32 depth map in metres, if the backend
        provides it (useful for obstacle avoidance / VLM grounding).
    camera:
        The camera that produced this view (kept for provenance).
    """

    rgb: np.ndarray
    camera: Camera
    depth: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(f"rgb must be (H,W,3), got {self.rgb.shape}")
        if self.depth is not None and self.depth.shape != self.rgb.shape[:2]:
            raise ValueError(f"depth {self.depth.shape} must match rgb hw {self.rgb.shape[:2]}")


@runtime_checkable
class Renderer(Protocol):
    """Renders a camera view from the currently-loaded scene.

    Implementations may be stateful (holding a splat on GPU) and need not be
    thread-safe.
    """

    def render(self, camera: Camera) -> RenderResult:
        """Render the scene as seen by ``camera``.

        Raises
        ------
        RuntimeError
            If no scene is loaded or rendering fails.
        """
        ...
