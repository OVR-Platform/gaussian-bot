"""Scene representation.

A :class:`SplatScene` is a lightweight handle to a reconstruction. It keeps the
filesystem path plus geometric bounds the navigator needs to stay inside the
valid region of the splat. The actual gaussian data may live on GPU inside a
renderer — we don't duplicate it here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gaussian_robot.render.camera import parse_up_axis


@dataclass(frozen=True)
class SceneBounds:
    """Axis-aligned bounding box of the reconstructed region (world space)."""

    min: np.ndarray  # (3,) float64
    max: np.ndarray  # (3,) float64

    def __post_init__(self) -> None:
        if self.min.shape != (3,) or self.max.shape != (3,):
            raise ValueError("min and max must be (3,) arrays")
        if np.any(self.min > self.max):
            raise ValueError("min must be <= max component-wise")

    def contains(self, point: np.ndarray) -> bool:
        """True if ``point`` is inside the AABB (inclusive)."""
        return bool(np.all(point >= self.min) and np.all(point <= self.max))

    @property
    def center(self) -> np.ndarray:
        center: np.ndarray = (self.min + self.max) / 2.0
        return center


@dataclass(frozen=True)
class SplatScene:
    """A loaded (or to-be-loaded) Gaussian Splat reconstruction.

    Attributes
    ----------
    path:
        Filesystem location of the reconstruction (``.ply`` / ``.splat`` / dir).
    bounds:
        AABB of the navigable region. Used by the planner to clip poses.
    up_axis:
        Which world axis points up. ``"y"`` by default; revisit once a renderer
        / training pipeline is chosen (some use ``+Z`` up).
    """

    path: Path
    bounds: SceneBounds
    up_axis: str = "y"

    def __post_init__(self) -> None:
        parse_up_axis(self.up_axis)  # raises for anything but (optionally signed) x/y/z
