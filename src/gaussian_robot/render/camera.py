"""Camera geometry: poses and intrinsics.

Math types are plain dataclasses backed by numpy arrays. We keep them framework
-agnostic so they work regardless of which renderer/VLM backend is chosen.

Conventions (ADR-0002)
----------------------
- Right-handed world frame, **+Y up** (floor plane = XZ).
- **OpenCV** camera axes: **+Z = forward** (into the scene), +X = right,
  +Y = down.
- ``Pose.rotation`` is the **world->camera** rotation matrix. Therefore camera
  forward in world = ``rotation[2, :]`` (third row), right = ``rotation[0, :]``,
  and world-up = ``-rotation[1, :]``.
- Intrinsics follow the pinhole model: ``fx, fy, cx, cy`` in pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_IDENTITY_ROT = np.eye(3, dtype=np.float64)
_ORIGIN = np.zeros(3, dtype=np.float64)

_UP_VECTORS: dict[str, np.ndarray] = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def up_vector(up_axis: str) -> np.ndarray:
    """Unit vector along the named world axis."""
    try:
        return _UP_VECTORS[up_axis.lower()].copy()
    except KeyError:
        raise ValueError(f"up_axis must be one of x/y/z, got {up_axis!r}") from None


@dataclass(frozen=True)
class Pose:
    """A 6-DoF pose: position in world space + world->camera rotation.

    Parameters
    ----------
    position:
        ``(3,)`` float64 array, camera centre in world coordinates.
    rotation:
        ``(3, 3)`` float64 row-major rotation matrix mapping world -> camera.
    """

    position: np.ndarray = field(default_factory=_ORIGIN.copy)
    rotation: np.ndarray = field(default_factory=_IDENTITY_ROT.copy)

    def __post_init__(self) -> None:
        if self.position.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {self.position.shape}")
        if self.rotation.shape != (3, 3):
            raise ValueError(f"rotation must have shape (3,3), got {self.rotation.shape}")

    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        """Transform ``(N, 3)`` world points into camera space."""
        transformed: np.ndarray = (points - self.position) @ self.rotation.T
        return transformed

    def forward(self) -> np.ndarray:
        """World-space viewing direction (unit), per ADR-0002 (camera +Z)."""
        fwd: np.ndarray = self.rotation[2, :].astype(np.float64)
        n = np.linalg.norm(fwd)
        if n < 1e-12:
            raise ValueError("rotation has degenerate forward direction")
        unit: np.ndarray = fwd / n
        return unit

    def right(self) -> np.ndarray:
        """World-space camera-right direction (unit)."""
        r: np.ndarray = self.rotation[0, :].astype(np.float64)
        n = np.linalg.norm(r)
        if n < 1e-12:
            raise ValueError("rotation has degenerate right direction")
        unit: np.ndarray = r / n
        return unit

    def heading(self, up_axis: str) -> np.ndarray:
        """Horizontal forward projected onto the floor plane (unit).

        If the camera looks straight along the up axis (no horizontal component),
        an arbitrary perpendicular horizontal direction is returned.
        """
        up = up_vector(up_axis)
        fwd = self.forward()
        horizontal: np.ndarray = fwd - up * float(fwd @ up)
        n = np.linalg.norm(horizontal)
        if n < 1e-9:
            # Camera looks straight up/down: pick any horizontal axis orthogonal to up.
            fwd_world = self.forward()
            candidate = np.array([1.0, 0.0, 0.0])
            if abs(float(fwd_world @ up)) > 0.9:
                candidate = np.array([0.0, 0.0, 1.0])
            horizontal = candidate - up * float(candidate @ up)
            n = np.linalg.norm(horizontal)
        out: np.ndarray = horizontal / n
        return out


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics in pixel units."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if min(self.fx, self.fy) <= 0:
            raise ValueError("focal lengths must be positive")

    @property
    def k_matrix(self) -> np.ndarray:
        """The 3x3 intrinsics matrix K."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Camera:
    """A fully specified camera: extrinsics (pose) + intrinsics."""

    pose: Pose
    intrinsics: CameraIntrinsics
