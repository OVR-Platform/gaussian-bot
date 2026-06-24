"""Seed selection prefers sharp, well-reconstructed capture views (ADR-0009)."""

from __future__ import annotations

import numpy as np

from gaussian_robot.nav.explorer import SeedPose
from gaussian_robot.render.base import RenderResult
from gaussian_robot.render.camera import Camera, Pose
from gaussian_robot.session import _sharpness, validate_seed_poses


def _flat(n: int = 16) -> np.ndarray:
    return np.full((n, n, 3), 128, dtype=np.uint8)


def _textured(n: int = 16) -> np.ndarray:
    img = np.zeros((n, n, 3), dtype=np.uint8)
    img[::2, ::2] = 255
    img[1::2, 1::2] = 255
    return img


class _SharpnessRenderer:
    """Blurry (flat) render for x<0, crisp (checkerboard) render for x>=0."""

    def render(self, camera: Camera) -> RenderResult:
        n = 16
        rgb = _flat(n) if camera.pose.position[0] < 0 else _textured(n)
        depth = np.full((n, n), 5.0, dtype=np.float32)
        alpha = np.ones((n, n), dtype=np.float32)
        return RenderResult(rgb=rgb, camera=camera, depth=depth, alpha=alpha)


def test_sharpness_separates_flat_from_textured() -> None:
    assert _sharpness(_textured()) > _sharpness(_flat())


def test_validate_seed_poses_skips_blurry_real_views() -> None:
    # Two blurry (x<0) and two sharp (x>=0) capture-like candidates. Non-strict
    # selection must drop the blurry ones below the sharpness floor, keeping kind.
    candidates = [
        SeedPose(pose=Pose(position=np.array([-1.0, 0.0, 0.0])), kind="capture"),
        SeedPose(pose=Pose(position=np.array([-2.0, 0.0, 0.0])), kind="capture"),
        SeedPose(pose=Pose(position=np.array([1.0, 0.0, 0.0])), kind="capture"),
        SeedPose(pose=Pose(position=np.array([2.0, 0.0, 0.0])), kind="capture"),
    ]
    seeds = validate_seed_poses(
        _SharpnessRenderer(), candidates, num_seeds=2, step=1.0, strict=False
    )
    assert len(seeds) == 2
    assert all(s.pose.position[0] >= 0 for s in seeds)  # only the sharp views seed
    assert all(s.kind == "capture" for s in seeds)
