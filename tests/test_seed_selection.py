"""Seed selection prefers sharp, well-reconstructed capture views (ADR-0009)."""

from __future__ import annotations

import types

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


class _CloudRenderer:
    """Crisp render everywhere, plus a density grid with one interior gap."""

    def __init__(self, grid: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> None:
        self.cloud = types.SimpleNamespace(density_grid=grid, density_bounds=(lo, hi))

    def render(self, camera: Camera) -> RenderResult:
        n = 16
        return RenderResult(
            rgb=_textured(n),
            camera=camera,
            depth=np.full((n, n), 5.0, dtype=np.float32),
            alpha=np.ones((n, n), dtype=np.float32),
        )


def test_validate_seed_poses_skips_blurry_real_views() -> None:
    # Two blurry (x<0) and two sharp (x>=0) capture-like candidates, spaced well apart
    # (> the 8*step dedup window). Non-strict selection drops the blurry ones.
    candidates = [
        SeedPose(pose=Pose(position=np.array([-10.0, 0.0, 0.0])), kind="capture"),
        SeedPose(pose=Pose(position=np.array([-20.0, 0.0, 0.0])), kind="capture"),
        SeedPose(pose=Pose(position=np.array([10.0, 0.0, 0.0])), kind="capture"),
        SeedPose(pose=Pose(position=np.array([20.0, 0.0, 0.0])), kind="capture"),
    ]
    seeds = validate_seed_poses(
        _SharpnessRenderer(), candidates, num_seeds=2, step=1.0, strict=False
    )
    assert len(seeds) == 2
    assert all(s.pose.position[0] >= 0 for s in seeds)  # only the sharp views seed
    assert all(s.kind == "capture" for s in seeds)


def test_validate_seed_poses_honours_num_seeds_despite_tight_spacing() -> None:
    # four sharp candidates all within the 8*step spacing window; num_seeds must still be met
    candidates = [
        SeedPose(pose=Pose(position=np.array([float(x), 0.0, 0.0])), kind="capture")
        for x in range(4)
    ]
    seeds = validate_seed_poses(
        _SharpnessRenderer(), candidates, num_seeds=3, step=1.0, strict=False
    )
    assert len(seeds) == 3  # spacing-held candidates top up to the requested count


def test_validate_seed_poses_prefers_seeds_near_a_frontier() -> None:
    grid = np.full((8, 8), 0.6)
    grid[2, 2] = 0.05  # one interior gap -> frontier at world ~ (3.125, 3.125)
    renderer = _CloudRenderer(grid, np.zeros(3), np.array([10.0, 10.0, 10.0]))
    far = SeedPose(pose=Pose(position=np.array([9.0, 0.0, 9.0])), kind="capture")
    near = SeedPose(pose=Pose(position=np.array([3.0, 0.0, 3.0])), kind="capture")
    seeds = validate_seed_poses(renderer, [far, near], num_seeds=1, step=1.0, strict=False)
    assert len(seeds) == 1
    assert np.allclose(seeds[0].pose.position, [3.0, 0.0, 3.0])  # the gap-near seed wins
