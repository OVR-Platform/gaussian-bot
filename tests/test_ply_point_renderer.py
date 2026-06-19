"""Tests for the lightweight PLY point renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gaussian_robot.backends.ply_point import PLYPointRenderer, load_ply_point_cloud
from gaussian_robot.config import RunConfig
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose
from gaussian_robot.session import build_session


def _write_ascii_ply(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0 0 3 255 0 0",
                "0.2 0 4 0 255 0",
                "-0.2 0 5 0 0 255",
            ]
        )
        + "\n"
    )


def test_load_ascii_ply_points_and_colours(tmp_path: Path) -> None:
    ply = tmp_path / "scene.ply"
    _write_ascii_ply(ply)

    cloud = load_ply_point_cloud(ply)

    assert cloud.points.shape == (3, 3)
    assert cloud.colors.tolist()[0] == [255, 0, 0]
    assert np.allclose(cloud.bounds.min, [-0.2, 0.0, 3.0])
    assert np.allclose(cloud.bounds.max, [0.2, 0.0, 5.0])


def test_ply_point_renderer_projects_visible_points(tmp_path: Path) -> None:
    ply = tmp_path / "scene.ply"
    _write_ascii_ply(ply)
    renderer = PLYPointRenderer.from_path(ply)
    camera = Camera(
        pose=Pose(),
        intrinsics=CameraIntrinsics(fx=20, fy=20, cx=16, cy=16, width=32, height=32),
    )

    result = renderer.render(camera)

    assert result.rgb.shape == (32, 32, 3)
    assert result.depth is not None
    assert result.depth.shape == (32, 32)
    assert bool(np.isfinite(result.depth).any())
    assert bool((result.rgb[..., 0] == 255).any())


def test_build_session_uses_ply_renderer_when_path_is_set(tmp_path: Path) -> None:
    ply = tmp_path / "scene.ply"
    _write_ascii_ply(ply)

    explorer, _seeds, coverage = build_session(RunConfig(ply_path=str(ply)))

    assert isinstance(explorer.renderer, PLYPointRenderer)
    assert np.allclose(coverage.bounds_min, [-0.2, 0.0, 3.0])
    assert np.allclose(coverage.bounds_max, [0.2, 0.0, 5.0])
