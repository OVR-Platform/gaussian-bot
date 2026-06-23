"""Tests for loading capture poses (cameras.json / COLMAP) used as seed sources."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from gaussian_robot.render.camera import Pose
from gaussian_robot.session import look_at
from gaussian_robot.splat.capture_poses import (
    _qvec_to_rotmat,
    discover_capture_poses,
    infer_up_axis,
    load_capture_poses,
    parse_3dgs_cameras_json,
    parse_colmap_images_bin,
    parse_colmap_images_txt,
)


def _write_cameras_json(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps(entries))


def test_cameras_json_rotation_is_transposed_to_world_to_camera(tmp_path: Path) -> None:
    # A camera looking down +Z in world: camera->world rotation is identity, so
    # world->camera (Pose.rotation) is also identity here.
    r_c2w = np.eye(3).tolist()
    entry = {"id": 0, "img_name": "a", "position": [1.0, 2.0, 3.0], "rotation": r_c2w}
    p = tmp_path / "cameras.json"
    _write_cameras_json(p, [entry])

    poses = parse_3dgs_cameras_json(p)
    assert len(poses) == 1
    np.testing.assert_allclose(poses[0].position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(poses[0].rotation, np.eye(3))


def test_cameras_json_transpose_matters_for_nontrivial_rotation(tmp_path: Path) -> None:
    # 90deg about world Y as a camera->world rotation; Pose.rotation must be its transpose.
    theta = np.pi / 2
    r_c2w = np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    entry = {"position": [0.0, 0.0, 0.0], "rotation": r_c2w.tolist()}
    p = tmp_path / "cameras.json"
    _write_cameras_json(p, [entry])

    poses = parse_3dgs_cameras_json(p)
    np.testing.assert_allclose(poses[0].rotation, r_c2w.T)


def test_colmap_position_is_camera_centre(tmp_path: Path) -> None:
    # Identity rotation, tvec t => camera centre = -R^T t = -t.
    qvec = [1.0, 0.0, 0.0, 0.0]  # identity quaternion
    tvec = [2.0, -3.0, 4.0]
    img_bin = tmp_path / "images.bin"
    with open(img_bin, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<I", 1))  # image_id
        f.write(struct.pack("<4d", *qvec))
        f.write(struct.pack("<3d", *tvec))
        f.write(struct.pack("<I", 1))  # camera_id
        f.write(b"img.png\x00")
        f.write(struct.pack("<Q", 0))  # num_points2D

    poses = parse_colmap_images_bin(img_bin)
    assert len(poses) == 1
    np.testing.assert_allclose(poses[0].rotation, _qvec_to_rotmat(np.array(qvec)))
    np.testing.assert_allclose(poses[0].position, [-2.0, 3.0, -4.0])


def test_colmap_bin_and_txt_agree(tmp_path: Path) -> None:
    qvec = [0.9238795, 0.0, 0.3826834, 0.0]  # 45deg about Y
    tvec = [1.0, 2.0, 3.0]
    img_bin = tmp_path / "images.bin"
    with open(img_bin, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<I", 7))
        f.write(struct.pack("<4d", *qvec))
        f.write(struct.pack("<3d", *tvec))
        f.write(struct.pack("<I", 2))
        f.write(b"x.jpg\x00")
        f.write(struct.pack("<Q", 0))
    img_txt = tmp_path / "images.txt"
    img_txt.write_text(
        "# header\n"
        f"7 {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} 2 x.jpg\n"
        "1.0 2.0 -1\n"
    )

    pb = parse_colmap_images_bin(img_bin)[0]
    pt = parse_colmap_images_txt(img_txt)[0]
    np.testing.assert_allclose(pb.position, pt.position, atol=1e-9)
    np.testing.assert_allclose(pb.rotation, pt.rotation, atol=1e-9)


def test_load_capture_poses_prefers_cameras_json_in_dir(tmp_path: Path) -> None:
    _write_cameras_json(
        tmp_path / "cameras.json",
        [{"position": [0.0, 0.0, 0.0], "rotation": np.eye(3).tolist()}],
    )
    poses = load_capture_poses(tmp_path)
    assert len(poses) == 1


def test_discover_capture_poses_walks_up_from_ply(tmp_path: Path) -> None:
    # Mimic 3DGS layout: output/cameras.json + output/point_cloud/iter/point_cloud.ply
    output = tmp_path / "output"
    iter_dir = output / "point_cloud" / "iteration_30000"
    iter_dir.mkdir(parents=True)
    _write_cameras_json(
        output / "cameras.json",
        [{"position": [0.0, 0.0, 0.0], "rotation": np.eye(3).tolist()}],
    )
    ply = iter_dir / "point_cloud.ply"
    ply.write_bytes(b"")

    found = discover_capture_poses(ply)
    assert found is not None
    assert found.name == "cameras.json"


def test_discover_returns_none_when_absent(tmp_path: Path) -> None:
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"")
    assert discover_capture_poses(ply) is None


def test_infer_up_axis_detects_negative_y() -> None:
    # Cameras whose world-up is -Y (look_at with up=-y) must infer "-y".
    poses = [
        Pose(position=np.zeros(3), rotation=look_at(np.zeros(3), t, "-y"))
        for t in (np.array([0.0, 0, 1]), np.array([1.0, 0, 0]), np.array([0.0, 0, -1]))
    ]
    assert infer_up_axis(poses) == "-y"


def test_infer_up_axis_detects_positive_y() -> None:
    poses = [Pose(position=np.zeros(3), rotation=look_at(np.zeros(3), np.array([1.0, 0, 0]), "y"))]
    assert infer_up_axis(poses) == "y"


def test_infer_up_axis_none_when_empty() -> None:
    assert infer_up_axis([]) is None
