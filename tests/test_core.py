"""Tests for shared value types and the protocol seams (ADR-0001, ADR-0002)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gaussian_robot.render.base import Renderer, RenderResult
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose, axis_index, up_vector
from gaussian_robot.session import look_at
from gaussian_robot.splat.scene import SceneBounds, SplatScene
from gaussian_robot.vlm.client import VLMClient


def _scene() -> SplatScene:
    bounds = SceneBounds(min=np.zeros(3), max=np.ones(3) * 10.0)
    return SplatScene(path=Path(__file__), bounds=bounds)


def test_pose_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        Pose(position=np.zeros(4))
    with pytest.raises(ValueError):
        Pose(rotation=np.eye(2))


def test_intrinsics_reject_nonpositive() -> None:
    with pytest.raises(ValueError):
        CameraIntrinsics(fx=0, fy=1, cx=1, cy=1, width=10, height=10)


def test_up_vector_validation() -> None:
    assert np.allclose(up_vector("y"), [0, 1, 0])
    with pytest.raises(ValueError):
        up_vector("w")


def test_up_vector_signed_axes() -> None:
    assert np.allclose(up_vector("-y"), [0, -1, 0])
    assert np.allclose(up_vector("+z"), [0, 0, 1])
    assert np.allclose(up_vector("-x"), [-1, 0, 0])
    # Floor-plane axis index ignores sign: -y and y share the XZ floor plane.
    assert axis_index("-y") == axis_index("y") == 1


def test_look_at_respects_up_sign() -> None:
    # With -y up, the camera's "down" row (OpenCV +Y) must point toward +Y world.
    rot = look_at(np.zeros(3), np.array([0.0, 0.0, 1.0]), "-y")
    np.testing.assert_allclose(rot[1, :], [0, 1, 0], atol=1e-9)
    rot_plus = look_at(np.zeros(3), np.array([0.0, 0.0, 1.0]), "y")
    np.testing.assert_allclose(rot_plus[1, :], [0, -1, 0], atol=1e-9)


def test_look_at_is_a_proper_rotation_not_a_mirror() -> None:
    # det must be +1: a left-handed (det -1) basis is a reflection, which renders mirrored.
    for up in ("-y", "y", "z", "-x"):
        rot = look_at(np.zeros(3), np.array([1.0, 0.3, 0.7]), up)
        assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-6), up
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-6)  # orthonormal
    # -y up, looking +z: camera right (row 0) is +x (not -x, which was the mirror bug).
    rot = look_at(np.zeros(3), np.array([0.0, 0.0, 1.0]), "-y")
    np.testing.assert_allclose(rot[0, :], [1, 0, 0], atol=1e-9)


def test_forward_is_third_row_unit() -> None:
    pose = Pose(rotation=np.eye(3))
    assert np.allclose(pose.forward(), [0, 0, 1])
    assert np.allclose(pose.right(), [1, 0, 0])


def test_heading_projects_out_up_component() -> None:
    forward = np.array([1.0, 1.0, 1.0])
    forward /= np.linalg.norm(forward)
    rot = np.eye(3)
    rot[2, :] = forward
    pose = Pose(rotation=rot)
    heading = pose.heading("y")
    assert abs(heading[1]) < 1e-9
    assert np.allclose(np.linalg.norm(heading), 1.0)


def test_render_result_rejects_bad_rgb_shape() -> None:
    intr = CameraIntrinsics(fx=1, fy=1, cx=1, cy=1, width=4, height=4)
    cam = Camera(pose=Pose(), intrinsics=intr)
    with pytest.raises(ValueError):
        RenderResult(rgb=np.zeros((4, 4), dtype=np.uint8), camera=cam)


def test_scene_bounds_contains_and_center() -> None:
    b = SceneBounds(min=np.zeros(3), max=np.array([2.0, 2.0, 2.0]))
    assert b.contains(np.ones(3))
    assert not b.contains(np.array([2.5, 1.0, 1.0]))
    assert np.allclose(b.center, 1.0)


def test_scene_bounds_rejects_inverted() -> None:
    with pytest.raises(ValueError):
        SceneBounds(min=np.array([1.0, 0.0, 0.0]), max=np.array([0.0, 0.0, 0.0]))


class _DummyRenderer:
    """Renderer that returns a constant image; used only for protocol checks."""

    def render(self, camera: Camera) -> RenderResult:
        h, w = camera.intrinsics.height, camera.intrinsics.width
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        return RenderResult(rgb=rgb, camera=camera)


def test_renderer_protocol_is_runtime_checkable() -> None:
    assert isinstance(_DummyRenderer(), Renderer)


def test_vlmclient_protocol_is_runtime_checkable() -> None:
    class _FakeVLM:
        def reset(self) -> None: ...
        def act(self, observation: object) -> object: ...
        def describe(self, observation: object) -> str:
            return ""

    assert isinstance(_FakeVLM(), VLMClient)
