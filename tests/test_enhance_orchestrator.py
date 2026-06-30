"""Gap-FILL orchestration: pose synthesis, filler wiring, and the distill loop.

The CPU-only tests cover the deterministic plumbing (gap-pose synthesis, up-axis inference,
50/50 interleave, filler construction) without touching the GPU. The GPU test drives
``fill_gaps_scene`` end-to-end on a tiny synthetic scene written to a real COLMAP model + PLY,
using an injected stub :class:`~gaussian_robot.enhance.protocols.ViewFiller` so no diffusion
weights are downloaded — it verifies a NEW ply is written, the input is untouched, gap poses
are framed, and the held-out (anchor) views do not regress.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from gaussian_robot.enhance import orchestrator
from gaussian_robot.enhance.capture_images import RealView
from gaussian_robot.enhance.fillers import DiffusionFiller, GeometricFiller
from gaussian_robot.enhance.orchestrator import (
    _fill_gap_views,
    _infer_up_axis,
    _interleave,
    _select_nearest_gaps,
    build_filler,
    synthesize_gap_poses,
    synthesize_marked_pairs,
    synthesize_near_view_poses,
)
from gaussian_robot.enhance.protocols import SupervisionView
from gaussian_robot.render.base import RenderResult
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose
from gaussian_robot.session import look_at


def test_synthesize_gap_poses_frames_each_gap() -> None:
    # Cameras spread in the XZ plane at y=0; gaps sit off to one side.
    cam_pos = np.array(
        [[x, 0.0, z] for x in (-1.0, 0.0, 1.0) for z in (-1.0, 0.0, 1.0)], dtype=np.float64
    )
    gaps = np.array([[2.0, 0.0, 2.0], [-2.0, 0.0, -2.0], [0.0, 0.0, 3.0]], dtype=np.float64)
    intr = CameraIntrinsics(fx=512.0, fy=512.0, cx=256.0, cy=256.0, width=512, height=512)

    cams = synthesize_gap_poses(gaps, cam_pos, intr, up_axis="y", n_poses=3)

    assert len(cams) == 3
    for cam, gap in zip(cams, gaps, strict=True):
        # The camera's forward (+Z row of world->camera) should point toward the gap.
        to_gap = gap - cam.pose.position
        to_gap = to_gap / np.linalg.norm(to_gap)
        fwd = cam.pose.forward()
        assert float(fwd @ to_gap) > 0.9  # looking at the gap
        # Placed between a real camera and the gap, not on top of the gap.
        assert np.linalg.norm(cam.pose.position - gap) > 1e-3
        assert cam.intrinsics == intr


def test_synthesize_gap_poses_empty_inputs() -> None:
    intr = CameraIntrinsics(fx=8.0, fy=8.0, cx=4.0, cy=4.0, width=8, height=8)
    empty = np.empty((0, 3), dtype=np.float64)
    cams = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    assert synthesize_gap_poses(empty, cams, intr, up_axis="y", n_poses=4) == []
    one_gap = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    assert synthesize_gap_poses(one_gap, empty, intr, up_axis="y", n_poses=4) == []


def test_synthesize_gap_poses_caps_at_available_gaps() -> None:
    intr = CameraIntrinsics(fx=8.0, fy=8.0, cx=4.0, cy=4.0, width=8, height=8)
    cam_pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    gaps = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    cams = synthesize_gap_poses(gaps, cam_pos, intr, up_axis="y", n_poses=10)
    assert len(cams) == 2  # cannot exceed the number of gaps


def _real_view(pos: np.ndarray, target: np.ndarray, intr: CameraIntrinsics, idx: int) -> RealView:
    rot = look_at(pos, target, "y")
    cam = Camera(pose=Pose(position=pos, rotation=rot), intrinsics=intr)
    return RealView(camera=cam, image_path=Path(f"img_{idx}.png"), name=f"img_{idx}", camera_id=1)


def test_synthesize_near_view_poses_stays_near_reference_and_frames_gap() -> None:
    intr = CameraIntrinsics(fx=512.0, fy=512.0, cx=256.0, cy=256.0, width=512, height=512)
    # A ring of training cameras around the origin, looking inward.
    cam_pos = np.array(
        [[x, 0.0, z] for x in (-1.0, 0.0, 1.0) for z in (-1.0, 0.0, 1.0)], dtype=np.float64
    )
    views = [_real_view(p, np.zeros(3), intr, i) for i, p in enumerate(cam_pos)]
    gaps = np.array([[2.0, 0.0, 2.0], [-2.0, 0.0, -2.0]], dtype=np.float64)

    frac = 0.15
    # Legacy "nearest" selection: this pins the dolly geometry + nearest-by-translation reference.
    pairs = synthesize_near_view_poses(
        views, gaps, intr, up_axis="y", n_poses=2, perturb_frac=frac, ref_select="nearest"
    )
    assert len(pairs) == 2
    for (cam, ref), gap in zip(pairs, gaps, strict=True):
        # The novel pose looks toward the gap.
        to_gap = gap - cam.pose.position
        to_gap = to_gap / np.linalg.norm(to_gap)
        assert float(cam.pose.forward() @ to_gap) > 0.9
        # It sits a perturb_frac dolly along the reference->gap ray (closer to the gap than the
        # reference, but well before it) — not most of the way like the old synthesize_gap_poses.
        ref_pos = ref.camera.pose.position
        assert np.linalg.norm(cam.pose.position - ref_pos) > 1e-6  # actually moved
        assert np.linalg.norm(cam.pose.position - gap) < np.linalg.norm(ref_pos - gap)  # toward gap
        expected = np.linalg.norm((gap - ref_pos) * frac)
        assert abs(np.linalg.norm(cam.pose.position - ref_pos) - expected) < 1e-6
        # The reference is the training camera nearest the gap.
        nearest = views[int(np.argmin(((cam_pos - gap) ** 2).sum(axis=1)))]
        assert ref.name == nearest.name


def test_reference_prefers_view_that_frames_the_gap() -> None:
    """The bug fix: a close-but-facing-away camera must lose to one that actually sees the gap.

    Camera A sits right next to the gap but looks the opposite way (gap behind it); camera B is
    farther but aimed straight at the gap. ``ref_select="visible"`` must pick B (Difix can only
    borrow appearance the reference contains); the legacy ``"nearest"`` picks the close-but-blind A.
    """
    intr = CameraIntrinsics(fx=256.0, fy=256.0, cx=256.0, cy=256.0, width=512, height=512)
    gap = np.array([[3.0, 0.0, 0.0]], dtype=np.float64)
    a = _real_view(np.array([2.5, 0.0, 0.0]), np.array([-10.0, 0.0, 0.0]), intr, 0)  # close, blind
    b = _real_view(np.array([0.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]), intr, 1)  # far, framing
    views = [a, b]

    vis = synthesize_near_view_poses(views, gap, intr, up_axis="y", n_poses=1, ref_select="visible")
    assert vis[0][1].name == b.name  # picked the framing camera
    near = synthesize_near_view_poses(views, gap, intr, up_axis="y", n_poses=1, ref_select="nearest")
    assert near[0][1].name == a.name  # legacy: nearest by translation, even though it's blind


def test_synthesize_marked_pairs_uses_mark_as_frame_and_aligned_reference() -> None:
    """Difix runs on the ROBOT's marked pose itself; reference = nearest + best-aligned real view."""
    intr = CameraIntrinsics(fx=256.0, fy=256.0, cx=256.0, cy=256.0, width=512, height=512)
    # Two real views near a mark: one looks the SAME way as the mark, one looks opposite.
    aligned = _real_view(np.array([0.1, 0.0, -1.0]), np.array([0.1, 0.0, 5.0]), intr, 0)  # +z
    opposed = _real_view(np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, -5.0]), intr, 1)  # -z
    mark = Pose(position=np.array([0.0, 0.0, 0.0]), rotation=look_at(
        np.zeros(3), np.array([0.0, 0.0, 5.0]), "y"))  # mark looks +z

    pairs = synthesize_marked_pairs([mark], [aligned, opposed], intr)

    assert len(pairs) == 1
    cam, ref = pairs[0]
    assert cam.pose is mark  # the frame to fix IS the mark, not a synthesized pose
    assert cam.intrinsics == intr
    assert ref.name == aligned.name  # reference looks the same direction, not the opposed one


def test_synthesize_marked_pairs_empty_inputs() -> None:
    intr = CameraIntrinsics(fx=8.0, fy=8.0, cx=4.0, cy=4.0, width=8, height=8)
    v = [_real_view(np.zeros(3), np.array([0.0, 0.0, 1.0]), intr, 0)]
    assert synthesize_marked_pairs([], v, intr) == []
    assert synthesize_marked_pairs([Pose()], [], intr) == []


def test_synthesize_near_view_poses_empty_inputs() -> None:
    intr = CameraIntrinsics(fx=8.0, fy=8.0, cx=4.0, cy=4.0, width=8, height=8)
    empty = np.empty((0, 3), dtype=np.float64)
    v = [_real_view(np.zeros(3), np.array([0, 0, 1.0]), intr, 0)]
    assert synthesize_near_view_poses(v, empty, intr, up_axis="y", n_poses=4) == []
    one_gap = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    assert synthesize_near_view_poses([], one_gap, intr, up_axis="y", n_poses=4) == []


def test_select_nearest_gaps_maps_robot_marks_onto_gaps() -> None:
    # Robot marks select the SUBSET of gaps nearest the vantage points it walked to.
    gaps = np.array(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float64
    )
    marks = np.array([[0.2, 0.0, 0.1], [9.6, 0.0, 0.0]], dtype=np.float64)  # near gap 0 and gap 2
    out = _select_nearest_gaps(gaps, marks)
    assert np.allclose(out, np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))


def test_select_nearest_gaps_dedupes_and_falls_back() -> None:
    gaps = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)
    # Two marks collapse to the same nearest gap -> one row (order-stable, deduped).
    dup = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=np.float64)
    assert _select_nearest_gaps(gaps, dup).shape == (1, 3)
    # No marks (robot found nothing) -> full gap set is returned unchanged.
    empty = np.empty((0, 3), dtype=np.float64)
    assert np.allclose(_select_nearest_gaps(gaps, empty), gaps)
    assert _select_nearest_gaps(empty, dup).shape == (0, 3)


def test_infer_up_axis_picks_low_spread_axis() -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal(0.0, 1.0, size=(50, 3))
    pts[:, 1] *= 0.01  # Y is the flat (up) axis
    pts[:, 1] += 5.0  # cameras sit above the floor
    axis = _infer_up_axis(pts)
    assert axis[-1] == "y"


def test_interleave_is_balanced() -> None:
    intr = CameraIntrinsics(fx=8.0, fy=8.0, cx=4.0, cy=4.0, width=8, height=8)
    cam = Camera(pose=Pose(), intrinsics=intr)
    img = np.zeros((8, 8, 3), dtype=np.float32)
    mask = np.ones((8, 8), dtype=np.float32)
    gap = [SupervisionView(camera=cam, target_rgb=img, mask=mask) for _ in range(2)]
    anchor = [SupervisionView(camera=cam, target_rgb=img, mask=None) for _ in range(3)]
    out = _interleave(gap, anchor)
    assert len(out) == 5
    assert out[0].mask is not None  # gap first
    assert out[1].mask is None  # anchor second
    # all five preserved
    assert sum(1 for v in out if v.mask is not None) == 2


def test_build_filler_dispatch() -> None:
    assert isinstance(build_filler("geometric"), GeometricFiller)
    assert isinstance(build_filler("identity"), GeometricFiller)
    assert isinstance(build_filler("difix"), DiffusionFiller)
    with pytest.raises(ValueError, match="unknown filler"):
        build_filler("nope")


def test_build_filler_dtype_selects_precision() -> None:
    # Default keeps NVIDIA's reference precision; fp16 is the opt-in VRAM-headroom lever.
    assert build_filler("difix")._dtype_name == "float32"
    assert build_filler("difix", dtype="float16")._dtype_name == "float16"


class _FakeRenderer:
    """Stand-in for GsplatRenderer so _fill_gap_views runs on CPU with no gsplat/GPU."""

    def __init__(self, cloud: object) -> None:
        self.cloud = cloud

    def render(self, camera: Camera) -> RenderResult:
        return RenderResult(
            rgb=np.zeros((8, 8, 3), np.uint8),
            camera=camera,
            alpha=np.ones((8, 8), np.float32),
        )


class _RecordingFiller:
    """Records the camera carried by the reference RenderResult it is handed."""

    def __init__(self) -> None:
        self.seen_ref_cameras: list[Camera] = []

    def fill(self, degraded: RenderResult, references: object) -> SupervisionView:
        self.seen_ref_cameras.append(references[0].camera)  # type: ignore[index]
        return SupervisionView(
            camera=degraded.camera,
            target_rgb=np.zeros((8, 8, 3), np.float32),
            mask=np.ones((8, 8), np.float32),
        )


def test_fill_gap_views_passes_true_reference_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard: the reference RenderResult must carry the REFERENCE pose, not the gap pose.

    A pose-aware filler reads ``references[0].camera`` to recover the ref->gap relative pose; the
    old code passed the degraded gap camera, collapsing that to identity. This pins the fix at the
    orchestrator seam without needing the GPU (renderer + image loader are stubbed).
    """
    monkeypatch.setattr(orchestrator, "GsplatRenderer", _FakeRenderer)
    monkeypatch.setattr(orchestrator, "load_image", lambda path, w, h: np.zeros((h, w, 3), np.float32))

    intr = CameraIntrinsics(fx=8.0, fy=8.0, cx=4.0, cy=4.0, width=8, height=8)
    ref_view = _real_view(np.array([1.0, 0.0, 0.0]), np.zeros(3), intr, 0)
    gap_cam = Camera(
        pose=Pose(position=np.array([0.5, 0.0, 0.5]), rotation=look_at(
            np.array([0.5, 0.0, 0.5]), np.zeros(3), "y"
        )),
        intrinsics=intr,
    )
    rec = _RecordingFiller()

    out = _fill_gap_views(
        cloud=None,  # consumed only by the stubbed renderer
        gap_pairs=[(gap_cam, ref_view)],
        ref_size=(8, 8),
        filler="difix",
        device="cpu",
        view_filler=rec,
    )

    assert len(out) == 1
    # The filler saw the TRUE reference camera (same object), not the degraded gap camera.
    assert rec.seen_ref_cameras[0] is ref_view.camera
    assert rec.seen_ref_cameras[0] is not gap_cam


# --------------------------------------------------------------------------------------------
# GPU end-to-end on a tiny synthetic COLMAP scene with an injected stub filler.
# --------------------------------------------------------------------------------------------

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")

if not torch.cuda.is_available():  # pragma: no cover - environment dependent
    pytest.skip("requires a CUDA GPU", allow_module_level=True)

from PIL import Image  # noqa: E402

from gaussian_robot.backends.gsplat_renderer import load_gaussian_cloud  # noqa: E402
from gaussian_robot.enhance.orchestrator import FillReport, fill_gaps_scene  # noqa: E402
from gaussian_robot.splat.ply_writer import write_gaussian_ply  # noqa: E402

_DEVICE = "cuda"


class _StubFiller:
    """A no-weights ViewFiller that returns the render verbatim with a full-frame mask.

    Lets the GPU test drive the whole distill loop without any diffusion download. ``mask`` is
    a mid weight so the gap views carry real (but gentle) signal into the distiller.
    """

    free_called = False

    def fill(self, degraded: RenderResult, references: object) -> SupervisionView:
        rgb = np.ascontiguousarray(degraded.rgb).astype(np.float32) / 255.0
        h, w = rgb.shape[:2]
        mask = np.full((h, w), 0.5, dtype=np.float32)
        return SupervisionView(camera=degraded.camera, target_rgb=rgb, mask=mask)

    def free(self) -> None:
        self.free_called = True


def _write_synthetic_ply(path: Path, n: int = 4000) -> None:
    rng = np.random.default_rng(1)
    means = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1))
    scales = np.full((n, 3), 0.04, np.float32)
    opacities = np.full((n,), 0.6, np.float32)
    sh = rng.uniform(0.0, 1.0, size=(n, 1, 3)).astype(np.float32)  # sh_degree 0 (DC only)
    write_gaussian_ply(path, means, quats, scales, opacities, sh)


def _write_colmap_model(model_dir: Path, images_dir: Path, n_views: int = 16) -> None:
    """Write a minimal COLMAP cameras.bin + images.bin and matching blank PNG images.

    Cameras ring the synthetic cloud at y=0 looking inward (SIMPLE_PINHOLE, camera_id 1), the
    exact format ``read_cameras_bin`` / ``read_images_bin`` parse.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    w = h = 64
    f = 64.0

    # cameras.bin: one SIMPLE_PINHOLE camera (model 0): f, cx, cy
    with (model_dir / "cameras.bin").open("wb") as fh:
        fh.write(struct.pack("<Q", 1))
        fh.write(struct.pack("<ii", 1, 0))  # camera_id=1, model_id=0 (SIMPLE_PINHOLE)
        fh.write(struct.pack("<QQ", w, h))
        fh.write(struct.pack("<3d", f, w / 2.0, h / 2.0))

    def _qvec_tvec(rot: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # COLMAP stores world->camera R as a quaternion + t = -R @ c.
        t = -rot @ pos
        tr = float(np.trace(rot))
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * s
            qx = (rot[2, 1] - rot[1, 2]) / s
            qy = (rot[0, 2] - rot[2, 0]) / s
            qz = (rot[1, 0] - rot[0, 1]) / s
        else:
            i = int(np.argmax(np.diag(rot)))
            j, k = (i + 1) % 3, (i + 2) % 3
            s = np.sqrt(rot[i, i] - rot[j, j] - rot[k, k] + 1.0) * 2
            q = np.zeros(3)
            q[i] = 0.25 * s
            q[j] = (rot[j, i] + rot[i, j]) / s
            q[k] = (rot[k, i] + rot[i, k]) / s
            qw = (rot[k, j] - rot[j, k]) / s
            qx, qy, qz = q
        return np.array([qw, qx, qy, qz]), t

    with (model_dir / "images.bin").open("wb") as fh:
        fh.write(struct.pack("<Q", n_views))
        for idx in range(n_views):
            ang = 2.0 * np.pi * idx / n_views
            pos = np.array([3.0 * np.cos(ang), 0.0, 3.0 * np.sin(ang)], dtype=np.float64)
            rot = look_at(pos, np.zeros(3), "y")
            qvec, tvec = _qvec_tvec(rot, pos)
            name = f"img_{idx:03d}.png"
            fh.write(struct.pack("<I", idx + 1))  # image_id
            fh.write(struct.pack("<4d", *qvec.tolist()))
            fh.write(struct.pack("<3d", *tvec.tolist()))
            fh.write(struct.pack("<I", 1))  # camera_id
            fh.write(name.encode("utf-8") + b"\x00")
            fh.write(struct.pack("<Q", 0))  # no 2D points
            Image.new("RGB", (w, h), color=(120, 120, 120)).save(images_dir / name)


@pytest.mark.gpu
def test_fill_gaps_scene_writes_new_ply_and_holds_anchors(tmp_path: Path) -> None:
    in_ply = tmp_path / "scene.ply"
    out_ply = tmp_path / "enhanced" / "scene_filled.ply"
    model_dir = tmp_path / "sparse" / "0"
    images_dir = tmp_path / "images"
    _write_synthetic_ply(in_ply)
    _write_colmap_model(model_dir, images_dir)

    before_bytes = in_ply.read_bytes()
    stub = _StubFiller()

    report = fill_gaps_scene(
        in_ply,
        model_dir,
        images_dir,
        out_ply,
        device=_DEVICE,
        camera_id=1,
        downscale=1.0,
        iters=60,
        n_gap_poses=4,
        max_anchor=8,
        eval_stride=4,
        gap_grid=16,
        cap_max_factor=1.05,
        regression_tol_db=10.0,  # don't let the gate trip on the tiny synthetic scene
        view_filler=stub,
    )

    assert isinstance(report, FillReport)
    # NEW ply written; input PLY untouched (read-only guarantee).
    assert out_ply.exists()
    assert in_ply.read_bytes() == before_bytes
    # The injected filler was used and NOT freed (only orchestrator-built fillers are freed).
    assert stub.free_called is False
    # Geometry is frozen + densification OFF: gaussian count is unchanged.
    assert report.n_gaussians_after == report.n_gaussians_before
    # The new ply round-trips through the loader.
    reloaded = load_gaussian_cloud(out_ply, device="cpu")
    assert reloaded.means.shape[0] == report.n_gaussians_after
    # Regression guard: held-out anchors must not collapse (identity-ish supervision).
    assert report.psnr_after > report.psnr_before - 1.5
    assert report.n_eval > 0


@pytest.mark.gpu
def test_fill_gaps_refuses_to_overwrite_input(tmp_path: Path) -> None:
    in_ply = tmp_path / "scene.ply"
    _write_synthetic_ply(in_ply, n=100)
    model_dir = tmp_path / "sparse" / "0"
    images_dir = tmp_path / "images"
    _write_colmap_model(model_dir, images_dir, n_views=6)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        fill_gaps_scene(
            in_ply, model_dir, images_dir, in_ply, device=_DEVICE, view_filler=_StubFiller()
        )
