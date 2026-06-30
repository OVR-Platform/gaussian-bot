"""Load the REAL posed capture images a splat was trained on (COLMAP sparse model).

The base ``capture_poses`` loader returns poses only; an anchored enhancement fine-tune needs
the full triple — pose + intrinsics + the on-disk image — so the photometric loss can be
supervised against ground truth. This reads a COLMAP ``cameras.bin`` + ``images.bin`` model
and resolves each registered image to a file (tolerating a ``.jpg``/``.png`` extension swap,
common when perspective views are re-encoded).

Returns world->camera :class:`Camera` objects (ADR-0002) in the SAME frame as the PLY.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose
from gaussian_robot.splat.capture_poses import _colmap_pose

# COLMAP camera model id -> number of params (subset we handle; others fall back to first 1-2).
_MODEL_NPARAMS: dict[int, int] = {
    0: 3,
    1: 4,
    2: 4,
    3: 5,
    4: 8,
    5: 8,
    6: 12,
    7: 5,
    8: 4,
    9: 5,
    10: 12,
}


@dataclass(frozen=True)
class RealView:
    """A registered capture: its camera (pose + intrinsics) and the resolved image path."""

    camera: Camera
    image_path: Path
    name: str
    camera_id: int


def _intrinsics(
    model_id: int, width: int, height: int, params: tuple[float, ...]
) -> CameraIntrinsics:
    """Map a COLMAP camera model to a pinhole :class:`CameraIntrinsics` (distortion ignored)."""
    if model_id in (0, 2, 8):  # SIMPLE_PINHOLE / SIMPLE_RADIAL / SIMPLE_RADIAL_FISHEYE: f, cx, cy
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    else:  # PINHOLE and richer models: fx, fy, cx, cy, ...
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    return CameraIntrinsics(
        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), width=width, height=height
    )


def read_cameras_bin(path: str | Path) -> dict[int, CameraIntrinsics]:
    """Parse a COLMAP ``cameras.bin`` into ``{camera_id: CameraIntrinsics}``."""
    cams: dict[int, CameraIntrinsics] = {}
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        for _ in range(num):
            cid, model_id = struct.unpack("<ii", f.read(8))
            width, height = struct.unpack("<QQ", f.read(16))
            nparams = _MODEL_NPARAMS.get(model_id, 4)
            params = struct.unpack(f"<{nparams}d", f.read(8 * nparams))
            cams[cid] = _intrinsics(model_id, int(width), int(height), params)
    return cams


def read_images_bin(path: str | Path) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    """Parse a COLMAP ``images.bin`` into ``[(name, camera_id, qvec, tvec)]``."""
    out: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        for _ in range(num):
            f.read(4)  # image_id
            qvec = np.array(struct.unpack("<4d", f.read(32)), dtype=np.float64)
            tvec = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            (cid,) = struct.unpack("<I", f.read(4))
            name_bytes = bytearray()
            while True:
                ch = f.read(1)
                if ch in (b"\x00", b""):
                    break
                name_bytes += ch
            (n_pts,) = struct.unpack("<Q", f.read(8))
            f.read(n_pts * 24)  # skip 2D points (x, y, point3D_id)
            out.append((name_bytes.decode("utf-8", "replace"), int(cid), qvec, tvec))
    return out


def _resolve(images_dir: Path, name: str) -> Path | None:
    """Find ``name`` in ``images_dir``, tolerating a .jpg/.png extension swap."""
    stem = Path(name).stem
    for cand in (name, f"{stem}.png", f"{stem}.jpg", f"{stem}.jpeg", f"{stem}.JPG"):
        p = images_dir / cand
        if p.exists():
            return p
    return None


def load_colmap_views(
    model_dir: str | Path,
    images_dir: str | Path,
    *,
    camera_id: int | None = None,
) -> list[RealView]:
    """Load registered COLMAP views whose image exists on disk.

    ``camera_id`` filters to a single camera (e.g. the perspective training rig); ``None``
    keeps all. Views whose image cannot be resolved are skipped.
    """
    model = Path(model_dir)
    imgs = Path(images_dir)
    cams = read_cameras_bin(model / "cameras.bin")
    views: list[RealView] = []
    for name, cid, qvec, tvec in read_images_bin(model / "images.bin"):
        if camera_id is not None and cid != camera_id:
            continue
        if cid not in cams:
            continue
        path = _resolve(imgs, name)
        if path is None:
            continue
        pose: Pose = _colmap_pose(qvec, tvec)
        views.append(RealView(Camera(pose=pose, intrinsics=cams[cid]), path, name, cid))
    return views


def scale_intrinsics(intr: CameraIntrinsics, factor: float) -> CameraIntrinsics:
    """Scale a pinhole camera to a different resolution (factor in (0, 1] downsamples)."""
    return CameraIntrinsics(
        fx=intr.fx * factor,
        fy=intr.fy * factor,
        cx=intr.cx * factor,
        cy=intr.cy * factor,
        width=max(1, round(intr.width * factor)),
        height=max(1, round(intr.height * factor)),
    )


def load_image(path: str | Path, width: int, height: int) -> np.ndarray:
    """Load an RGB image resized to ``(width, height)`` as ``(H, W, 3)`` float in ``[0, 1]``."""
    img = Image.open(path).convert("RGB").resize((width, height))
    return np.asarray(img, dtype=np.float32) / 255.0


def camera_fovs(views: list[RealView]) -> tuple[np.ndarray, np.ndarray]:
    """Per-view (hfov, vfov) in radians, for ``build_coverage3d``."""
    hfov = np.array(
        [2.0 * np.arctan(v.camera.intrinsics.width / (2.0 * v.camera.intrinsics.fx)) for v in views]
    )
    vfov = np.array(
        [
            2.0 * np.arctan(v.camera.intrinsics.height / (2.0 * v.camera.intrinsics.fy))
            for v in views
        ]
    )
    return hfov, vfov
