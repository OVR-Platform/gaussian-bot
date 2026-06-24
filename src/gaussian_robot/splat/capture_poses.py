"""Load the capture camera poses a Gaussian Splat was reconstructed from.

The original capture cameras are the single most reliable source of *valid*
viewpoints: every one of them looks at geometry that was actually observed and
reconstructed, so a seed placed there is guaranteed to see real scene content
rather than an empty hole. We use them as the seed-candidate pool (see
:mod:`gaussian_robot.session`).

Two sources are supported, in order of frame-safety:

1. **3DGS ``cameras.json``** — the per-camera dump written by Gaussian Splatting
   trainers next to the saved ``point_cloud.ply``. It is emitted in the *same
   world frame as the PLY*, so positions need no alignment. ``position`` is the
   camera centre in world; ``rotation`` is the camera->world matrix.
2. **COLMAP** ``images.bin``/``images.txt`` — the raw sparse reconstruction.
   Use only when its world frame matches the PLY (raw COLMAP output may be
   rescaled/realigned before training; prefer ``cameras.json`` when present).

All poses are returned in the repo convention (ADR-0002): ``Pose.position`` is
the camera centre in world, ``Pose.rotation`` is the **world->camera** matrix
(OpenCV axes), which COLMAP and 3DGS share.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np

from gaussian_robot.render.camera import Pose

_CAMERAS_JSON = "cameras.json"
_COLMAP_IMAGE_FILES = ("images.bin", "images.txt")
_DEFAULT_FOV = (math.radians(70.0), math.radians(50.0))  # fallback (h, v) when intrinsics absent


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) -> world->camera rotation matrix."""
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def parse_3dgs_cameras_json(path: str | Path) -> list[Pose]:
    """Parse a 3DGS ``cameras.json`` into world->camera poses.

    Each entry stores ``position`` (camera centre in world) and ``rotation``
    (camera->world). We transpose the rotation to get world->camera (ADR-0002).
    """
    with open(path) as f:
        entries = json.load(f)
    poses: list[Pose] = []
    for e in entries:
        pos = np.asarray(e["position"], dtype=np.float64)
        r_c2w = np.asarray(e["rotation"], dtype=np.float64)
        if pos.shape != (3,) or r_c2w.shape != (3, 3):
            continue
        poses.append(Pose(position=pos, rotation=r_c2w.T.copy()))
    return poses


def _colmap_pose(qvec: np.ndarray, tvec: np.ndarray) -> Pose:
    """Build a world->camera :class:`Pose` from COLMAP qvec/tvec."""
    r_w2c = _qvec_to_rotmat(qvec)
    position = -r_w2c.T @ tvec
    return Pose(position=position, rotation=r_w2c)


def parse_colmap_images_bin(path: str | Path) -> list[Pose]:
    """Parse a COLMAP ``images.bin`` into world->camera poses."""
    poses: list[Pose] = []
    with open(path, "rb") as f:
        (num_images,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_images):
            f.read(4)  # image_id (uint32)
            qvec = np.array(struct.unpack("<4d", f.read(32)), dtype=np.float64)
            tvec = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            f.read(4)  # camera_id (uint32)
            name_bytes = bytearray()
            while True:
                ch = f.read(1)
                if ch in (b"\x00", b""):
                    break
                name_bytes += ch
            (num_points2d,) = struct.unpack("<Q", f.read(8))
            f.read(num_points2d * 24)  # skip (x, y, point3D_id) per 2D point
            poses.append(_colmap_pose(qvec, tvec))
    return poses


def parse_colmap_images_txt(path: str | Path) -> list[Pose]:
    """Parse a COLMAP ``images.txt`` into world->camera poses."""
    poses: list[Pose] = []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        qvec = np.array([float(p) for p in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(p) for p in parts[5:8]], dtype=np.float64)
        poses.append(_colmap_pose(qvec, tvec))
        i += 2  # skip the POINTS2D line that follows each image header
    return poses


def load_capture_poses(path: str | Path) -> list[Pose]:
    """Load capture poses from a ``cameras.json``, COLMAP file, or directory.

    A directory is searched for ``cameras.json`` first, then COLMAP
    ``images.bin``/``images.txt``.
    """
    p = Path(path)
    if p.is_dir():
        resolved = _resolve_in_dir(p)
        if resolved is None:
            raise FileNotFoundError(f"no capture poses found under {p}")
        p = resolved
    name = p.name.lower()
    if name.endswith(".json"):
        return parse_3dgs_cameras_json(p)
    if name == "images.bin":
        return parse_colmap_images_bin(p)
    if name == "images.txt":
        return parse_colmap_images_txt(p)
    raise ValueError(f"unrecognised capture-pose file: {p}")


def _resolve_in_dir(directory: Path) -> Path | None:
    """Find a capture-pose file inside ``directory`` (cameras.json preferred)."""
    cj = directory / _CAMERAS_JSON
    if cj.is_file():
        return cj
    for fname in _COLMAP_IMAGE_FILES:
        f = directory / fname
        if f.is_file():
            return f
    # Common COLMAP layout: sparse/0/images.bin
    for sub in ("sparse/0", "sparse"):
        for fname in _COLMAP_IMAGE_FILES:
            f = directory / sub / fname
            if f.is_file():
                return f
    return None


def representative_fov(source: str | Path | None) -> tuple[float, float]:
    """A representative ``(hfov, vfov)`` (radians) for the capture rig.

    Reads ``fx/fy/width/height`` from the first usable 3DGS ``cameras.json`` entry
    (capture rigs are near-uniform). Falls back to a default for COLMAP/unknown
    sources, which don't carry intrinsics here.
    """
    if source is None:
        return _DEFAULT_FOV
    p = Path(source)
    if p.is_dir():
        resolved = _resolve_in_dir(p)
        if resolved is None:
            return _DEFAULT_FOV
        p = resolved
    if not p.name.lower().endswith(".json"):
        return _DEFAULT_FOV
    try:
        entries = json.loads(p.read_text())
    except (OSError, ValueError):
        return _DEFAULT_FOV
    for e in entries:
        try:
            fx, fy = float(e["fx"]), float(e["fy"])
            w, h = float(e["width"]), float(e["height"])
            if fx > 0 and fy > 0:
                return 2.0 * math.atan(w / (2.0 * fx)), 2.0 * math.atan(h / (2.0 * fy))
        except (KeyError, TypeError, ValueError):
            continue
    return _DEFAULT_FOV


def infer_up_axis(poses: list[Pose]) -> str | None:
    """Infer the world up axis from capture poses (their averaged up direction).

    Each camera's up direction in world is ``-rotation[1, :]`` (ADR-0002). Averaged
    over many cameras the per-frame tilt cancels and what remains points along
    gravity-up. Returns a signed axis string (e.g. ``"-y"``) or ``None`` if the
    poses disagree too much to call (e.g. a dome rig with no consistent up).
    """
    if not poses:
        return None
    ups = np.array([-p.rotation[1, :] for p in poses], dtype=np.float64)
    mean = ups.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 0.5:  # cameras point every which way -> no reliable up
        return None
    mean /= norm
    idx = int(np.argmax(np.abs(mean)))
    letter = "xyz"[idx]
    return f"-{letter}" if mean[idx] < 0 else letter


def discover_capture_poses(ply_path: str | Path, *, max_levels: int = 4) -> Path | None:
    """Search upward from a PLY for the capture poses that produced it.

    3DGS writes ``point_cloud/iteration_N/point_cloud.ply`` with ``cameras.json``
    a couple of levels up, so we walk parent directories looking for a
    co-located ``cameras.json`` (frame-safe) and then a COLMAP reconstruction.
    Returns the resolved file path, or ``None`` if nothing is found.
    """
    start = Path(ply_path).resolve().parent
    for parent in (start, *start.parents[:max_levels]):
        resolved = _resolve_in_dir(parent)
        if resolved is not None:
            return resolved
    return None
