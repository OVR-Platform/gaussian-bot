"""GPU Gaussian Splat renderer using gsplat.

Loads a full 3DGS PLY (positions, quaternions, scales, opacities, SH
coefficients) onto the GPU and renders via ``gsplat.rasterization``.
Requires the ``gsplat`` optional extra (torch + gsplat).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from gsplat import rasterization

from gaussian_robot.render.base import RenderResult
from gaussian_robot.render.camera import Camera
from gaussian_robot.splat.scene import SceneBounds


@dataclass(frozen=True)
class GaussianCloud:
    means: torch.Tensor  # (N, 3)
    quats: torch.Tensor  # (N, 4) wxyz
    scales: torch.Tensor  # (N, 3) already exp'd
    opacities: torch.Tensor  # (N,) already sigmoided
    sh_coeffs: torch.Tensor  # (N, K, 3) where K = (degree+1)^2
    sh_degree: int
    bounds: SceneBounds  # tight (percentile) for exploration
    full_bounds: SceneBounds  # raw min/max of all gaussians
    density_grid: np.ndarray | None = None  # (G, G) floor-plane density, normalised [0,1]
    density_bounds: tuple[np.ndarray, np.ndarray] | None = None  # (lo, hi) for the grid


class GsplatRenderer:
    """Renderer protocol implementation using gsplat GPU rasterization."""

    def __init__(self, cloud: GaussianCloud) -> None:
        self.cloud = cloud

    @classmethod
    def from_path(cls, path: str | Path, *, device: str = "cuda") -> GsplatRenderer:
        cloud = load_gaussian_cloud(path, device=device)
        return cls(cloud)

    def render(self, camera: Camera) -> RenderResult:
        h, w = camera.intrinsics.height, camera.intrinsics.width
        rot = camera.pose.rotation
        t = -rot @ camera.pose.position
        viewmat = np.eye(4, dtype=np.float64)
        viewmat[:3, :3] = rot
        viewmat[:3, 3] = t

        device = self.cloud.means.device
        viewmats = torch.tensor(viewmat, dtype=torch.float32, device=device).unsqueeze(0)
        ks = torch.tensor(camera.intrinsics.k_matrix, dtype=torch.float32, device=device).unsqueeze(
            0
        )

        with torch.no_grad():
            rendered, alphas, _meta = rasterization(
                means=self.cloud.means,
                quats=self.cloud.quats,
                scales=self.cloud.scales,
                opacities=self.cloud.opacities,
                colors=self.cloud.sh_coeffs,
                viewmats=viewmats,
                Ks=ks,
                width=w,
                height=h,
                sh_degree=self.cloud.sh_degree,
                render_mode="RGB+D",
                packed=True,
            )

        # gsplat rasterises with OpenGL Y-up framebuffer; flip to image convention.
        rgb_float = rendered[0, :, :, :3].flip(0).clamp(0.0, 1.0)
        rgb = (rgb_float * 255.0).to(torch.uint8).cpu().numpy()
        depth = rendered[0, :, :, 3].flip(0).cpu().numpy().astype(np.float32)
        depth[depth <= 0] = np.inf
        alpha = alphas[0, :, :, 0].flip(0).cpu().numpy().astype(np.float32)

        return RenderResult(rgb=rgb, camera=camera, depth=depth, alpha=alpha)


_PLY_STRUCT_CODES = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def load_gaussian_cloud(path: str | Path, *, device: str = "cuda") -> GaussianCloud:
    """Load a 3DGS PLY file into GPU tensors."""
    p = Path(path)
    with p.open("rb") as fh:
        fmt, n_verts, props = _parse_header(fh)
        data = _read_binary_fast(fh, fmt, n_verts, props)

    means = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)

    has_gaussian = all(
        k in data
        for k in ("rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2", "opacity")
    )
    if not has_gaussian:
        raise ValueError(f"PLY lacks gaussian properties (rot/scale/opacity): {p}")

    quats = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1).astype(
        np.float32
    )
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    quats = quats / norms

    scales_log = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1).astype(
        np.float32
    )
    scales = np.exp(scales_log)

    opacity_logit = data["opacity"].astype(np.float32)
    opacities = 1.0 / (1.0 + np.exp(-opacity_logit))

    rest_keys = sorted(
        [k for k in data if k.startswith("f_rest_")],
        key=lambda k: int(k.split("_")[-1]),
    )
    n_rest_per_channel = len(rest_keys) // 3 if rest_keys else 0
    sh_degree = int(np.sqrt(1 + n_rest_per_channel)) - 1 if n_rest_per_channel > 0 else 0
    k = (sh_degree + 1) ** 2

    sh = np.zeros((n_verts, k, 3), dtype=np.float32)
    if all(name in data for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        sh[:, 0, 0] = data["f_dc_0"].astype(np.float32)
        sh[:, 0, 1] = data["f_dc_1"].astype(np.float32)
        sh[:, 0, 2] = data["f_dc_2"].astype(np.float32)

    for c in range(3):
        for j in range(n_rest_per_channel):
            key = f"f_rest_{c * n_rest_per_channel + j}"
            if key in data:
                sh[:, 1 + j, c] = data[key].astype(np.float32)

    lo = np.percentile(means, 2, axis=0).astype(np.float64)
    hi = np.percentile(means, 98, axis=0).astype(np.float64)
    bounds = SceneBounds(min=lo, max=hi)
    full_bounds = SceneBounds(
        min=means.min(axis=0).astype(np.float64),
        max=means.max(axis=0).astype(np.float64),
    )

    # Floor-plane density grid (opacity-weighted) for guiding exploration.
    grid_size = 64
    x_bins = np.linspace(lo[0], hi[0], grid_size + 1)
    z_bins = np.linspace(lo[2], hi[2], grid_size + 1)
    density, _, _ = np.histogram2d(
        means[:, 0],
        means[:, 2],
        bins=[x_bins, z_bins],
        weights=opacities,
    )
    dmax = density.max()
    density_norm = np.log1p(density) / np.log1p(dmax) if dmax > 0 else density

    return GaussianCloud(
        means=torch.tensor(means, device=device),
        quats=torch.tensor(quats, device=device),
        scales=torch.tensor(scales, device=device),
        opacities=torch.tensor(opacities, device=device),
        sh_coeffs=torch.tensor(sh, device=device),
        sh_degree=sh_degree,
        bounds=bounds,
        full_bounds=full_bounds,
        density_grid=density_norm,
        density_bounds=(lo, hi),
    )


def _parse_header(fh: BinaryIO) -> tuple[str, int, list[tuple[str, str]]]:
    first = fh.readline().decode("ascii", errors="replace").strip()
    if first != "ply":
        raise ValueError("not a PLY file")
    fmt = ""
    n_verts = 0
    props: list[tuple[str, str]] = []
    current = ""
    while True:
        line = fh.readline().decode("ascii", errors="replace").strip()
        if not line:
            raise ValueError("unexpected EOF in PLY header")
        if line == "end_header":
            break
        parts = line.split()
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[:2] == ["element", "vertex"]:
            current = "vertex"
            n_verts = int(parts[2])
        elif parts[0] == "element":
            current = parts[1]
        elif parts[0] == "property" and current == "vertex":
            if parts[1] == "list":
                raise ValueError("list properties not supported")
            props.append((parts[2], parts[1]))
    return fmt, n_verts, props


def _read_binary_fast(
    fh: BinaryIO, fmt: str, n_verts: int, props: list[tuple[str, str]]
) -> dict[str, np.ndarray]:
    if fmt not in ("binary_little_endian", "binary_big_endian"):
        return _read_ascii(fh, n_verts, props)

    endian = "<" if fmt == "binary_little_endian" else ">"
    struct_fmt = endian + "".join(_PLY_STRUCT_CODES[t] for _, t in props)
    row_size = struct.calcsize(struct_fmt)
    raw = fh.read(row_size * n_verts)
    if len(raw) != row_size * n_verts:
        raise ValueError("unexpected EOF reading binary PLY")

    np_dtype = np.dtype([(name, endian + _PLY_STRUCT_CODES[t]) for name, t in props])
    arr = np.frombuffer(raw, dtype=np_dtype, count=n_verts)
    return {name: arr[name].astype(np.float64) for name, _ in props}


def _read_ascii(fh: BinaryIO, n_verts: int, props: list[tuple[str, str]]) -> dict[str, np.ndarray]:
    rows = []
    for _ in range(n_verts):
        row = fh.readline().decode("ascii", errors="replace").split()
        rows.append([float(v) for v in row[: len(props)]])
    arr = np.asarray(rows, dtype=np.float64)
    return {name: arr[:, i] for i, (name, _) in enumerate(props)}
