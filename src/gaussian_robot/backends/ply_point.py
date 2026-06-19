"""CPU PLY preview renderer.

This backend is a lightweight bridge until a full Gaussian renderer is chosen.
It reads PLY vertices, projects them with the existing camera model, and returns
approximate RGB/depth panels through the renderer protocol.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from gaussian_robot.render.base import RenderResult
from gaussian_robot.render.camera import Camera
from gaussian_robot.splat.scene import SceneBounds

_PLY_TYPES: dict[str, tuple[str, np.dtype[Any]]] = {
    "char": ("b", np.dtype(np.int8)),
    "int8": ("b", np.dtype(np.int8)),
    "uchar": ("B", np.dtype(np.uint8)),
    "uint8": ("B", np.dtype(np.uint8)),
    "short": ("h", np.dtype(np.int16)),
    "int16": ("h", np.dtype(np.int16)),
    "ushort": ("H", np.dtype(np.uint16)),
    "uint16": ("H", np.dtype(np.uint16)),
    "int": ("i", np.dtype(np.int32)),
    "int32": ("i", np.dtype(np.int32)),
    "uint": ("I", np.dtype(np.uint32)),
    "uint32": ("I", np.dtype(np.uint32)),
    "float": ("f", np.dtype(np.float32)),
    "float32": ("f", np.dtype(np.float32)),
    "double": ("d", np.dtype(np.float64)),
    "float64": ("d", np.dtype(np.float64)),
}


@dataclass(frozen=True)
class PLYPointCloud:
    """Vertex positions and display colours loaded from a PLY file."""

    points: np.ndarray
    colors: np.ndarray

    @property
    def bounds(self) -> SceneBounds:
        return SceneBounds(
            min=self.points.min(axis=0).astype(np.float64),
            max=self.points.max(axis=0).astype(np.float64),
        )


class PLYPointRenderer:
    """Renderer protocol implementation for PLY point previews."""

    def __init__(
        self,
        cloud: PLYPointCloud,
        *,
        background: tuple[int, int, int] = (8, 8, 10),
        point_radius: int = 1,
    ) -> None:
        self.cloud = cloud
        self.background = background
        self.point_radius = point_radius

    @classmethod
    def from_path(cls, path: str | Path, *, max_points: int = 250_000) -> PLYPointRenderer:
        return cls(load_ply_point_cloud(path, max_points=max_points))

    def render(self, camera: Camera) -> RenderResult:
        h, w = camera.intrinsics.height, camera.intrinsics.width
        rgb = np.full((h, w, 3), self.background, dtype=np.uint8)
        depth = np.full((h, w), np.inf, dtype=np.float32)

        points_cam = camera.pose.world_to_camera(self.cloud.points)
        z = points_cam[:, 2]
        valid_z = z > 1e-6
        if not np.any(valid_z):
            return RenderResult(rgb=rgb, camera=camera, depth=depth)

        x = points_cam[:, 0]
        y = points_cam[:, 1]
        u = np.rint(camera.intrinsics.fx * x / z + camera.intrinsics.cx).astype(np.int32)
        v = np.rint(camera.intrinsics.fy * y / z + camera.intrinsics.cy).astype(np.int32)
        in_frame = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        visible = np.flatnonzero(in_frame)
        if visible.size == 0:
            return RenderResult(rgb=rgb, camera=camera, depth=depth)

        order = visible[np.argsort(z[visible])]
        radius = max(0, self.point_radius)
        for idx in order:
            px = int(u[idx])
            py = int(v[idx])
            z_val = np.float32(z[idx])
            x0 = max(0, px - radius)
            x1 = min(w, px + radius + 1)
            y0 = max(0, py - radius)
            y1 = min(h, py + radius + 1)
            patch = depth[y0:y1, x0:x1]
            mask = z_val < patch
            if np.any(mask):
                patch[mask] = z_val
                rgb_patch = rgb[y0:y1, x0:x1]
                rgb_patch[mask] = self.cloud.colors[idx]

        return RenderResult(rgb=rgb, camera=camera, depth=depth)


def load_ply_point_cloud(path: str | Path, *, max_points: int = 250_000) -> PLYPointCloud:
    """Load vertex positions and display colours from an ASCII/binary PLY."""
    p = Path(path)
    with p.open("rb") as fh:
        fmt, vertex_count, properties = _read_header(fh)
        if vertex_count <= 0:
            raise ValueError(f"PLY has no vertices: {p}")
        data = _read_vertices(fh, fmt, vertex_count, properties)

    points = _positions(data)
    colors = _colors(data, points)
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[idx]
        colors = colors[idx]
    return PLYPointCloud(points=points, colors=colors)


def _read_header(fh: BinaryIO) -> tuple[str, int, list[tuple[str, str]]]:
    first = fh.readline().decode("ascii", errors="replace").strip()
    if first != "ply":
        raise ValueError("not a PLY file")

    fmt = ""
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    current_element = ""
    while True:
        line = fh.readline().decode("ascii", errors="replace").strip()
        if line == "":
            raise ValueError("unexpected EOF in PLY header")
        if line == "end_header":
            break
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[:2] == ["element", "vertex"]:
            current_element = "vertex"
            vertex_count = int(parts[2])
        elif parts[0] == "element":
            current_element = parts[1]
        elif parts[0] == "property" and current_element == "vertex":
            if parts[1] == "list":
                raise ValueError("list properties in vertex elements are not supported")
            properties.append((parts[2], parts[1]))

    if fmt not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"unsupported PLY format: {fmt!r}")
    return fmt, vertex_count, properties


def _read_vertices(
    fh: BinaryIO, fmt: str, vertex_count: int, properties: list[tuple[str, str]]
) -> dict[str, np.ndarray]:
    if fmt == "ascii":
        return _read_ascii_vertices(fh, vertex_count, properties)
    return _read_binary_vertices(fh, fmt, vertex_count, properties)


def _read_ascii_vertices(
    fh: BinaryIO, vertex_count: int, properties: list[tuple[str, str]]
) -> dict[str, np.ndarray]:
    rows: list[list[float]] = []
    for _ in range(vertex_count):
        row = fh.readline().decode("ascii", errors="replace").split()
        if len(row) < len(properties):
            raise ValueError("vertex row has fewer fields than the PLY header declares")
        rows.append([float(v) for v in row[: len(properties)]])
    arr = np.asarray(rows, dtype=np.float64)
    return {name: arr[:, i] for i, (name, _type_name) in enumerate(properties)}


def _read_binary_vertices(
    fh: BinaryIO, fmt: str, vertex_count: int, properties: list[tuple[str, str]]
) -> dict[str, np.ndarray]:
    endian = "<" if fmt == "binary_little_endian" else ">"
    struct_fmt = endian + "".join(_PLY_TYPES[type_name][0] for _name, type_name in properties)
    row_struct = struct.Struct(struct_fmt)
    raw = fh.read(row_struct.size * vertex_count)
    if len(raw) != row_struct.size * vertex_count:
        raise ValueError("unexpected EOF while reading binary PLY vertices")
    out = {
        name: np.empty(vertex_count, dtype=_PLY_TYPES[type_name][1])
        for name, type_name in properties
    }
    for i in range(vertex_count):
        values = row_struct.unpack_from(raw, i * row_struct.size)
        for value, (name, _type_name) in zip(values, properties, strict=True):
            out[name][i] = value
    return {name: values.astype(np.float64) for name, values in out.items()}


def _positions(data: dict[str, np.ndarray]) -> np.ndarray:
    missing = [name for name in ("x", "y", "z") if name not in data]
    if missing:
        raise ValueError(f"PLY missing vertex properties: {missing}")
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)


def _colors(data: dict[str, np.ndarray], points: np.ndarray) -> np.ndarray:
    if all(name in data for name in ("red", "green", "blue")):
        colors = np.stack([data["red"], data["green"], data["blue"]], axis=1)
        return np.asarray(colors.clip(0, 255), dtype=np.uint8)
    if all(name in data for name in ("r", "g", "b")):
        colors = np.stack([data["r"], data["g"], data["b"]], axis=1)
        return np.asarray(colors.clip(0, 255), dtype=np.uint8)
    if all(name in data for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1)
        colors = (1.0 / (1.0 + np.exp(-dc)) * 255.0).clip(0, 255)
        return np.asarray(colors, dtype=np.uint8)

    mins = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - mins, 1e-9)
    colors = ((points - mins) / span * 255.0).clip(0, 255)
    return np.asarray(colors, dtype=np.uint8)
