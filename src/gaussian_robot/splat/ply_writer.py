"""Serialize a gaussian cloud back to a 3DGS-format binary PLY.

The renderer's loader (:func:`gaussian_robot.backends.gsplat_renderer.load_gaussian_cloud`)
reads scenes but never writes them; an enhancement pass that fine-tunes the gaussians needs
to persist the result. This writer is the exact inverse of that loader:

- ``scales`` are written as ``log`` (the loader applies ``exp``),
- ``opacities`` as logit (the loader applies ``sigmoid``),
- SH coefficients as ``f_dc_{c}`` (DC term) plus ``f_rest_{c*nrest + j}`` in the
  channel-major order the loader expects,
- quaternions ``rot_0..3`` (wxyz) and ``x/y/z`` means verbatim.

Round-trip (write -> ``load_gaussian_cloud``) reproduces the inputs up to float32 precision
and the loader's quaternion renormalisation. See ``tests/test_ply_writer.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_EPS = 1e-6


def _to_numpy(x: Any) -> np.ndarray:
    """Accept a numpy array or a torch tensor (without importing torch) -> numpy."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def write_gaussian_ply(
    path: str | Path,
    means: Any,
    quats: Any,
    scales: Any,
    opacities: Any,
    sh_coeffs: Any,
) -> Path:
    """Write a gaussian set to a binary-little-endian 3DGS PLY; return the path.

    ``scales`` and ``opacities`` are the **activated** values (matching
    :class:`~gaussian_robot.backends.gsplat_renderer.GaussianCloud`); they are inverted to
    log / logit on write. ``sh_coeffs`` is ``(N, K, 3)`` with ``K = (sh_degree + 1) ** 2``.
    """
    means_a = _to_numpy(means).astype(np.float32)
    quats_a = _to_numpy(quats).astype(np.float32)
    scales_a = _to_numpy(scales).astype(np.float32)
    opac_a = _to_numpy(opacities).astype(np.float32).reshape(-1)
    sh = _to_numpy(sh_coeffs).astype(np.float32)

    n = means_a.shape[0]
    if means_a.shape != (n, 3) or quats_a.shape != (n, 4) or scales_a.shape != (n, 3):
        raise ValueError("means/quats/scales must be (N,3)/(N,4)/(N,3)")
    if opac_a.shape != (n,):
        raise ValueError(f"opacities must be (N,), got {opac_a.shape}")
    if sh.ndim != 3 or sh.shape[0] != n or sh.shape[2] != 3:
        raise ValueError(f"sh_coeffs must be (N, K, 3), got {sh.shape}")
    n_rest = sh.shape[1] - 1  # rest coeffs per channel (DC is index 0)

    # invert the loader's activations
    scale_log = np.log(np.maximum(scales_a, _EPS))
    opac_clip = np.clip(opac_a, _EPS, 1.0 - _EPS)
    opacity_logit = np.log(opac_clip / (1.0 - opac_clip))

    zeros = np.zeros(n, dtype=np.float32)
    cols: list[tuple[str, np.ndarray]] = [
        ("x", means_a[:, 0]),
        ("y", means_a[:, 1]),
        ("z", means_a[:, 2]),
        ("nx", zeros),
        ("ny", zeros),
        ("nz", zeros),
        ("f_dc_0", sh[:, 0, 0]),
        ("f_dc_1", sh[:, 0, 1]),
        ("f_dc_2", sh[:, 0, 2]),
    ]
    for c in range(3):  # channel-major rest layout, matching the loader
        for j in range(n_rest):
            cols.append((f"f_rest_{c * n_rest + j}", sh[:, 1 + j, c]))
    cols.append(("opacity", opacity_logit))
    cols += [
        ("scale_0", scale_log[:, 0]),
        ("scale_1", scale_log[:, 1]),
        ("scale_2", scale_log[:, 2]),
    ]
    cols += [
        ("rot_0", quats_a[:, 0]),
        ("rot_1", quats_a[:, 1]),
        ("rot_2", quats_a[:, 2]),
        ("rot_3", quats_a[:, 3]),
    ]

    names = [name for name, _ in cols]
    table = np.stack([col.astype(np.float32) for _, col in cols], axis=1).astype("<f4")

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    header += "".join(f"property float {name}\n" for name in names)
    header += "end_header\n"

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(np.ascontiguousarray(table).tobytes())
    return p
