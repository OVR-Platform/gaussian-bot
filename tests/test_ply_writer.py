"""PLY writer round-trip: write -> load_gaussian_cloud reproduces the inputs.

This guards study trap #1 — the loader stores *activated* scales (exp) and opacities
(sigmoid), so the writer must invert to log / logit, and the SH ``f_rest`` layout must match
channel-major. Pure CPU (no rasterisation), so it runs without a GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")  # the loader stores tensors; gsplat_renderer imports both
pytest.importorskip("gsplat")

from gaussian_robot.backends.gsplat_renderer import load_gaussian_cloud  # noqa: E402
from gaussian_robot.splat.ply_writer import write_gaussian_ply  # noqa: E402


def _synthetic(
    n: int = 64, sh_degree: int = 1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = (sh_degree + 1) ** 2
    means = rng.uniform(-2.0, 2.0, size=(n, 3)).astype(np.float32)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)  # pre-normalise: loader renormalises
    scales = np.exp(rng.uniform(-3.0, 0.0, size=(n, 3))).astype(np.float32)  # activated (positive)
    opacities = rng.uniform(0.05, 0.95, size=(n,)).astype(np.float32)  # activated, away from clip
    sh = rng.normal(scale=0.5, size=(n, k, 3)).astype(np.float32)
    return means, quats, scales, opacities, sh


def test_round_trip_reproduces_tensors(tmp_path: Path) -> None:
    means, quats, scales, opacities, sh = _synthetic()
    path = tmp_path / "rt.ply"
    write_gaussian_ply(path, means, quats, scales, opacities, sh)

    cloud = load_gaussian_cloud(path, device="cpu")

    assert cloud.sh_degree == 1
    assert cloud.means.shape == (64, 3)
    np.testing.assert_allclose(cloud.means.numpy(), means, atol=1e-4)
    np.testing.assert_allclose(cloud.quats.numpy(), quats, atol=1e-5)
    # scales survive log->exp, opacities survive logit->sigmoid
    np.testing.assert_allclose(cloud.scales.numpy(), scales, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(cloud.opacities.numpy(), opacities, atol=1e-4)
    np.testing.assert_allclose(cloud.sh_coeffs.numpy(), sh, atol=1e-4)


def test_round_trip_degree_zero(tmp_path: Path) -> None:
    # K=1 (DC only): no f_rest properties — exercises the n_rest=0 path.
    means, quats, scales, opacities, sh = _synthetic(n=32, sh_degree=0)
    path = tmp_path / "dc.ply"
    write_gaussian_ply(path, means, quats, scales, opacities, sh)

    cloud = load_gaussian_cloud(path, device="cpu")
    assert cloud.sh_degree == 0
    assert cloud.sh_coeffs.shape == (32, 1, 3)
    np.testing.assert_allclose(cloud.sh_coeffs.numpy(), sh, atol=1e-4)
