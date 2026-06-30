"""Render-quality scoring used to flag the robot's degraded frames."""

from __future__ import annotations

import numpy as np

from gaussian_robot.enhance.frame_quality import (
    hole_fraction,
    rank_degraded,
    sharpness,
)


def _checkerboard(n: int = 64, cell: int = 4) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n]
    g = (((xx // cell) + (yy // cell)) % 2).astype(np.float32)
    return np.repeat(g[..., None], 3, axis=2)


def _blur(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    for _ in range(6):  # repeated 3x3 box blur -> smeared
        s = out.copy()
        s[1:-1] = (out[:-2] + out[1:-1] + out[2:]) / 3.0
        out = s.copy()
        out[:, 1:-1] = (s[:, :-2] + s[:, 1:-1] + s[:, 2:]) / 3.0
    return out


def test_sharpness_drops_with_blur() -> None:
    sharp = _checkerboard()
    blurred = _blur(sharp)
    assert sharpness(sharp) > sharpness(blurred) * 3  # blur kills high-frequency energy


def test_hole_fraction_counts_low_alpha() -> None:
    alpha = np.ones((10, 10), dtype=np.float32)
    alpha[:3] = 0.0  # 30% holes
    assert abs(hole_fraction(alpha) - 0.3) < 1e-6
    assert hole_fraction(None) == 0.0


def test_rank_degraded_blurriest_among_covered_drops_holey() -> None:
    # 0 sharp+covered, 1 blurry+covered (the target), 2 blurry+holey (void -> hallucination trap).
    sharps = [100.0, 2.0, 1.0]
    holes = [0.02, 0.05, 0.40]
    order = rank_degraded(sharps, holes, max_hole=0.12)
    assert 2 not in order  # holey frame excluded — no real content to sharpen, only hallucinate
    assert order[0] == 1  # blurriest well-covered frame is worst-first
    assert order[-1] == 0  # the sharp covered frame is last


def test_rank_degraded_all_holey_returns_empty() -> None:
    # Nothing well-covered -> refuse to feed Difix a void rather than invent detail.
    assert rank_degraded([1.0, 2.0], [0.5, 0.8], max_hole=0.12) == []
