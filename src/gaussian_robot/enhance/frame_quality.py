"""Render-quality scoring so the robot flags the DEGRADED frames — the ones Difix should fix.

The coverage-frontier auto-mark flagged any frame with geometry in view, so it kept picking
already-clean frames and Difix had nothing to do. These metrics score a rendered view by how
*bad* it looks, so the navigator can mark the genuinely degraded viewpoints (blurry / under-
reconstructed) and the fill targets those.

- :func:`sharpness` — variance of the Laplacian (the classic focus measure). LOW = blurry.
- :func:`hole_fraction` — fraction of pixels the splat barely covers (low accumulated alpha).
- :func:`rank_degraded` — combine both into a badness ranking, dropping frames that are mostly
  void (looking into empty space — nothing to repair there).
"""

from __future__ import annotations

import numpy as np

# 3x3 discrete Laplacian; its response variance over a grayscale image measures high-frequency
# energy — high on sharp edges, near-zero on a blurred/smeared render.
_LAPLACIAN = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb, dtype=np.float32)
    if a.max() > 1.5:  # uint8 -> [0,1]
        a = a / 255.0
    gray: np.ndarray = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    return gray


def _conv2d_valid(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Tiny valid-region 2D correlation (no SciPy dependency)."""
    kh, kw = kernel.shape
    h, w = img.shape
    if h < kh or w < kw:
        return np.zeros((0, 0), dtype=np.float32)
    out = np.zeros((h - kh + 1, w - kw + 1), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * img[i : i + out.shape[0], j : j + out.shape[1]]
    return out


def sharpness(rgb: np.ndarray) -> float:
    """Variance of the Laplacian of ``rgb`` — higher is sharper, lower is blurrier."""
    lap = _conv2d_valid(_to_gray(rgb), _LAPLACIAN)
    return float(lap.var()) if lap.size else 0.0


def hole_fraction(alpha: np.ndarray | None, tau: float = 0.5) -> float:
    """Fraction of pixels with accumulated opacity below ``tau`` (under-covered / holes)."""
    if alpha is None:
        return 0.0
    a = np.asarray(alpha, dtype=np.float32)
    return float((a < tau).mean()) if a.size else 0.0


def rank_degraded(
    sharpnesses: list[float],
    hole_fracs: list[float],
    *,
    max_hole: float = 0.12,
) -> list[int]:
    """Blurriest-first indices among the WELL-COVERED frames — the "bad but real" ones.

    Holes are an EXCLUSION filter, not a badness reward: a frame the splat barely covers
    (``hole_fraction > max_hole``) has no real content for Difix to sharpen, so distilling its
    "fix" is pure hallucination (Difix invents a room from a smear). Those are dropped. Among the
    frames that DO have real geometry, the ones worth fixing are the blurriest — soft, smeared
    renders of content that is genuinely there. Returned worst (blurriest) first.

    If every frame is holey (nothing well-covered), returns ``[]`` — correctly refusing to feed
    Difix a void rather than inventing detail.
    """
    n = len(sharpnesses)
    if n == 0:
        return []
    sh = np.asarray(sharpnesses, dtype=np.float64)
    ho = np.asarray(hole_fracs, dtype=np.float64)
    covered = np.nonzero(ho <= max_hole)[0]
    if covered.size == 0:
        return []
    order = covered[np.argsort(sh[covered])]  # ascending sharpness = blurriest first
    return [int(i) for i in order]
