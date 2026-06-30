"""Geometric (no-weights) :class:`ViewFiller` — the always-available fallback.

Given a degraded :class:`RenderResult` at a coverage-gap pose, fill the low-alpha holes with a
cheap *classical* inpaint and return a :class:`SupervisionView` whose ``mask`` is the feathered
coverage mask ``M`` (see :mod:`gaussian_robot.enhance.mask`). This carries no generative prior —
it only diffuses trusted neighbours into the holes — but it has zero dependencies beyond OpenCV
(optional) and numpy, so it ALWAYS works and is the safe default when no diffusion weights are
present.

The trusted (high-alpha) pixels are taken verbatim from the render; only the hole pixels are
synthesized, exactly as the hard-recomposite ``M*generated + (1-M)*render`` would do — here the
"generated" frame is the classical inpaint.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from gaussian_robot.enhance.mask import coverage_mask
from gaussian_robot.enhance.protocols import SupervisionView

if TYPE_CHECKING:
    from gaussian_robot.render.base import RenderResult


def _alpha_of(degraded: RenderResult) -> np.ndarray:
    """Accumulated-opacity map ``(H, W)`` float32 in ``[0, 1]`` for the render.

    A render with ``alpha=None`` is treated as fully covered (no holes), so the filler is a
    safe identity in that case rather than crashing.
    """
    if degraded.alpha is not None:
        return np.asarray(degraded.alpha, dtype=np.float32)
    h, w = degraded.rgb.shape[:2]
    return np.ones((h, w), dtype=np.float32)


def _cv2_inpaint(render_u8: np.ndarray, hole: np.ndarray) -> np.ndarray | None:
    """OpenCV Telea inpaint of ``hole`` (bool ``(H, W)``) into ``render_u8`` ``(H, W, 3)``.

    Returns ``(H, W, 3)`` uint8, or ``None`` if OpenCV is unavailable. OpenCV wants BGR, but
    inpainting is channel-symmetric so the RGB/BGR distinction is irrelevant here.
    """
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return None
    mask_u8 = (hole.astype(np.uint8)) * 255
    filled: np.ndarray = cv2.inpaint(
        np.ascontiguousarray(render_u8), mask_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )
    return filled.astype(np.uint8)


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur of ``(H, W, C)`` float via a summed-area table (no scipy/cv2)."""
    h, w = img.shape[:2]
    pad = radius
    padded = np.pad(img, ((pad + 1, pad), (pad + 1, pad), (0, 0)), mode="edge")
    sat = padded.cumsum(axis=0).cumsum(axis=1)
    ys = np.arange(h)
    xs = np.arange(w)
    y0 = ys
    y1 = ys + 2 * radius + 1
    x0 = xs
    x1 = xs + 2 * radius + 1
    a = sat[np.ix_(y1, x1)]
    b = sat[np.ix_(y0, x1)]
    c = sat[np.ix_(y1, x0)]
    d = sat[np.ix_(y0, x0)]
    area = float((2 * radius + 1) ** 2)
    out: np.ndarray = (a - b - c + d) / area
    return out


def _blur_fill(render_u8: np.ndarray, hole: np.ndarray, *, iters: int = 24) -> np.ndarray:
    """Diffuse trusted pixels into the holes by iterated masked box-blur (numpy fallback).

    Each pass blurs the current image and the (binary) trust map, then renormalises so the hole
    pixels take the blurred *trusted* colour; trusted pixels are restored exactly every pass. A
    few dozen passes propagate colour across typical gap widths.
    """
    img = render_u8.astype(np.float32) / 255.0
    trust = (~hole).astype(np.float32)[..., None]
    known = img * trust
    for _ in range(iters):
        num = _box_blur(known, radius=2)
        den = _box_blur(trust, radius=2)
        guess = num / np.clip(den, 1e-6, None)
        known = np.where(trust > 0.5, img, guess)
        trust = np.where(den > 1e-4, 1.0, trust)
    return (known.clip(0.0, 1.0) * 255.0).astype(np.uint8)


class GeometricFiller:
    """Classical hole-filling :class:`ViewFiller` (OpenCV Telea inpaint, blurred-neighbour fallback).

    Parameters
    ----------
    tau_lo, feather:
        Coverage-mask thresholds (forwarded to :func:`coverage_mask`). The hole region inpainted
        is ``M >= hole_threshold``.
    hole_threshold:
        Pixels with mask ``M`` at/above this are treated as holes to synthesize.
    prefer_cv2:
        Use OpenCV Telea inpaint when available; otherwise the numpy blurred-neighbour fill.
    """

    def __init__(
        self,
        *,
        tau_lo: float = 0.5,
        feather: float = 0.15,
        hole_threshold: float = 0.5,
        prefer_cv2: bool = True,
    ) -> None:
        self._tau_lo = tau_lo
        self._feather = feather
        self._hole_threshold = hole_threshold
        self._prefer_cv2 = prefer_cv2

    def fill(self, degraded: RenderResult, references: Sequence[RenderResult]) -> SupervisionView:
        """Return a :class:`SupervisionView` with holes classically inpainted; ``mask = M``."""
        del references  # geometric fill is single-view; references are unused
        import torch  # noqa: PLC0415

        render_u8 = np.ascontiguousarray(degraded.rgb).astype(np.uint8)
        alpha = _alpha_of(degraded)
        mask_t = coverage_mask(
            torch.as_tensor(alpha, dtype=torch.float32),
            tau_lo=self._tau_lo,
            feather=self._feather,
        )
        mask = mask_t.cpu().numpy().astype(np.float32)
        hole = mask >= self._hole_threshold

        generated_u8: np.ndarray
        if not bool(hole.any()):
            generated_u8 = render_u8
        else:
            inpainted = _cv2_inpaint(render_u8, hole) if self._prefer_cv2 else None
            generated_u8 = inpainted if inpainted is not None else _blur_fill(render_u8, hole)

        generated = generated_u8.astype(np.float32) / 255.0
        render = render_u8.astype(np.float32) / 255.0
        m3 = mask[..., None]
        target = (m3 * generated + (1.0 - m3) * render).clip(0.0, 1.0).astype(np.float32)
        return SupervisionView(camera=degraded.camera, target_rgb=target, mask=mask)
