"""The disagreement-gated recomposite mask in :class:`DiffusionFiller`.

Regression guard for the bug where the alpha-only mask discarded ~all of Difix's output on a
well-reconstructed (alpha≈1) interior — so the supervision target was the blurry render and the
splat never changed. The fix: trust Difix where it meaningfully changed the image. These tests
stub the diffusion forward so they run on CPU with no weights download.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")  # fillers -> mask imports torch at module level

from gaussian_robot.enhance.fillers import DiffusionFiller  # noqa: E402
from gaussian_robot.render.base import RenderResult  # noqa: E402
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose  # noqa: E402


def _degraded(rgb: np.ndarray) -> RenderResult:
    intr = CameraIntrinsics(fx=8.0, fy=8.0, cx=4.0, cy=4.0, width=rgb.shape[1], height=rgb.shape[0])
    cam = Camera(pose=Pose(), intrinsics=intr)
    # alpha all ones -> the alpha-only coverage mask is ~0 everywhere (no transparent holes).
    return RenderResult(rgb=rgb, camera=cam, alpha=np.ones(rgb.shape[:2], dtype=np.float32))


def _filler(generated: np.ndarray, **kw: object) -> DiffusionFiller:
    f = DiffusionFiller(filler_mode="difix", device="cpu", **kw)  # type: ignore[arg-type]
    f._pipe = object()  # make load() a no-op (don't fetch weights)

    def _fake_difix(
        render_u8: np.ndarray,
        ref_u8: np.ndarray | None,
        alpha: np.ndarray | None = None,
    ) -> np.ndarray:
        return generated

    f._difix = _fake_difix  # type: ignore[method-assign]
    return f


def test_disagreement_mask_keeps_difix_where_it_changed_the_image() -> None:
    h, w = 16, 16
    render = np.full((h, w, 3), 0.5, dtype=np.float32)  # flat grey, alpha=1 everywhere
    generated = render.copy()
    generated[:, : w // 2] = 0.9  # Difix "sharpened" the left half strongly
    f = _filler(generated)

    sv = f.fill(_degraded(render), references=[])

    # alpha-only mask would be ~0 (alpha=1). Disagreement lifts the CHANGED half toward 1.
    assert sv.mask is not None
    assert sv.mask[:, : w // 2].mean() > 0.8  # left: trust Difix
    assert sv.mask[:, w // 2 :].mean() < 0.05  # right: unchanged -> protected
    # The target adopts Difix on the left, keeps the render on the right.
    assert sv.target_rgb[:, : w // 2].mean() > 0.8
    assert np.allclose(sv.target_rgb[:, w // 2 :], 0.5, atol=5e-3)


def test_disagreement_off_falls_back_to_alpha_only() -> None:
    h, w = 16, 16
    render = np.full((h, w, 3), 0.5, dtype=np.float32)
    generated = np.full((h, w, 3), 0.9, dtype=np.float32)  # Difix changed everything
    f = _filler(generated, disagreement=False)

    sv = f.fill(_degraded(render), references=[])
    # With alpha=1 and no disagreement term, the mask is ~empty -> target ~= render (the old bug).
    assert sv.mask is not None
    assert sv.mask.mean() < 0.05
    assert np.allclose(sv.target_rgb, 0.5, atol=5e-3)
