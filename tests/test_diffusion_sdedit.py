"""Multi-step τ-ladder + SDEdit opacity-mix plumbing in :class:`DiffusionFiller` (ADR-0011).

These pin the two closeness levers at the filler seam without weights or a GPU: a recording
stub stands in for the Difix pipeline and the tests assert exactly what ``fill`` hands it —
the descending timestep ladder, and the ``O_z`` init mask (present only with ``sdedit=True``,
max-pooled to latent resolution).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")  # fillers -> mask imports torch at module level

import torch  # noqa: E402

from gaussian_robot.enhance.fillers import DiffusionFiller  # noqa: E402
from gaussian_robot.render.base import RenderResult  # noqa: E402
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose  # noqa: E402


class _RecordingPipe:
    """Stands in for DifixPipeline: records call kwargs, returns a fixed image."""

    def __init__(self, out: np.ndarray) -> None:
        self.calls: list[dict[str, object]] = []
        self._out = out

    def __call__(self, prompt: str, **kw: object) -> SimpleNamespace:
        self.calls.append({"prompt": prompt, **kw})
        return SimpleNamespace(images=[self._out])


def _degraded(h: int = 16, w: int = 16, alpha: np.ndarray | None = None) -> RenderResult:
    intr = CameraIntrinsics(fx=8.0, fy=8.0, cx=4.0, cy=4.0, width=w, height=h)
    return RenderResult(
        rgb=np.full((h, w, 3), 0.5, dtype=np.float32),
        camera=Camera(pose=Pose(), intrinsics=intr),
        alpha=np.ones((h, w), dtype=np.float32) if alpha is None else alpha,
    )


def _filler_with_pipe(**kw: object) -> tuple[DiffusionFiller, _RecordingPipe]:
    f = DiffusionFiller(filler_mode="difix", device="cpu", **kw)  # type: ignore[arg-type]
    pipe = _RecordingPipe(np.full((16, 16, 3), 0.5, dtype=np.float32))
    f._pipe = pipe  # pre-staged: load() becomes a no-op, no weights download
    return f, pipe


def test_single_step_default_keeps_published_recipe() -> None:
    f, pipe = _filler_with_pipe()
    f.fill(_degraded(), references=[])
    call = pipe.calls[0]
    assert call["timesteps"] == [199]
    assert call["num_inference_steps"] == 1
    assert call["init_mask"] is None
    assert call["guidance_scale"] == 0.0


def test_multi_step_walks_a_descending_tau_ladder() -> None:
    f, pipe = _filler_with_pipe(num_inference_steps=3)
    f.fill(_degraded(), references=[])
    call = pipe.calls[0]
    assert call["timesteps"] == [199, 133, 66]  # τ·(N-i)/N, strictly descending
    assert call["num_inference_steps"] == 3
    assert call["init_mask"] is None  # multi-step alone does not noise the init


def test_tiny_tau_ladder_dedupes_to_a_valid_schedule() -> None:
    f = DiffusionFiller(filler_mode="difix", device="cpu", timestep=2, num_inference_steps=5)
    ladder = f._denoise_timesteps()
    assert ladder == [2, 1]  # deduped, strictly descending, floored at 1


def test_sdedit_pools_opacity_to_latent_resolution() -> None:
    # Left half is a hole (alpha=0), right half observed (alpha=1): O_z must be the MAX-pooled
    # alpha at latent (H/8, W/8) resolution — 1 wherever any pixel in the cell is covered.
    alpha = np.ones((16, 16), dtype=np.float32)
    alpha[:, :8] = 0.0
    f, pipe = _filler_with_pipe(num_inference_steps=2, sdedit=True)
    f.fill(_degraded(alpha=alpha), references=[])
    mask = pipe.calls[0]["init_mask"]
    assert isinstance(mask, torch.Tensor)
    assert tuple(mask.shape) == (2, 2)  # 16/8 x 16/8 latent cells
    assert torch.allclose(mask[:, 0], torch.zeros(2))  # empty cells -> generate
    assert torch.allclose(mask[:, 1], torch.ones(2))  # observed cells -> keep z_deg


def test_sdedit_requires_multi_step() -> None:
    with pytest.raises(ValueError, match="sdedit=True requires num_inference_steps >= 2"):
        DiffusionFiller(filler_mode="difix", device="cpu", sdedit=True)
