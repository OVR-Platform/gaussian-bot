"""Depth estimation protocols and implementations.

Provides a pluggable :class:`DepthEstimator` protocol so the observation
pipeline can swap in a feed-forward monocular depth model (e.g. Depth
Anything 3) instead of relying on the rasterised depth channel.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DepthEstimator(Protocol):
    """Estimates per-pixel depth from an RGB image."""

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        """Return ``(H, W)`` float32 depth from ``(H, W, 3)`` uint8 RGB.

        Depth values are in arbitrary metric-like units (near = small).
        Non-finite values indicate missing depth.
        """
        ...


class DA3DepthEstimator:
    """Depth Anything 3 monocular depth estimator.

    Wraps the ``depth-anything-3`` package.  The model is loaded once at
    construction and kept on GPU for repeated inference.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/DA3-BASE",
        device: str = "cuda",
    ) -> None:
        from depth_anything_3.api import DepthAnything3  # noqa: PLC0415

        self._model = DepthAnything3.from_pretrained(model_name)
        self._model = self._model.to(device=device)
        self._model.eval()
        self._device = device

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        import torch  # noqa: PLC0415

        h, w = rgb.shape[:2]
        prediction = self._model.inference([rgb])

        depth = prediction.depth[0]
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        depth = depth.astype(np.float32)

        if depth.shape != (h, w):
            from PIL import Image  # noqa: PLC0415

            depth_img = Image.fromarray(depth, mode="F")
            depth = np.array(
                depth_img.resize((w, h), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )

        depth[depth <= 0] = np.inf
        return np.asarray(depth, dtype=np.float32)
