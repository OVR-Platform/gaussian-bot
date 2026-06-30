"""Pluggable seams for the enhancement pipeline, mirroring the ``DepthEstimator`` pattern.

Two roles, swappable behind ``runtime_checkable`` Protocols:

- :class:`ViewFiller` — turn a degraded render (+ its coverage mask + reference views) into
  a clean target image for an under-observed pose. Implementations: geometric depth-warp
  (P1), single-step diffusion / Difix (P2).
- :class:`SplatDistiller` — fine-tune a gaussian cloud so it reproduces a set of
  ``(pose, target, mask)`` supervision views, distilling the fills into the geometry.

Milestone-0 implements a concrete :class:`SplatDistiller`
(:class:`gaussian_robot.enhance.distiller.GaussianDistiller`) with identity targets; the
fillers are stubs until P1/P2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from gaussian_robot.render.camera import Camera

if TYPE_CHECKING:
    from gaussian_robot.backends.gsplat_renderer import GaussianCloud
    from gaussian_robot.render.base import RenderResult


@dataclass(frozen=True)
class SupervisionView:
    """One target the distiller fits the cloud to.

    ``mask`` is the per-pixel generative weight M in ``[0, 1]`` (1 = synthesized, supervise
    fully; 0 = trusted, ignore). ``None`` means an unmasked real anchor view.
    """

    camera: Camera
    target_rgb: np.ndarray  # (H, W, 3) float in [0, 1]
    mask: np.ndarray | None = None  # (H, W) float in [0, 1], or None for anchors
    # Per-view loss scale. NOTE: near-INERT under the distiller's per-view Adam (Adam is
    # scale-invariant), so this scalar does NOT meaningfully bias real-anchors-over-fills on its
    # own — to actually reweight, vary sampling frequency / accumulate gradients. Kept for callers
    # that fold it into a single combined loss. Default 1.0 (no-op).
    weight: float = 1.0


@runtime_checkable
class ViewFiller(Protocol):
    """Produce a clean target image for an under-observed pose (P1 geometric / P2 diffusion)."""

    def fill(
        self, degraded: RenderResult, references: Sequence[RenderResult]
    ) -> SupervisionView: ...


@runtime_checkable
class SplatDistiller(Protocol):
    """Fine-tune a gaussian cloud to reproduce a set of supervision views, then export it."""

    def fit(self, views: Sequence[SupervisionView], iters: int) -> None: ...

    def to_cloud(self) -> GaussianCloud: ...
