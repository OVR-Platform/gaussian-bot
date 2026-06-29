"""Coverage mask from a render's alpha channel — the reference paper's opacity map ``O``.

``coverage_mask`` is a feathered per-pixel generative weight ``M``: ~1 where the splat is
under-reconstructed (low accumulated opacity / "holes"), ~0 where geometry is trusted. It
gates the distillation loss (only synthesized pixels carry signal) and, downscaled via
``downscale_to_latent`` (``O_z``), drives SDEdit opacity-mixing in the P2 diffusion filler.
Pure torch; negligible cost.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def coverage_mask(
    alpha: torch.Tensor, *, tau_lo: float = 0.5, feather: float = 0.15
) -> torch.Tensor:
    """Feathered generative mask ``M`` in ``[0, 1]`` from accumulated opacity ``alpha`` (H, W).

    ``M -> 1`` as ``alpha`` falls below ``tau_lo - feather`` (a hole); ``M -> 0`` at/above
    ``tau_lo`` (trusted). The transition is a smoothstep, so there are no hard seams for the
    distiller to bake in.
    """
    lo = tau_lo - feather
    span = max(tau_lo - lo, 1e-6)
    t = ((alpha - lo) / span).clamp(0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)  # smoothstep: 0 at lo, 1 at tau_lo
    return 1.0 - smooth


def downscale_to_latent(alpha: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Max-pool ``alpha`` (H, W) down to ``size`` ``(h, w)`` -> ``O_z`` for opacity-mixing.

    Max-pooling (not average) matches the reference paper: a latent cell is considered
    covered if *any* of its pixels is covered, so generative fill is reserved for cells that
    are entirely empty.
    """
    pooled = F.adaptive_max_pool2d(alpha[None, None], output_size=size)
    return pooled[0, 0]
