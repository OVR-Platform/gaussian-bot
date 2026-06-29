"""Coverage mask: low alpha -> high generative weight M, monotone, feathered. Pure CPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gaussian_robot.enhance.mask import coverage_mask, downscale_to_latent  # noqa: E402


def test_mask_low_alpha_is_generative() -> None:
    alpha = torch.tensor([[0.0, 0.2, 0.5, 1.0]])
    m = coverage_mask(alpha, tau_lo=0.5, feather=0.15)
    assert m.shape == alpha.shape
    assert torch.all(m >= 0.0) and torch.all(m <= 1.0)
    assert m[0, 0] == pytest.approx(1.0, abs=1e-5)  # alpha=0 -> fully generative
    assert m[0, 3] == pytest.approx(0.0, abs=1e-5)  # alpha=1 -> trusted
    # monotone non-increasing in alpha
    assert m[0, 0] >= m[0, 1] >= m[0, 2] >= m[0, 3]


def test_downscale_max_pools() -> None:
    alpha = torch.zeros(4, 4)
    alpha[0, 0] = 0.9  # a single covered pixel in the top-left 2x2 block
    out = downscale_to_latent(alpha, (2, 2))
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(0.9)  # max-pool keeps the covered pixel
    assert out[1, 1] == pytest.approx(0.0)
