"""GaussianDistiller plumbing on a synthetic cloud (requires CUDA + gsplat).

Validates the Milestone-0 load-bearing pieces without a real scene:
- pre-activation ParameterDict + per-attribute Adam + MCMC pass check_sanity,
- a grad-enabled render backprops and the loop runs (identity target -> no regression),
- MCMC refine replaces params / grows N within cap_max (the `_meta`-into-strategy path),
- save_ply round-trips the (possibly grown) cloud through load_gaussian_cloud.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gsplat")
import torch  # noqa: E402

if not torch.cuda.is_available():  # pragma: no cover - environment dependent
    pytest.skip("requires a CUDA GPU", allow_module_level=True)

from gaussian_robot.backends.gsplat_renderer import (  # noqa: E402
    GaussianCloud,
    load_gaussian_cloud,
)
from gaussian_robot.enhance.distiller import GaussianDistiller  # noqa: E402
from gaussian_robot.enhance.protocols import SupervisionView  # noqa: E402
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose  # noqa: E402
from gaussian_robot.splat.scene import SceneBounds  # noqa: E402

pytestmark = pytest.mark.gpu

_DEVICE = "cuda"


def _synthetic_cloud(n: int = 3000) -> GaussianCloud:
    rng = np.random.default_rng(0)
    means = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1))
    scales = np.full((n, 3), 0.03, np.float32)
    opacities = np.full((n,), 0.7, np.float32)
    sh = rng.uniform(0.0, 1.0, size=(n, 1, 3)).astype(np.float32)  # sh_degree 0 (DC only)
    bounds = SceneBounds(min=np.full(3, -1.0), max=np.full(3, 1.0))
    return GaussianCloud(
        means=torch.tensor(means, device=_DEVICE),
        quats=torch.tensor(quats, device=_DEVICE),
        scales=torch.tensor(scales, device=_DEVICE),
        opacities=torch.tensor(opacities, device=_DEVICE),
        sh_coeffs=torch.tensor(sh, device=_DEVICE),
        sh_degree=0,
        bounds=bounds,
        full_bounds=bounds,
    )


def _camera() -> Camera:
    intr = CameraIntrinsics(fx=256.0, fy=256.0, cx=128.0, cy=128.0, width=256, height=256)
    pose = Pose(position=np.array([0.0, 0.0, -3.0]), rotation=np.eye(3))
    return Camera(pose=pose, intrinsics=intr)


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b) ** 2).mean().item())
    return 99.0 if mse <= 1e-12 else -10.0 * math.log10(mse)


def test_render_is_differentiable() -> None:
    dist = GaussianDistiller(_synthetic_cloud(), device=_DEVICE)
    gr = dist.render(_camera())
    assert gr.rgb.shape == (256, 256, 3)
    assert gr.alpha.shape == (256, 256)
    assert gr.rgb.requires_grad  # grad path intact (unlike GsplatRenderer.render)
    gr.rgb.mean().backward()  # type: ignore[no-untyped-call]
    assert dist.params["means"].grad is not None


def test_identity_distill_does_not_regress() -> None:
    dist = GaussianDistiller(
        _synthetic_cloud(),
        device=_DEVICE,
        freeze_means_iters=40,  # geometry frozen for the whole run
        refine_start_iter=10_000,  # no densification in this short run
        ssim_weight=0.2,
    )
    cam = _camera()
    dist.reset_peak_vram()
    target = dist.render(cam).rgb.detach().clamp(0.0, 1.0)  # identity target
    views = [SupervisionView(camera=cam, target_rgb=target.cpu().numpy(), mask=None)]

    dist.fit(views, iters=40)

    out = dist.render(cam).rgb.detach().clamp(0.0, 1.0)
    assert _psnr(out, target) > 30.0  # identity target -> no regression
    assert dist.num_gaussians == 3000  # no growth (refine disabled)
    assert torch.isfinite(dist.params["means"]).all()
    assert dist.peak_vram_gb() < 14.0  # the Milestone-0 budget ceiling


def test_mcmc_refine_grows_within_cap_and_round_trips(tmp_path: Path) -> None:
    cloud = _synthetic_cloud(n=2000)
    dist = GaussianDistiller(
        cloud,
        device=_DEVICE,
        cap_max_factor=1.5,
        freeze_means_iters=0,
        refine_start_iter=1,
        refine_every=2,
    )
    cam = _camera()
    target = dist.render(cam).rgb.detach().clamp(0.0, 1.0)
    views = [SupervisionView(camera=cam, target_rgb=target.cpu().numpy(), mask=None)]

    dist.fit(views, iters=8)  # exercises relocate/add/noise + param replacement

    n_after = dist.num_gaussians
    assert 2000 <= n_after <= 3000  # within [N, cap_max]
    assert torch.isfinite(dist.params["means"]).all()

    out = tmp_path / "enhanced.ply"
    dist.save_ply(out)
    reloaded = load_gaussian_cloud(out, device="cpu")
    assert reloaded.means.shape[0] == n_after  # round-trips the grown cloud
    assert reloaded.sh_degree == 0
