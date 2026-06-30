"""Fine-tune a gaussian cloud to reproduce supervision views — the ArtiFixer3D distill-back.

Milestone-0 scope is the loop itself, wired to the gsplat 1.5.3 ``MCMCStrategy`` contract:

- The parameters are a ``torch.nn.ParameterDict`` over the **pre-activation** leaves
  (``means``, log-``scales``, ``quats``, logit-``opacities``, ``sh``). MCMC's relocate /
  noise math assumes log-scale + logit-opacity, and our loader stores the *activated*
  values, so the distiller inverts them on the way in and re-applies ``exp`` / ``sigmoid``
  on every render and on export. Optimizing the stored activated tensors would be a silent
  corruption bug (study trap #1).
- One ``Adam`` per attribute (the Strategy requires one optimizer / one param-group per
  parameter), and ``MCMCStrategy`` capped at ``cap_max_factor * N`` so VRAM stays bounded.
- The loss is a mask-weighted ``L1`` (only synthesized pixels carry signal) plus a global
  SSIM term; anchor views pass ``mask=None`` for full supervision.

See ``docs/research/splat-enhancement-study.md`` and ``tests/test_enhance_distiller.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from gsplat import MCMCStrategy

from gaussian_robot.backends.gsplat_renderer import (
    GaussianCloud,
    GradRender,
    rasterize_gaussians,
)
from gaussian_robot.enhance.protocols import SupervisionView
from gaussian_robot.render.camera import Camera
from gaussian_robot.splat.ply_writer import write_gaussian_ply

# Standard 3DGS per-attribute learning rates (Kerbl et al. 2023 / gsplat defaults).
_DEFAULT_LRS: dict[str, float] = {
    "means": 1.6e-4,
    "scales": 5e-3,
    "quats": 1e-3,
    "opacities": 5e-2,
    "sh": 2.5e-3,
}


def _ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Global SSIM between two ``(H, W, 3)`` images in ``[0, 1]``; scalar in ``[-1, 1]``."""
    x = pred.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
    y = target.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
    win, c = 11, x.shape[1]
    coords = torch.arange(win, device=x.device, dtype=x.dtype) - win // 2
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).view(1, 1, win, win).expand(c, 1, win, win).contiguous()

    def filt(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, kernel, padding=win // 2, groups=c)

    mu_x, mu_y = filt(x), filt(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx, sy = filt(x * x) - mu_x2, filt(y * y) - mu_y2
    sxy = filt(x * y) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_xy + c1) * (2 * sxy + c2)) / ((mu_x2 + mu_y2 + c1) * (sx + sy + c2))
    return ssim_map.mean()


class GaussianDistiller:
    """Masked photometric fine-tune of a 3DGS cloud with gsplat MCMC densification."""

    def __init__(
        self,
        cloud: GaussianCloud,
        *,
        device: str | torch.device | None = None,
        lrs: dict[str, float] | None = None,
        cap_max_factor: float = 1.15,
        noise_lr: float = 5e5,
        refine_start_iter: int = 100,
        refine_every: int = 100,
        refine_stop_iter: int = 25_000,
        freeze_means_iters: int = 100,
        ssim_weight: float = 0.2,
        densify: bool = True,
    ) -> None:
        dev = torch.device(device) if device is not None else cloud.means.device
        self.device = dev
        self.sh_degree = cloud.sh_degree
        self.ssim_weight = ssim_weight
        self.freeze_means_iters = freeze_means_iters
        # densify=False -> pure Adam fine-tune: fixed N, NO MCMC relocate/add/position-noise.
        # This is the safe mode for a localized, anchored fine-tune (see the Milestone-0
        # post-mortem: unconstrained global MCMC + thin supervision scatters floaters).
        self.densify = densify
        # Optional (N,) bool mask of gaussians inside coverage gaps. When set, FILL steps
        # (masked views) update only these gaussians' colour/opacity, so generated content
        # cannot bleed onto the gaussians anchor views trust. Set externally per round.
        self.gap_index: torch.Tensor | None = None
        # Gradient gain applied to gap gaussians on FILL steps (>1 = push fills harder). Safe to
        # exceed the global LR because gap_index confines it to gap gaussians; anchors are untouched.
        self.fill_gap_gain: float = 1.0
        self._cloud_meta = cloud  # reused for bounds on export

        means = cloud.means.to(dev).float()
        quats = cloud.quats.to(dev).float()
        scales_log = torch.log(cloud.scales.to(dev).float().clamp_min(1e-6))
        opac_logit = torch.logit(cloud.opacities.to(dev).float().clamp(1e-6, 1.0 - 1e-6))
        sh = cloud.sh_coeffs.to(dev).float()

        self.params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(means),
                "scales": torch.nn.Parameter(scales_log),
                "quats": torch.nn.Parameter(quats),
                "opacities": torch.nn.Parameter(opac_logit),
                "sh": torch.nn.Parameter(sh),
            }
        )
        rates = lrs or _DEFAULT_LRS
        self.optimizers: dict[str, torch.optim.Optimizer] = {
            key: torch.optim.Adam([self.params[key]], lr=rates[key], eps=1e-15)
            for key in self.params
        }
        n = int(means.shape[0])
        self.strategy = MCMCStrategy(
            cap_max=max(n, int(round(n * cap_max_factor))),
            noise_lr=noise_lr,
            refine_start_iter=refine_start_iter,
            refine_every=refine_every,
            refine_stop_iter=refine_stop_iter,
        )
        self.state = self.strategy.initialize_state()
        self.strategy.check_sanity(self.params, self.optimizers)

    @property
    def num_gaussians(self) -> int:
        return int(self.params["means"].shape[0])

    def state_snapshot(self) -> dict[str, torch.Tensor]:
        """Clone the live parameters to CPU (cheap insurance for a progressive-round rollback)."""
        return {k: v.detach().to("cpu", copy=True) for k, v in self.params.items()}

    def load_snapshot(self, snap: dict[str, torch.Tensor]) -> None:
        """Restore parameters from :meth:`state_snapshot` (in-place; optimizer state is left as-is).

        Intended for *terminal* rollback (revert to the best-scoring round, then stop) — not for
        resuming optimization, where the stale Adam moments would no longer match the parameters.
        """
        with torch.no_grad():
            for k, v in self.params.items():
                v.copy_(snap[k].to(v.device))

    def render(self, camera: Camera) -> GradRender:
        """Differentiable render from the current (live) parameters."""
        return rasterize_gaussians(
            self.params["means"],
            self.params["quats"],
            torch.exp(self.params["scales"]),
            torch.sigmoid(self.params["opacities"]),
            self.params["sh"],
            self.sh_degree,
            camera,
        )

    def _loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
        weight: float = 1.0,
    ) -> torch.Tensor:
        l1_map = (pred - target).abs()
        if mask is not None:
            m = mask.unsqueeze(-1)
            l1 = (l1_map * m).sum() / (m.sum() * 3.0 + 1e-6)
        else:
            l1 = l1_map.mean()
        loss = (1.0 - self.ssim_weight) * l1
        if self.ssim_weight > 0.0:
            # On FILL views the recomposite makes trusted pixels identical, so a full-frame SSIM
            # is ~1 and contributes nothing; mask pred/target to the hole so SSIM supervises the
            # synthesized structure. Anchor views (mask=None) take full-frame SSIM.
            if mask is not None:
                mm = mask.unsqueeze(-1)
                loss = loss + self.ssim_weight * (1.0 - _ssim(pred * mm, target * mm))
            else:
                loss = loss + self.ssim_weight * (1.0 - _ssim(pred, target))
        # NOTE: `weight` scales this view's gradient, but the per-view Adam step below is
        # scale-invariant, so a scalar weight is near-inert on its own (see SupervisionView.weight).
        return weight * loss

    def fit(self, views: Sequence[SupervisionView], iters: int, *, step_offset: int = 0) -> None:
        """Run ``iters`` masked-photometric steps, cycling through ``views``.

        Means are frozen (no Adam step, zero MCMC noise) for the first
        ``freeze_means_iters`` steps so colour/opacity settle before geometry moves.

        ``step_offset`` makes the freeze + MCMC-refine scheduling continuous across many short
        ``fit`` calls: the progressive scheme calls ``fit`` once per pose-step, and without a
        running offset every call would re-freeze means and restart MCMC's refine clock at 0.
        """
        if not views:
            raise ValueError("need at least one supervision view")
        for local in range(iters):
            step = local + step_offset
            view = views[local % len(views)]
            grad_render = self.render(view.camera)
            target = torch.as_tensor(view.target_rgb, dtype=torch.float32, device=self.device)
            mask = (
                None
                if view.mask is None
                else torch.as_tensor(view.mask, dtype=torch.float32, device=self.device)
            )
            loss = self._loss(grad_render.rgb, target, mask, float(view.weight))

            frozen = step < self.freeze_means_iters
            loss.backward()  # type: ignore[no-untyped-call]
            # On FILL steps, confine colour/opacity updates to gap gaussians so generated
            # content does not perturb the well-observed gaussians anchors trust. NOTE: only sh +
            # opacities are masked; means/scales/quats are NOT, so if a caller ever unfreezes
            # geometry together with gap-restriction, fill views could still move non-gap geometry.
            if mask is not None and self.gap_index is not None:
                gi = (self.gap_index.to(self.device).float() * self.fill_gap_gain)
                for pname in ("sh", "opacities"):
                    g = self.params[pname].grad
                    if g is not None:
                        g.mul_(gi.view(-1, *([1] * (g.dim() - 1))))
            for name, opt in self.optimizers.items():
                if frozen and name == "means":
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()
                opt.zero_grad(set_to_none=True)
            if self.densify:
                means_lr = 0.0 if frozen else float(self.optimizers["means"].param_groups[0]["lr"])
                # MCMC reads nothing from `info`; passing it keeps the Strategy API satisfied
                # and ready for a DefaultStrategy swap (which consumes means2d gradients).
                self.strategy.step_post_backward(
                    self.params, self.optimizers, self.state, step, grad_render.info, lr=means_lr
                )

    def to_cloud(self) -> GaussianCloud:
        """Export the fine-tuned parameters as an (activated) :class:`GaussianCloud`.

        ``density_grid`` is dropped (densification may have changed N, staling the grid);
        callers that need it should rebuild from the means.
        """
        with torch.no_grad():
            return GaussianCloud(
                means=self.params["means"].detach().clone(),
                quats=F.normalize(self.params["quats"].detach(), dim=1),
                scales=torch.exp(self.params["scales"].detach()),
                opacities=torch.sigmoid(self.params["opacities"].detach()),
                sh_coeffs=self.params["sh"].detach().clone(),
                sh_degree=self.sh_degree,
                bounds=self._cloud_meta.bounds,
                full_bounds=self._cloud_meta.full_bounds,
                density_grid=None,
                density_bounds=self._cloud_meta.density_bounds,
            )

    def save_ply(self, path: str | Path) -> Path:
        """Write the fine-tuned cloud to a 3DGS PLY readable by ``load_gaussian_cloud``."""
        cloud = self.to_cloud()
        return write_gaussian_ply(
            path, cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.sh_coeffs
        )

    @staticmethod
    def reset_peak_vram() -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def peak_vram_gb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.max_memory_allocated()) / (1024.0**3)
