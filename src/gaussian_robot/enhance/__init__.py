"""Splat enhancement: synthesize novel views in under-observed regions and distil them
back into the gaussians (see ``docs/research/splat-enhancement-study.md``).

Milestone-0 ships the substrate: a grad-enabled render path
(:func:`gaussian_robot.backends.gsplat_renderer.rasterize_gaussians`), a PLY writer
(:mod:`gaussian_robot.splat.ply_writer`) and the fine-tune loop
(:class:`gaussian_robot.enhance.distiller.GaussianDistiller`). Coverage masks come from
:mod:`gaussian_robot.enhance.mask`. The gap-driven fillers (geometric warp, diffusion) and
the orchestrator land in later phases behind the protocols in
:mod:`gaussian_robot.enhance.protocols`.

``distiller`` is imported lazily (it pulls torch + gsplat) so ``import gaussian_robot.enhance``
stays light.
"""

from gaussian_robot.enhance.protocols import SplatDistiller, SupervisionView, ViewFiller

__all__ = ["SplatDistiller", "SupervisionView", "ViewFiller"]
