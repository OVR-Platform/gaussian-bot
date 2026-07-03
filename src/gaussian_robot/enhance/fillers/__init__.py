"""Concrete :class:`~gaussian_robot.enhance.protocols.ViewFiller` implementations.

Two interchangeable fillers turn a coverage-gap pose's degraded render into a clean supervision
target for the distiller:

- :class:`GeometricFiller` — classical hole-fill (OpenCV inpaint, numpy blurred-neighbour
  fallback). No weights, no GPU, always available; the safe default.
- :class:`DiffusionFiller` — reference-conditioned Difix generative artifact-fixer with a hard
  recomposite. Single-step at τ=199 by default (the published recipe); opt-in multi-step
  τ-ladder (``num_inference_steps>1``) and SDEdit latent opacity-mixing (``sdedit=True``).
  Weights load lazily on first :meth:`fill`; ``free()`` releases the VRAM.

Both keep heavy/optional dependencies (torch, diffusers, cv2) behind lazy imports so
``import gaussian_robot.enhance.fillers`` stays light.
"""

from gaussian_robot.enhance.fillers.diffusion import DiffusionFiller
from gaussian_robot.enhance.fillers.geometric import GeometricFiller

__all__ = ["DiffusionFiller", "GeometricFiller"]
