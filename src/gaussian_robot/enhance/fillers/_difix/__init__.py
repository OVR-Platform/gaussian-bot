"""Vendored Difix3D+ pipeline (NVIDIA, CVPR 2025) — the official reference-conditioned filler.

These three modules are copied verbatim from ``nv-tlabs/Difix3D`` (``src/pipeline_difix.py``) and
the ``nvidia/difix_ref`` Hub repo (``unet/unet_2d_condition.py``, ``vae/autoencoder_kl.py``), with
ONLY the diffusers 0.25→0.38 import renames patched in-source (``FromOriginalVAEMixin`` →
``FromOriginalModelMixin``; ``PositionNet`` → ``GLIGENTextBoundingboxProjection``;
``diffusers.models.unet_2d_blocks`` → ``diffusers.models.unets.unet_2d_blocks``; the custom VAE now
inherits ``PeftAdapterMixin`` so ``add_adapter`` resolves).

Vendoring (vs ``trust_remote_code``) keeps the filler reproducible and pinned — no network fetch or
remote-code execution at load time. This is third-party code: it is excluded from ruff/mypy. The
typed loader that assembles these into a pipeline lives in
:func:`gaussian_robot.enhance.fillers.diffusion.load_difix_pipeline`.
"""
