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

LOCAL PATCHES beyond the import renames (all in ``pipeline_difix.py``, each marked ``LOCAL PATCH``
in-source; ADR-0011, docs/research/artifixer-closeness-24gb.md Steps 2–3):

- ``__call__`` accepts ``init_mask`` (O_z at latent resolution) and, when given, initialises the
  degraded image's latent as the SDEdit opacity-mix ``z_mix = O_z*z_deg + (1-O_z)*eps`` noised to
  the first timestep of the schedule.
- With a multi-step schedule the reference latent is held FIXED (clean encode) across steps
  instead of being co-stepped by the scheduler; single-step output is bit-identical.
"""
