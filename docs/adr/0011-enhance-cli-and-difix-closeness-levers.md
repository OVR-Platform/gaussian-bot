# 0011. One supported enhance CLI + Difix closeness levers

- Status: accepted
- Date: 2026-07-03

## Context

The splat-enhancement subsystem (`enhance/`: fillers, orchestrator, distiller,
explore-fill, before/after) grew through three research scripts
(`enhance_scene.py`, `fill_gaps.py`, `explore_and_fill.py`), each a different
entry into the same machinery. The research line
(`docs/research/artifixer-closeness-24gb.md`, the current actionable plan)
identified the cheapest "closeness to ArtiFixer" levers as changes to the
vendored `difix_ref` backbone we already ship — fp16, a multi-step denoise
schedule, and the SDEdit opacity-mix latent init that `mask.downscale_to_latent`
was written for but nothing consumed. Meanwhile the CLI had no enhance command
at all, and `RunConfig` carried an `"enhance"` session mode plus six `enhance_*`
knobs that nothing consumed (mode `"enhance"` silently ran as densify).

## Decision

1. **One supported path.** `uv run gaussian-robot enhance <scene.ply> --out
   <new.ply> --colmap <sparse/0> --images <dir>` wraps
   `enhance.explore_fill.explore_and_fill`: the robot explores (densify
   session), marks under-observed viewpoints, the reference-conditioned Difix
   filler cleans them, the distiller writes a **NEW** ply. It emits a
   before/after GIF, prints the held-out Δ-PSNR guard and the fill-phase peak
   VRAM, and can dump a JSON report. The default is the **rounds** fill
   (frozen geometry, gentle LRs) — the measured-safe recipe the research
   validated; `--progressive` opts into the faithful Difix3D+ loop (geometry
   moves + densify), which measured an immediate ≥0.9 dB held-out regression
   on the office scene (the gate reverts it to a no-op there). The three
   research scripts are marked superseded and kept for reference;
   `milestone0_identity_distill.py` stays as the plumbing gate.
2. **Closeness levers on the vendored backbone.**
   - **fp16 is the CLI default** (`--dtype float16`), halving the measured
     8.4 GB fp32 forward peak; `--dtype float32` restores NVIDIA's reference
     precision.
   - **Multi-step denoise** (`--denoise-steps N`): `DiffusionFiller` walks a
     descending τ-ladder `τ·(N-i)/N` (N=1 keeps the published single-step at
     τ=199, bit-identical).
   - **SDEdit opacity-mix init** (`--sdedit`): `z_mix = O_z·z_deg + (1-O_z)·ε`
     with `O_z` max-pooled to latent resolution via `downscale_to_latent`,
     noised to the top of the ladder. Guarded: requires `--denoise-steps ≥ 2`
     (a single fixed-τ step cannot denoise a noised init — it only degrades).
     The hard pixel-space recomposite `M·gen + (1-M)·render` stays regardless.
3. **Vendored-file deviation.** `_difix/pipeline_difix.py` is no longer
   verbatim: it carries two local patches (an `init_mask` kwarg implementing
   the opacity-mix init; the reference latent held fixed across multi-step
   schedules), each marked `LOCAL PATCH` in-source and listed in
   `_difix/__init__.py`. Single-step output is bit-identical.
4. **Config cleanup.** The dead `enhance_*` `RunConfig` fields and the
   `"enhance"` session mode are removed; session modes are `densify` and
   `task`. Enhancement is a CLI concern, not a session mode.

## Consequences

- One documented, testable command; the CLI is covered by CPU tests (guards,
  flag plumbing, τ-ladder, O_z pooling) and the existing GPU e2e orchestrator
  tests.
- Defaults stay the *measured-safe* configuration: single-step, fp16, gate on
  (`regression_tol_db=0.3`). Multi-step and SDEdit are opt-in because their
  gains in empty regions are perceptual-only (no ground truth there) and must
  be gated on held-out PSNR per scene.
- Honest regime: the fill pays on SPARSE captures; on dense well-covered
  scenes it measures ~0 gain (see `docs/research/README.md`). The CLI help
  says so.
- Upgrading the vendored files now requires re-applying the two local patches
  (they are small and marked).
