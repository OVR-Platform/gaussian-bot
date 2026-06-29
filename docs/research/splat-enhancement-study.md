# Splat Enhancement Study — generative novel-view fill of blurry / under-observed areas

> How to "enhance" a pre-trained 3DGS scene by synthesizing novel views in
> under-observed (low-opacity / blurry) regions and **distilling them back into
> the gaussians**, on a single 24 GB GPU. Inspired in spirit by ArtiFixer
> (arXiv 2603.00492v2), adapted to our stack and budget.
>
> Produced by the `splat-enhance-study` workflow (4 phases, 13 agents: paper
> deep-read · stack map · lightweight-analog lit survey · 24 GB-prior survey →
> 4-architecture judge panel → adversarial VRAM/consistency/integration/license
> verification → synthesis). Findings below were cross-checked against the code.

## TL;DR

- **Adopt the spirit, not the stack.** ArtiFixer's value is one portable idea —
  *opacity as the where-to-enhance signal, generative fill only where coverage
  is low, then distill the fixed views back into the splat*. Its **14B Wan video
  model trained on 128 H100s (~15k GPU-h)** is out of reach and unnecessary.
- **The closest 24 GB-feasible realization already exists: Difix3D+** (NVIDIA,
  CVPR 2025 Oral). Single-step SD-Turbo image "fixer" + progressive distillation,
  with **released weights and native gsplat integration**. It is ArtiFixer's
  recipe at 1/15th the size; it trains "in a few hours on one consumer GPU."
- **Recommended design: `CoverGapFixer`-over-`GeoFill`** — a single masked,
  progressive distill-back loop into our resident `GaussianCloud`, where the
  "fixed view" comes from a **tiered filler that escalates by hole difficulty**:
  geometry first (warp real captured pixels), generative prior only on genuinely
  uncovered pixels. Projected **peak VRAM ≈ 12.5 GB** (ceiling 24 GB).
- **The 3DGS fine-tune loop does not exist yet and is the main thing we build.**
  Our full gaussian parameter set is already resident as differentiable CUDA
  tensors; "densify" in this repo is a *navigation* mode, not gaussian
  optimization. Adding an optimizer is mechanically small but has three exact
  correctness traps (below).
- **Ship in tiers.** `GeoFill` alone (no diffusion weights, Apache-licensed,
  ~11 GB) is a standalone, commercial-clean deliverable and the safety net under
  the learned prior.

---

## 1. What ArtiFixer does, and what we keep

ArtiFixer maps degraded 3DGS renders → clean novel views with a flow-matching
**video** diffusion model, using **opacity mixing** in latent space —
`z_mix = O_z · z_deg + (1 − O_z) · ε` — so well-reconstructed (high-opacity)
pixels are preserved and only uncovered pixels get generative freedom. A causal
auto-regressive student (distilled via DMD) then generates hundreds of
consistent frames; `ArtiFixer3D` distills those views back into the gaussians.

| ArtiFixer component | Fate on 24 GB | What we use instead |
|---|---|---|
| Opacity map `O` as where-to-fill signal | **KEEP** (free) | Our `RenderResult.alpha` (accumulated per-pixel opacity) **is** `O`; `coverage3d.gap_mask`/`gap_centers()` is the 3D dual |
| Opacity mixing `z_mix = O_z·z_deg+(1−O_z)·ε` | **KEEP** (model-agnostic) | SDEdit partial-noise in a small image diffusion latent + a **hard pixel-space recomposite** `out = M·gen + (1−M)·render` |
| Distill generated views into the splat (`ArtiFixer3D`) | **KEEP** — this is the deliverable | A masked gsplat fine-tune loop (Difix3D+'s "progressive 3D update") |
| Wan-14B video backbone, trained on 128 H100s | **DROP** | Single-step SD-Turbo / Difix image fixer (~0.9 B), or geometric warp |
| Causal AR student + DMD (hundreds of frames) | **DROP** | We generate < ~20 views/scene, progressively |
| Learned Plücker camera-control adapters | **DROP** | We *render* the target pose first ("render-then-fix"), so geometry is exact |
| Reference-view cross-attention (trained) | **DROP** | Training-free img2img/IP anchor from nearest capture cameras |
| One universal model across 10k scenes | **DROP** | We operate **per-scene at inference**; the 200k-scene ambition = run the cheap loop 200k times |

**Multiview consistency without the AR model.** A single-image prior has no
cross-view guarantee. The substitute (validated by Difix3D+ being a peer-reviewed
realization of exactly this) is **progressive interleaved distillation**: enhance
a small frontier batch → distill a few hundred iters → re-render (coverage rises)
→ advance the frontier. The splat itself + 50/50 real-anchor co-supervision act
as a hard geometric consensus that averages out / penalizes view-inconsistent
fills.

## 2. Our stack reality (verified against the code)

- **The differentiable substrate is already resident.** `GaussianCloud`
  (`backends/gsplat_renderer.py:24-35`) holds `means/quats/scales/opacities/
  sh_coeffs` as CUDA tensors; `gsplat.rasterization` is already imported
  (`:17`). Rendering returns **RGB + depth + alpha** (`:85`).
- **No gaussian optimizer exists.** Whole-tree grep: no `torch.optim`, no
  `.backward()`, no `requires_grad`, no gsplat `Strategy`. The repo's "densify"
  mode is *navigation* (propose new capture poses), **not** optimization.
  gsplat 1.5.3 *does* expose `DefaultStrategy`/`MCMCStrategy` — unused.
- **Coverage signals (our "O" + where-to-look):** per-pixel `alpha`
  (`render/base.py:36`), `Coverage3D.gap_mask`/`gap_centers()`
  (`metrics/coverage3d.py:57-78`, exposed-surface voxels no camera saw, clipped
  to the material band so it **cannot invent the absent upper floor** — see
  scene memory), `_sharpness` blur metric (`session.py:611`), and
  `quality_floor_coverage` (alpha-thresholded) as a stop criterion.
- **Depth prior:** `DA3DepthEstimator` (Depth Anything 3), lazy/staged — also the
  template for staging a generative model in 24 GB.
- **Scenes are pre-trained 3DGS** (`.ply` + `cameras.json`/COLMAP), loaded never
  trained. **No PLY writer exists** — only the reader.

### Three exact correctness traps (must be handled — verified in-tree)

1. **Activated vs pre-activation storage.** The loader stores **exp'd scales**
   (`:136`) and **sigmoided opacities** (`:139`). An optimizer must wrap the
   **pre-activation** leaves (`log_scale`, `opacity_logit`) and re-apply
   `exp`/`sigmoid` each step. Optimizing the stored activated tensors is a
   silent corruption bug.
2. **`_meta` is discarded** (`gsplat_renderer.py:64`). `MCMCStrategy.
   step_post_backward` requires that info dict — without it the strategy is
   inoperative. A new grad-render path must retain it.
3. **No serializer.** A PLY writer (invert `log`/`logit`, repack `f_dc_*`/
   `f_rest_*` SH layout) must be built and round-trip-tested before any distill.

> A 4th, lower-severity gap: `cameras.json`/COLMAP loaders **discard `img_name`
> and per-camera intrinsics** — needed for `GeoFill` to warp *real* pixels.
> Additive loader work, not a rewrite.

## 3. Recommended design — `CoverGapFixer`-over-`GeoFill` (tiered)

One masked, progressive distill-back loop. The "fixed view" is produced by a
filler that **escalates by hole difficulty**, so the learned prior is spent only
where geometry genuinely can't reach:

- **Tier 0 — hard recomposite (always):** `enhanced = M·fill + (1−M)·render`.
  High-alpha pixels are byte-identical to the real render across *all* views — a
  model-free consistency guarantee that cannot regress good geometry.
- **Tier 1 — geometric warp (default):** reproject **real captured pixels** from
  the 1–3 nearest capture cameras into the target pose using DA3 depth
  (affine-fit to rasterized depth, two-way consistency rejection + soft-z
  dilation). Small residual holes → classical/LaMa inpaint. **No diffusion
  weights.** Structurally consistent; cannot hallucinate never-seen surface.
- **Tier 2 — single-step prior (escalation only):** for genuinely uncovered
  (alpha→0) pixels Tier 1 leaves empty, a **staged** SD-Turbo / Difix fixer with
  opacity-mixing SDEdit. Loaded, run on a frontier batch, then freed.

All tiers feed **one** masked gsplat fine-tune: `L = 0.8·L1 + 0.2·(1−SSIM) +
0.2·LPIPS`, loss masked by feathered `M` on generated views, **unmasked on 50%
real-anchor views** (prevents drift), plus a **mandatory DA3 depth-consistency
term** (suppresses floaters). `MCMCStrategy(cap_max ≈ 1.15·N_current)` bounds
growth. Means frozen for the first ~100 iters (colour-correct before moving
geometry).

### Why not the alternatives (scored 8/8/6/6)

- **OrbitFix3D (video diffusion, CogVideoX-2B) — 6.** Hard-locked to
  720×480 / 49-frame / 8 fps; a 13-frame ~1 MP orbit must be resampled+looped,
  destroying the pixel↔render correspondence "render-then-fix" depends on, and
  paying full 49-frame cost regardless. Temporal coherence ≠ multiview-geometric
  coherence. (CogVideoX-2B's Apache-2.0 license is attractive — keep as an
  optional upgrade, not the baseline.)
- **GapSDS (score distillation) — 6.** Verified hard contradiction:
  `MCMCStrategy._update_param_with_optimizer` *replaces* the full param tensor
  and rebuilds the optimizer group every refine, **destroying any
  `requires_grad=False`/`lr=0` freeze**, and `inject_noise_to_position` perturbs
  all means bypassing `requires_grad`. So its headline "freeze outside radius R +
  subset-Adam" VRAM saving is invalid with MCMC — a redesign, not a tweak.
- **GeoFill — 8.** Most consistent + best license, but cannot invent correct
  unseen structure. → we keep it as **Tier 1 + the fallback**.
- **CoverGapFixer — 8.** Adds the missing hallucination capability via a
  published single-step prior with released gsplat distill-back code; license +
  alpha→0 floater risk are the costs → fenced by tiering, depth loss, anchors,
  progressive frontier.

Grafting the two 8s under one loop is strictly better than either alone:
geometry handles the common case at full fidelity and zero hallucination risk;
the prior fires rarely, only on residual holes.

## 4. Module layout (mirrors the pluggable `depth/` seam)

```
src/gaussian_robot/enhance/
├── protocols.py        # ViewFiller + SplatDistiller runtime_checkable Protocols
├── gap_poses.py        # coverage3d gaps -> progressive frontier look-at poses
├── mask.py             # alpha -> feathered M; max-pool -> O_z (opacity mixing)
├── capture_images.py   # surface img_name + intrinsics + on-disk images (loader ext.)
├── fillers/
│   ├── geometric.py    # Tier-1 GeoFill: depth-warp real pixels + inpaint
│   └── difix.py        # Tier-2: staged SD-Turbo/Difix + opacity-mixing SDEdit
├── distiller.py        # masked gsplat fine-tune (pre-activation params, MCMC, depth loss)
└── orchestrator.py     # detect->sample->fill->stage-out->distill->re-evaluate->advance
# EDITS:
backends/gsplat_renderer.py  # + train_render(): drop no_grad, RETAIN _meta, float tensors
splat/ply_writer.py          # NEW: serialize fine-tuned cloud (invert log/logit, repack SH)
config.py                    # + mode='enhance' + knobs (tau_lo, cap_max_factor, filler, ...)
```

`render()` stays the untouched no_grad uint8 inference protocol; the trainer gets
its own grad-enabled path. Fillers are swappable (geometric ↔ difix ↔ sd2-inpaint)
exactly like depth backends.

## 5. The enhancement loop

1. **Detect** — `build_coverage3d` → `gap_mask = surface & ~seen`, clipped to the
   material band → `gap_centers()`. (Optionally rank by `_sharpness`.)
2. **Sample poses** — for a small frontier batch (4–12 gaps just outside current
   coverage), synthesize 1–3 look-at poses each (`session.look_at` + bounds +
   up-axis); cap poses-per-gap.
3. **Render degraded** — RGB+D+alpha (existing path). `M = smoothstep(alpha <
   tau_lo≈0.5)`, feathered; max-pool → `O_z`.
4. **Fill (tiered)** — Tier-1 warp → residual inpaint → Tier-2 prior only on
   leftover alpha→0 pixels → hard recomposite. Save `(pose, enhanced, M)`. **Free
   the filler.**
5. **Distill (grad-enabled, renderer-only)** — pre-activation params + Adam +
   `MCMCStrategy(cap_max=1.15·N)`, ~300–800 iters. Masked loss on generated
   views, unmasked on 50% anchors, + DA3 depth-consistency. Means frozen first
   ~100 iters.
6. **Re-evaluate** — re-render frontier; alpha rises, `M` shrinks,
   `quality_floor_coverage` up. Stop per-region when covered.
7. **Advance frontier** — next ring of gaps; repeat. When no material-band gaps
   remain, write the cloud via `ply_writer` (round-trip-tested). Optional
   `ArtiFixer3D+` cosmetic post-pass (not distilled).

## 6. VRAM budget (projected, single 24 GB GPU; **peak, after staging**)

| Component | GB | Staged |
|---|---:|:--:|
| Resident `GaussianCloud` (1.5 M gaussians, SH-3, fp32) | 0.36 | — |
| Rasterization forward intermediates (~1 MP) | 2.5 | — |
| DA3-BASE depth (Tier-1 warp + distill depth-loss) | 3.5 | ✓ |
| Difix/SD-Turbo fixer fp16 (Tier-2 only) | 4.0 | ✓ |
| VAE encode/decode spike (Tier-2) | 3.0 | ✓ |
| Distill optimizer state (params+grad+2 Adam moments @1.5 M) | 1.42 | ✓ |
| Grad backward raster intermediates (`_meta` retained, ~1 MP) | 4.0 | ✓ |
| LPIPS(VGG) + MCMC bookkeeping + image batch | 1.0 | ✓ |
| CUDA context + fragmentation headroom | 2.0 | — |

**Peak ≈ 12.5 GB** (Tier-1-only ≈ 11 GB; staged Tier-2 ≈ 12–13 GB). The sole
realistic OOM path is **unbounded MCMC growth** (state scales linearly with N) →
pin `cap_max`. Staging (free the filler before distilling) is mandatory — the
budget is "max over stages," never "sum."

## 7. Milestone-0 and roadmap

**Milestone-0 — identity distill round-trip** (no diffusion, no warp): load the
office `.ply`; implement `train_render()` (+`_meta`) and `ply_writer`;
write→reload→**assert tensor-equality + re-render PSNR > 45 dB** (proves the
pre-activation inversion + SH repack — trap #1); then a tiny masked distill loop
(Adam + `MCMCStrategy(cap_max=1.05·N)`, ~300 iters) supervising held-out renders
against the splat's own re-renders; **assert anchor-PSNR does not regress and
`max_memory_allocated < 14 GB`**. Validates gradient plumbing, `_meta`→MCMC,
pre-activation params, staging, the writer, and the VRAM ceiling with zero
generative risk.

| Phase | Deliverable | Effort |
|---|---|---|
| **P0 Substrate** (candidate-agnostic) | `train_render`+`_meta`; pre-activation params; `ply_writer` round-trip; `mode='enhance'`; Milestone-0 passing with VRAM instrumentation | ~1 wk |
| **P1 GeoFill** (fallback becomes shippable) | `capture_images` loader ext.; `geometric.py` warp; `mask.py`; `gap_poses.py`; full detect→warp→distill→re-eval loop; measurable alpha-rise, zero new floaters | ~2 wk |
| **P2 Difix/SD-Turbo Tier-2** (primary) | env fix (pin `hf-hub<1.0`/bump diffusers; install `peft`); `difix.py` staged filler + opacity-mixing; tiered escalation; A/B vs GeoFill-only on uncovered-pocket sharpness | ~2–3 wk |
| **P3 Hardening + commercial path** | per-scene LoRA (~1–2 h); `ArtiFixer3D+` post-pass; license swap to SD-2-inpaint (OpenRAIL++); per-scene wall-clock scale-test toward 200k; dashboard EventSink | ~2–3 wk |

## 8. Risks & mitigations

- **PLY-writer / pre-activation bug silently corrupts the splat** → Milestone-0
  round-trip gate (PSNR > 45 dB) before any distill.
- **`_meta` discarded → MCMC inoperative** → `train_render()` retains it.
- **Unbounded MCMC = the only realistic OOM** → `cap_max = 1.15·N`, ~1 MP, small
  batch, per-round `max_memory_allocated`.
- **Floaters / hallucinated geometry at alpha→0** (acute for single-floor office)
  → gaps clipped to material-band surface (never empty air); cap poses-per-gap;
  mandatory DA3 depth loss; small per-round hole; 50/50 anchors; Tier-1 carries
  most cases so Tier-2 fires rarely.
- **Multiview inconsistency from a per-image prior** → progressive distillation +
  hard recomposite + anchors (the Difix3D+ consensus mechanism).
- **License blocks the 200k-scene product** (Difix = NVIDIA + SD-Turbo
  research-only) → GeoFill fallback ships commercial-clean (LaMa Apache-2.0, DA3
  already in-tree); Tier-2 backbone swappable to SD-2-inpaint (OpenRAIL++) behind
  the `ViewFiller` protocol.
- **Env clash** (diffusers 0.35.2 + hf-hub 1.4.1 breaks `AutoPipeline`; `peft`
  missing) → bounded P2 install fix; P0/P1 have no such dependency.

## 9. Open questions (the P0/P1 acceptance gates)

Everything here is **projected** — no `.ply`/`cameras.json` is in-tree, so VRAM,
gaussian-count, PSNR and runtime numbers must be measured in Milestone-0.

1. **Does the office scene ship its source capture *images*?** Determines whether
   GeoFill's "warp real pixels" premise holds, i.e. primary vs fallback. (If
   poses+PLY only, Tier-1 degrades to render-of-render warp and Tier-2 becomes
   load-bearing earlier.)
2. **Gap-size distribution of the actual scene** → how often Tier-2 (licensed,
   riskier) actually fires vs Tier-1.
3. **Difix license** vs the proprietary 200k-scene goal — is internal-R&D use
   enough, or target an OpenRAIL/Apache backbone from day one?
4. **`tau_lo`, M feather radius, 50/50 ratio, iters/round, `cap_max` factor** are
   carried from convention — tune on this scene's alpha distribution.
5. **Single-step prior sharpness at `O_z→0`** on indoor office materials is
   unverified — may need per-scene LoRA / LPIPS weight / `ArtiFixer3D+` post-pass.

## References

- ArtiFixer — *Enhancing and Extending 3D Reconstruction with Auto-Regressive
  Diffusion Models* (arXiv 2603.00492v2). The spirit; not the stack.
- **Difix3D+** — NVIDIA, CVPR 2025 Oral. `github.com/nv-tlabs/Difix3D`,
  `huggingface.co/nvidia/difix`. The 24 GB-feasible blueprint (native gsplat).
- Deceptive-3DGS, Instruct-GS2GS / IN2N — iterative-dataset-update distill-back
  mechanics. 3DGS-Enhancer, ReconFusion / CAT3D — heavier/closed references.
- Backbone candidates: SD-Turbo, SD-2-inpaint (OpenRAIL++, commercial-clean),
  SDXL-Lightning; CogVideoX-2B (Apache-2.0) as an optional video upgrade; LaMa
  (Apache-2.0) for Tier-1 inpaint; MoGe-2 as a depth-warp alternative.
