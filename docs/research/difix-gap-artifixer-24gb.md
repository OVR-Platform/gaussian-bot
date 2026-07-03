# Closing the Difix3D+ gap & ArtiFixer at 24 GB

*Research + design synthesis. Produced by the `difix-gap-and-artifixer-24gb` workflow (27 agents,
adversarial VRAM verification) + manual consolidation. No code changed by this pass — it decides
what is worth implementing.*

---

## TL;DR — recommendation

**Do Plan A first.** Our pipeline already implements the *exact* Difix3D+ reference-conditioned
inner loop (`nvidia/difix_ref`, single-step, τ=199, gs=0.0, "remove degradation", nearest clean
photo as `ref_image`, hard recomposite, progressive dolly, distill-back). **The fill-quality gap is
entirely on the 3DGS supervision side**, and most of it closes with ~zero-risk supervision fixes
that need no new model and no new VRAM:

1. **Gap-restrict the SH/opacity updates** (highest leverage, low risk).
2. **0.7 / 0.3 anchor-vs-fill source reweighting** (we do a flat 50/50).
3. **Masked SSIM on the fills** (we force `ssim_weight=0`).
4. **Recompute gaps per round** on the current cloud (we freeze them once on the original).
5. **Angular nearest-camera selection** (360 rig: translation-only argmin can pick a camera facing away).

**Treat Plan B (ArtiFixer-at-24 GB via an alternative diffusion backbone) as the *next-regime*
experiment, not a drop-in upgrade.** The best 24 GB-feasible surrogate is **SEVA / Stable Virtual
Camera (1.3 B)**, but the adversarial verdict is blunt: in our *current* regime (small holes near
training cameras, geometry frozen) **SEVA does not beat `difix_ref`** — it is slower, heavier,
NonCommercial, and unvalidated on 24 GB. SEVA only wins once we (a) target gaps far from any
training camera, and (b) let the distiller move/add geometry so multi-view-consistent generations
can build new surface. ArtiFixer's other two pillars (DMD few-step distillation, causal-AR
rolling-KV student) should be **dropped** at our scale (<20 views/scene).

**The no-regression guarantee stays sacred.** Frozen geometry / `densify=False` / gentle LRs were
*measured* safeguards (full-opt → −1.5 dB, opacity 2e-2 → −4.8 dB held-out). Every aggressive change
below ships **off by default behind the per-round `regression_tol_db` gate**.

---

## 1. The gap vs the official Difix3D+

Both pipelines run the **same** diffusion inner loop. Divergences cluster in the **distillation
half** and split cleanly into *cheap pure wins* and *deliberate safeguards*.

| Aspect | Official Difix3D+ | Ours | Verdict |
|---|---|---|---|
| **SH/opacity update scope** | masked per-frame supervision, pinned by full train set over 29 rounds | mask gates pixel **loss** but a **global** SH leaf updates — fill colour bleeds onto trusted gaussians | **FIX (cheap, high leverage)** |
| **Source mixing** | 0.7 real / 0.3 novel + `novel_data_lambda=0.3` (~5:1 real bias) | hard 50/50 round-robin, no scaling | **FIX (cheap)** |
| **Loss** | 0.8·L1 + 0.2·(1−SSIM) incl. fills | L1 only (`ssim_weight=0`) | **FIX (cheap)** |
| **Gap set** | novel poses creep toward targets; extent grows each round | gaps frozen once on the *original* cloud | **FIX (cheap)** |
| **Reference selection** | translation **+ angular** pose distance | translation-only argmin | **FIX (cheap)** — matters on a 360 rig |
| **Real-time enhancer** | Difix also cleans the final render (~76 ms A100) | not used (we ship a PLY, render raw) | **OPTIONAL** (no regression risk; touches no geometry) |
| **Geometry LRs** | all params optimized (means 1.6e-5·scale, scales 1e-3, quats 2e-4) over 60k iters | **frozen** (means/scales/quats LR=0) | **KEEP FROZEN** — matching it = measured −1.5 dB on our thin anchor set |
| **Densification** | `DefaultStrategy` clone/split for 15k iters, driven by fixed novel views | **off** | **KEEP OFF** by default; add a *narrow gap-local* opt-in |
| **Opacity LR** | 1e-2 | 5e-3 | raise **only** behind the gate, after fix #1 |

**The one structural limit:** our `coverage3d` defines a gap as **occupied-but-unseen** (gaussians
must already be present). So frozen-opacity fill is internally consistent for *occluded existing
surface* — but it **structurally cannot fill a truly empty voxel** (no gaussians to raise). That is
exactly the regime the official global densify covers. On a **single-floor interior** this is mostly
benign (gaps are occluded surface, not empty volume), so the recommendation is: ship the cheap
supervision fixes, keep geometry frozen / densify off as the default, and expose **one narrow
gap-local densify opt-in** for the rare under-populated voxel.

---

## 2. Plan A — close the gap, keep the guarantee (ranked)

**Pure wins (do these — low risk, no new VRAM, no new deps):**

1. **Gap-restrict SH + opacity gradients.** Build a `(N,)` bool mask of gaussians inside gap voxels
   (`coverage3d` voxel index ∩ `gap_mask`); on *fill* steps (`mask is not None`), zero the
   `sh.grad`/`opacities.grad` outside it before `opt.step()`. Anchor steps keep full gradients.
   *Why:* today the mask gates pixel loss but not *which* gaussians update, so generated colour
   bleeds onto trusted gaussians — the likely cause of held-out drift and the reason SH LR had to
   sit 10× below default. This is the clean analogue of "gaps drive their own gaussians" without
   densify, and it **unlocks safely raising the LRs**. *(distiller.py `fit`, orchestrator passes the index.)*
2. **0.7/0.3 source reweighting.** Tag each `SupervisionView` (fill vs anchor) and scale loss
   (fill ×0.3, anchor ×1.5), optionally replace the 50/50 `_interleave` with weighted sampling.
3. **Masked SSIM on fills.** Pass `ssim_weight≈0.1`, applied to the hole region of fill views only.
4. **Per-round gap recompute.** Move `build_coverage3d` into `_progressive_distill` so later (larger-
   dolly) rounds target still-degraded regions and discover newly exposed gaps instead of re-fixing
   solved voxels. Optionally *retain* prior rounds' fill supervision (mirror the official growing set).
5. **Angular nearest-camera term** in `synthesize_near_view_poses` (translation + rotation distance),
   so the Difix reference actually overlaps the gap view on the 360 rig.
6. *(Optional)* **Post-render Difix enhancer** — reuse `DiffusionFiller.fill` on the final rendered
   novel view; additive, never touches the stored PLY, ~zero regression risk.

**Guarded opt-ins (ship OFF, gate on `regression_tol_db`):**

7. **Narrow gap-local densify** — seed a handful of gaussians *only* inside under-populated gap
   voxels (duplicate nearest gap gaussians or splat-from-depth at filled poses), accept only if the
   PSNR gate holds. The *only* way to fill a truly empty hole; keep off on single-floor interiors.
8. **Raise per-gap SH/opacity LRs** toward official (sh→2.5e-3, opacity→1e-2) — safe **only after
   fix #1**, applied to gap gaussians, gated.

**Do NOT match (keep as-is):** full geometry optimization, global `DefaultStrategy` densify, the
dolly-toward-gap pose strategy (the official "shift toward val cams" is undefined here — no held-out
cameras aim at the gaps, and the 360 rig's ~0 inter-camera spacing would disable a spacing-scaled
shift), and the `[0,0.9]` perturb clamp (pushing fully toward the gap reproduces the abandoned
full-frame-smear failure).

**VRAM:** unchanged — Difix staged-then-freed (~8.4 GB peak) + fixed-N distiller. The fixes add only
a bool index and per-view scalars.

---

## 3. Plan B — ArtiFixer-style at 24 GB

ArtiFixer = Wan2.1-T2V-**14B** (16.9 B total, A100-80 GB) → infeasible. Its transferable mechanisms:

- **Opacity-mixing latent init (SDEdit-like):** encode the rendered RGB to latent, mix with Gaussian
  noise weighted by the rendered **opacity map** (covered tokens stay faithful, empty tokens become
  fully generative). **Backbone-agnostic and near-free** — our `mask.downscale_to_latent` already
  implements the max-pool-to-latent step and is currently unused. **Replicate this.**
- **Camera control** via learned Plücker raymaps → we substitute "render the target pose ourselves
  and hand it over" (the render-then-fix seam we already use).
- **Bidirectional video teacher + causal-AR DMD-distilled student** → **drop both**: they exist to
  stream hundreds of consistent frames cheaply; we generate <20 views/scene and a chunked sampler
  already bounds VRAM.

**Best 24 GB backbone: SEVA / Stable Virtual Camera (1.3 B).** It natively does the one thing
`difix_ref` structurally lacks — **multi-view-consistent full novel views at arbitrary target
cameras in one pass** — and slots into the existing `ViewFiller` seam unchanged (distiller, coverage
signal, pose synth, gate all reused). Staged plan: (1) **VRAM smoke test first** (fp16, T≤21,
576², 50 steps — confirm ≤~18 GB; it was never validated <48 GB); (2) `SevaFiller` implementing
`fill(degraded, references)` with the existing recomposite; (3) correct pose-convention adapter
(ADR-0002 world→cam, OpenCV axes) + round-trip unit test; (4) per-gap **mini-orbit** (3–8 consistent
target poses in one chunk) — the actual reason to use SEVA; (5) optional opacity-mix latent init;
(6) A/B vs `difix_ref` through the existing held-out gate.

**Frank verdict:** ship `difix_ref` as default; SEVA is the experiment that unlocks large/far gaps +
moving geometry, **not** a win in today's small-hole/frozen-geometry regime. Caveats: SEVA is
**NonCommercial** (blocks productization), has **no native mask** (all opacity-awareness lives in our
recomposite — mask quality becomes load-bearing), and per-scene wall-clock rises ~10× (matters for
the 200k-scene ambition → argues for a tiered geometric → difix → SEVA-only-on-hard-gaps policy).

---

## 4. 24 GB feasibility — model survey (adversarially verified)

"fits" = inference on a single RTX 4090. "rec" = recommended for *our* posed, mask-aware gap-fill.

| Model | params | fits 24 GB | rec | conditioning / note |
|---|---|---|---|---|
| **SEVA (Stable Virtual Camera)** | 1.3 B | marginal (fp16, T≤21) | **YES** | ref-view + **arbitrary target camera**, multi-view-consistent; **no mask**; NonCommercial |
| CameraCtrl | adapter + SVD/AnimateDiff ~1.5 B | yes | ~no | **TRUE Plücker camera pose** + I2V (SVD branch), **Apache-2.0**; but SD1.5/SVD-era low fidelity, no mask |
| CogVideoX-5B-I2V | ~5 B | yes (~15 GB bf16) | partial | I2V ref **released**; camera-pose + mask **research-only** (CamPilot/FrameCrafter ControlNet); commercial-OK |
| CogVideoX-2B | 2 B | yes | no | text-to-video only as shipped (no ref-image) |
| LTX-Video (2B) | 2 B + T5-XXL | yes (~11–14 GB) | no | I2V + depth/canny/OpenPose; **no camera extrinsics, no mask**; OpenRAIL-M (commercial) |
| Wan2.1-T2V-1.3B | 1.3 B | yes | no | text-to-video only |
| AnimateDiff (+MotionLoRA) | ~1.4 B | yes | no | no faithful pose, no mask |
| MotionCtrl | adapter + SVD | yes | no | camera *motion* control, weak scene fidelity |
| GenWarp | undisclosed | yes | no | warp-and-inpaint; mismatched granularity vs our seam |
| CamI2V | ~1.66 B | yes | no | ref + camera, but poor fit as scoped |
| SVD-XT | ~1.5 B | marginal (offload) | no | image-only, implicit motion bucket — **no posed control, no mask** |
| SVD (base) | ~1.5 B | marginal (offload) | no | image-only — no pose, no mask |
| SV3D | ~1.5 B | marginal | no | fixed orbit only |
| ViewCrafter | ~1.4 B | marginal | no | point-cloud-conditioned; granularity mismatch |
| Wan2.2-TI2V-5B | 5 B | marginal | no | first-frame I2V only, weak posed control |
| Mochi-1 (preview) | 10 B | marginal | no | text-to-video only |

**Pattern:** the small posed models split into "true camera control but low fidelity / no mask"
(CameraCtrl, MotionCtrl, CamI2V) and "high fidelity but no posed/mask control" (SVD/LTX/CogVideoX as
shipped). **SEVA is the only ≤24 GB option that combines posed + multi-view-consistent full-view
generation** — hence Plan B picks it, with the honest caveat that it only pays off in a regime we're
not in yet.

---

## 5. Staged roadmap

1. **Plan A pure wins (#1–#5)** — implement behind the existing gate; A/B on the office scene. Expect
   a *modest but real* held-out gain over the current +0.32 dB, with sharper fills and no regression.
   *(This is the recommended next implementation step.)*
2. **Plan A optional #6** (post-render enhancer) for far-from-training views.
3. **Guarded opt-ins #7–#8** only if a scene shows truly-empty gaps or under-filled holes.
4. **Plan B SEVA experiment** — VRAM smoke test → `SevaFiller` → mini-orbit → A/B. Gate the whole
   thing on the smoke test; abandon if fp16 SEVA exceeds ~18–20 GB at our resolution.
5. **Opacity-mix latent init** (ArtiFixer's near-free idea) wherever the chosen backbone exposes a
   settable init latent.

---

## 6. Open questions / self-critique

*(The workflow's completeness-critic agent was one of the 5 that failed; this section folds in its role.)*

- **Plan A #1 needs `fit()` to distinguish fill vs anchor steps.** `mask is not None` is the clean
  signal today, but is fragile if a future anchor path sets a mask — prefer an explicit `source`
  field on `SupervisionView`.
- **Gap-gaussian index is coarse** at `grid=32` (voxel halo includes some trusted gaussians, diluting
  the restriction). Tunable, costs coverage3d time not VRAM.
- **The held-out gate samples `eval_stride` views** — if none sit near the gaps, a *local* regression
  could pass. Watch per-eval PSNR, not just the mean.
- **SEVA's 24 GB fit is unmeasured** (estimate leans on the SVD same-family anchor). Step 1 is a hard
  gate, not a formality.
- **Not deeply explored:** training a small **camera+mask ControlNet on CogVideoX-5B** (fits 24 GB,
  commercial-OK) — a heavier but commercially-clean alternative to NonCommercial SEVA if
  productization matters. Worth a dedicated feasibility pass before committing to SEVA.

---

## Update — Plan A measured on the office scene (negative result)

All five Plan A changes were implemented and measured individually against the +0.318 dB held-out /
+0.019 hole-coverage baseline. **None improved this scene; the baseline is best.** The fixes are
shipped as **opt-in knobs, defaulting to neutral**, so the default reproduces the baseline exactly
(+0.318 dB / +0.019). Why each didn't transfer to a single-floor interior with small near-view holes:

| Change | Knob (default) | Measured effect | Why |
|---|---|---|---|
| #1 gap-restrict SH/opacity | `restrict_to_gaps=False` | coverage +0.019 → **−0.003** | The gap-VOXEL gaussians (9.9% of the cloud) are **not** the gaussians that compose the rendered hole at the near-view pose, so restricting to them **zeros the fill gradient** (gain 1/4/8 were bit-identical — multiplying ~0). Mis-targeted for this geometry. |
| #2 fill reweighting | `fill_weight=1.0`, `anchor_weight=1.0` | **inert** (no-op) | Adversarial review: a scalar per-view loss weight is **scale-invariant under the per-view Adam step** (verified byte-identical), so the 0.3/1.5 reweighting does ~nothing on its own. The −0.003 in the #1 run was **entirely #1**, not #2. To truly bias real>fills, vary sampling frequency / accumulate gradients — not implemented (no payoff here). |
| #3 masked SSIM on fills | `ssim_weight=0.0` | slightly lower coverage | Trades L1 fill magnitude for structure; on a conservative Difix cleanup of small holes that's a net loss. |
| #4 per-round gap recompute | always on | inert here | Round 0 (closest dolly) wins the best-snapshot selection, so recomputing gaps for the discarded later rounds doesn't change the shipped result. Kept on (more correct; helps when a later round wins). |
| #5 angular reference | `angular_weight=0.0` | coverage +0.019 → +0.011 | Picks a more frontal reference for some gaps → a slightly different (less coverage-raising) Difix fill. The translation-nearest reference is better here. |

**Bottom line:** the report's Plan A was derived from the official 60k-iter / full-train-set Difix3D+
recipe; those supervision fixes assume a regime (larger holes, geometry allowed to move, gap-voxel ≈
hole gaussians) that the single-floor interior is not in. The machinery is correct and retained as
gated knobs (`restrict_to_gaps`, `fill_weight`/`anchor_weight`, `fill_gap_gain`, `ssim_weight`,
`angular_weight`) for scenes that *are* in that regime; on this scene the honest best remains the
prior baseline. The genuine next lever for a stronger fill here is **Plan B's larger/far-gap regime
+ allowing geometry to move** (where these knobs start paying off), not more supervision tweaks.

## Appendix — workflow run notes

Run `wf_6e768fbe-cba`: 27 agents, ~1.2 M subagent tokens, ~28 min. **5 agents failed**, all from one
cause: strict `StructuredOutput` schemas with oversized multi-line text fields → JSON serialization
failed, and the shorter retries dropped required fields → retry cap exceeded → `null` (filtered out).
Affected: 4 verify verdicts (CogVideoX-5B-I2V, SVD-base, CameraCtrl, LTX-Video) + the completeness
critic. The 4 verdicts were **re-run manually with plain-text output** (folded into §4); the critic's
role is folded into §6. **Durable fix applied** to `.claude/workflows/difix-gap-and-artifixer-24gb.js`:
`VERDICT_SCHEMA` `required` reduced to `name/fits_24gb/recommended` and the verify prompt now forbids
literal newlines/backslashes and caps each field at ~80 words, so a compact retry always validates.
