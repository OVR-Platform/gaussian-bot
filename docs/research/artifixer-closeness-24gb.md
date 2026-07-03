# Getting as close as possible to ArtiFixer on 24 GB VRAM

*Decision doc for the gaussian-bot splat-enhancement team. Question answered literally: "if I wanted
to get as close as possible to ArtiFixer with 24 GB, what can I do?" This is the per-pillar +
per-backbone scoping that turns the prior survey (`difix-gap-artifixer-24gb.md`) into a build plan
wired to our actual repo seams. No code changed by this pass.*

**On the numbers below.** The only VRAM figure measured in this repo is the difix_ref filler's probe
(5.2 GB to load, 8.4 GB peak per forward — diffusion.py:21, from `experiments/filler_probe/`).
**Every other GB figure here is either taken from the prior survey or marked as an unverified
estimate that requires a smoke test before it can be trusted.** Where this doc previously asserted
backbone footprints as fact, those have been re-sourced to the survey or flagged. Treat any number
not traceable to a measurement or the survey as a hypothesis to validate, not a spec.

---

## 1. TL;DR — recommended path

**Spend your cheapest closeness levers on the difix_ref backbone you already ship before standing up
any new model.** The `ViewFiller.fill` contract is *one degraded image in, one `SupervisionView`
out* (protocols.py:53-55) — a single-image-per-call shape — so a video DiT is arguably the wrong
tool, and the highest-ROI moves are all on the vendored single-step Difix:

1. **fp16 difix** — the filler already accepts `dtype="float16"` (diffusion.py:159); it roughly
   halves the measured 8.4 GB peak for free. Cheapest headroom we have.
2. **Multi-step difix** — `__call__` already accepts `num_inference_steps`/`timesteps` and
   `retrieve_timesteps` supports a multi-step schedule (pipeline_difix.py:839,1021). Running N>1
   steps from a noised init is the *actual* way to get a real denoise trajectory on the backbone
   already in the repo — and it is the prerequisite for any honest opacity-mix / SDEdit init.
3. **Single-image posed inpainter** (FLUX/SD3.5 + a depth-or-pose ControlNet, or a multi-view image
   diffusion like Zero123++/MVDream) as the camera-conditioned filler that *matches our one-image-in
   contract* — lighter than CogVideoX/Wan and the unexamined option the prior survey skipped.

**If you do reach for a new backbone, the prior survey's pick stands: SEVA / Stable Virtual Camera
(1.3 B) is the best 24 GB surrogate** (difix-gap-artifixer-24gb.md:120-134) — and even SEVA "does not
beat difix_ref" in our current small-hole/frozen-geometry regime (survey lines 26-30). A
Wan2.1-T2V-1.3B stack is a *plausible alternative*, not a proven upgrade: base Wan-1.3B ships as
text-to-video only with no camera control and no mask (survey line 149); the ArtiFixer-shaped
capability would come entirely from separate, unintegrated add-ons (Fun-Camera adapter, Self-Forcing
student) whose wiring is unbuilt and unmeasured here. **The honest near-term win is fp16 + multi-step
difix with a real opacity-mix init — not a new video model.** The one thing 24 GB genuinely cannot
buy is the 14 B teacher's fidelity when *inventing content in truly empty, far-from-camera volume* —
the regime ArtiFixer's scale was built for, and one our current single-floor interior rarely hits.

---

## 2. The 3 closeness tiers

VRAM cells distinguish **[survey]** (from difix-gap-artifixer-24gb.md), **[measured]** (this repo),
and **[estimate — unverified]** (needs a smoke test). Do not treat estimates as commitments.

| Tier | Route | How close to real ArtiFixer | Fits 24 GB (VRAM basis) | Speed / scene | License | Build effort |
|------|-------|------------------------------|--------------------------|---------------|---------|--------------|
| **0** | Run the real `nvidia/ArtiFixer` (Wan2.1-T2V-14B) self-quantized + CPU offload + block-swap + clip ≤21 | **Highest in principle — it *is* ArtiFixer**, but see closeness ceiling: effectively **~0% recoverable at 24 GB** | **Out of scope at 24 GB.** The survey states only "14B (16.9 B total)" with no fp32 ckpt size and no quant arithmetic (survey:108); the 67.6/33.8 GB chain and the "+2.9 B custom-module" delta in earlier drafts were **unsourced** and are withdrawn. Modern 4-bit DiT quantizers exist (NUNCHAKU/SVDQuant, bitsandbytes NF4) and `enable_model_cpu_offload` / `enable_sequential_cpu_offload` are one-liners, so a 24 GB load is *not provably impossible* — but no quantized release, no offload script, and **no community loader for the per-block raymap/PRoPE/opacity modules** exist, so it is unbuilt and unmeasured. | ~10–30 min / clip est. (heavy offload) | **NonCommercial** (NVIDIA OneWay; base Wan Apache-2.0; code Apache-2.0) | **Very high / unbuilt** |
| **1** | **Wan2.1-T2V-1.3B + ArtiFixer mechanisms** — Self-Forcing student + Fun Plücker adapter + opacity-mix | **Plausible-but-unproven.** Same Wan *family* as the 14 B teacher, but the load-bearing per-block raymap/PRoPE/opacity modules are NOT in the 1.3 B family — "same family" ≠ architecturally faithful | **[survey] Wan2.1-T2V-1.3B fits "yes"** (survey:149) with no GB given. Component math is **estimate — unverified** and only closes with `t5_cpu` offload (the UMT5-XXL text encoder dominates otherwise); peak is a smoke-test gate, not a known number. | ~few s/clip est. with a 4-step student; minutes on the base — **unsourced, validate** | **Apache-2.0** if Wan-1.3B + Self-Forcing + Fun-Camera license strings check out (verify against HF files) | **High** — base ships T2V only; camera + multi-view + mask all come from separate add-ons that are not wired here |
| **2** | **Smaller non-Wan backbone** — SEVA 1.3B / CogVideoX-5B-I2V / LTX-Video 2B | **Medium (SEVA) → low (LTX)** — different family; hosts a subset of pillars; some need a train for camera control | **SEVA: [survey] 24 GB fit "unmeasured"** (survey:194) — a single smoke test is the gate, no scoped range is claimed here. **CogVideoX-5B-I2V: [survey] ~15 GB bf16** (survey:146); a camera-ControlNet add-on adds more (research-only, unquantified — do **not** cite a fixed 48–80 GB; that figure is unsourced and contradicts the survey). **LTX-2B: [survey] ~11–14 GB** (survey:148). | SEVA ~min/orbit; CogVideoX ~10–60 min/scene; LTX ~few s/clip — all est. | **SEVA NonCommercial**; **CogVideoX custom (commercial after registration)**; **LTX OpenRAIL-M (commercial OK)** | **Medium (SEVA, weights exist) → high (LTX/CogVideoX need a trained camera adapter)** |

**Read across the tiers:** Tier 0 is the literal ceiling but unbuilt at 24 GB and NonCommercial.
Tier 1 (Wan-1.3B) is a *candidate*, not the survey's pick — it trades SEVA's released multi-view
weights for a stack of unintegrated adapters. **The prior survey's recommendation is SEVA**
(survey:24-30,120-134); this doc does not overturn it. The cheapest closeness, though, is neither
tier — it is fp16 + multi-step difix on the backbone already in the repo (§1, §4 Step 1).

---

## 3. Pillar-by-pillar — what you can actually reproduce at 24 GB

ArtiFixer is six mechanisms. Two are **backbone-agnostic** (training-free). One is
**orchestration-only**. The remaining three are where the 14 B scale and/or a training run actually
live.

| Pillar | Faithful 24 GB realization | Cheap surrogate | Training | VRAM | Wires into (our file) | What is lost | Feasibility |
|--------|---------------------------|-----------------|----------|------|----------------------|--------------|-------------|
| **Opacity-mixing init** `z_mix = O_z·z_deg + (1−O_z)·ε` — **backbone-agnostic, but NOT free on single-step difix** | **Multi-step** backbone (multi-step difix, or Wan/SEVA) so empty O_z≈0 cells get a real denoise trajectory; pass `z_mix` as the sampler init | **First add multi-step to difix_ref**, *then* wire O_z. The single-step pipeline cannot denoise a noised init — SDEdit needs a multi-step trajectory | **None** (latent-init policy) — but requires the multi-step change to do anything but inject noise | ~0 over the difix peak ([measured] 8.4 GB) | `mask.downscale_to_latent` (mask.py:32-40, implemented, UNUSED); re-signature `prepare_latents` to accept `latents`/`noise`/`timestep` (it takes none today — pipeline_difix.py:625) and stop `__call__` overwriting them (pipeline_difix.py:1024) | On difix's **single step at τ=199** you get cleanup/routing only; a near-noise cell decodes to blur. Real empty-region synthesis needs the multi-step trajectory FIRST | **Moderate** (gated on multi-step difix landing) |
| **Single-pass generate-then-fit** — **orchestration-only, NO 14 B needed** | New `fit_fresh_scene`: pre-enumerate all gap poses (a **separate** change to `n_gap_poses`/the pose generator — orchestrator.py:493), generate once, free filler, fit fresh cloud (densify=True) on {targets, anchors} | **`rounds=1`** in the existing `fill_gaps_scene` (orchestrator.py:534) gives one pass — but it still draws only `n_gap_poses` poses per round, NOT all gaps; rounds=1 ≠ enumerate-all | **None** — pure orchestration + the built distiller | ~0 added if densify stays OFF. **Do NOT silently set `cap_max=1.15`**: the fill path uses `cap_max_factor=1.05` and `densify=False` (orchestrator.py:497,604-605) precisely to bound MCMC growth — 1.15+densify is the OOM/growth path the 1.05 cap exists to prevent | NEW `fit_fresh_scene` beside orchestrator.py:479; pose pre-enumeration in `synthesize_near_view_poses`; reuse distiller densify path (distiller.py:85,127); cheap one-pass hook = `rounds=1` | Targets aren't mutually consistent unless the upstream filler made them so → noisier fit. From-scratch on <20 views risks worse PSNR than warm-start (measured −1.5 dB full-opt) → ships OFF behind the gate | **Moderate** |
| **Multi-frame / video consistency** — **needs a real multi-view DiT** | **SEVA** (released, multi-view native) as a `TrajectoryFiller`: K-pose mini-orbit → one forward → K mutually-consistent views. **Multi-view IMAGE diffusion (Zero123++/MVDream)** fits the per-gap fill shape better and lighter than a video DiT | Keep per-frame difix_ref + buy consistency downstream: progressive distill + hard recomposite + 50/50 anchors (the 3D fit as consensus) | **None** to run released weights; commercial-clean refit = control-LoRA (days, 1 card) | [survey] SEVA 24 GB fit "unmeasured" (survey:194) — smoke-test gate, no range asserted | NEW `fillers/seva.py`; extend `synthesize_near_view_poses` (174-235) to K poses; `_fill_gap_views` (356-385) renders K + loads M refs; inject via `view_filler` kwarg (507) | 1.3 B vs 14 B fidelity; short coherent horizon; no native mask; slower (no DMD at our scale anyway) | **Hard** |
| **Plücker camera control** — per-block additive raymaps, VAE-bypassing | **Adopt, don't train.** SEVA conditions natively on Plücker (relative-to-first-frame). Wan-1.3B's `Wan2.1-Fun-Control-Camera` adapter is a candidate but unintegrated here | **Render-then-fix** (today's path): render the commanded pose from our 3DGS, hand to a pose-agnostic fixer — geometry exact for *observed* surface | **None** (adopt weights). From-scratch VD3D adapter ≈ 96 A100-days — infeasible | Plücker ~0; cost is the backbone footprint, staged | **BUG (one-line):** orchestrator.py:380 builds the reference as `RenderResult(rgb=ref_img, camera=cam)` where `cam` is the GAP/degraded pose, NOT the reference pose. So `references[0].camera` carries the wrong pose → relative pose is identity → useless for triangulation/PRoPE. **Fix:** use `camera=ref_view.camera` (the true ref pose is in `RealView.camera`, capture_images.py:44; returned as the pair's base from `synthesize_near_view_poses`) | **Render-then-fix recovers ~0% in the far/empty regime** — nothing to render where volume is truly empty, which is exactly the regime ArtiFixer is *for*. So this surrogate covers the near regime only | **Moderate (near) / nil (far)** |
| **Multi-reference PRoPE** — 0..N posed refs, relative pose in cross-attn | SEVA (native M-in/N-out) **+ PRoPE swap + retrain** on posed pairs | **Generalize the vendored difix UNet from `num_views=2` to `1+N`** — same joint self-attention fuses K refs, no new weights — but pose-**blind** | **Fine-tune** mandatory for faithful PRoPE (tens of GPU-hrs, 1 card); cheap surrogate needs none | PRoPE ~0; backbone footprint, staged | refs are already `Sequence[RenderResult]` (protocols.py:53-55) — *contract* ready, but the pose is mis-threaded (see Plücker bug). `num_views=2` is a **hardcoded local literal** in `BasicTransformerBlock.forward` (unet_2d_condition.py:75, "Assuming 2 views for simplicity"), NOT a config field — generalizing means editing the constant AND threading N through every block's forward: **vendored-file surgery, not a flag** | Cheap surrogate has no relative-pose signal → conflicting refs average rather than triangulate. 1.3 B empty-region invention weaker than 14 B | **Hard** |
| **Causal-AR streaming student** — 4-step DMD + rolling-KV — **DROP at our scale (<20 views/scene)** | **Self-Forcing** (Apache-2.0, Wan-1.3B, pre-distilled) for 4-step speed, run in plain chunk mode | The progressive-distill consensus we ship; **or distill difix_ref ITSELF** (LCM/turbo few-step) for speed on the backbone already vendored — the overlooked cheaper path | **None** to use a released ckpt; difix few-step distill is a small train | inference est. — staged-then-freed; **no measured figure** | NEW `SelfForcingFiller` mirroring `DiffusionFiller` lazy-load + free/unload (diffusion.py:53-95,179-191); inject via `view_filler` (507) | The rolling-KV long-horizon property is dead weight at <20 views | **Moderate (mostly unnecessary)** |

**The crisp takeaway:** single-pass generate-then-fit is the only genuinely free pillar.
Opacity-mixing is free *in policy* but needs multi-step difix to do anything. The camera surrogate
(render-then-fix) is free but recovers ~0% in the far/empty regime. Video consistency, learned
Plücker, and multi-ref PRoPE are where scale and/or training actually live.

---

## 4. The recommended staged build

Ranked cheapest-useful-first, wired to real seams. Every step respects the **no-regression gate**
(orchestrator.py:472, default `regression_tol_db=0.3`) and the **frozen-geometry default**
(means/scales/quats LR=0, densify=False — measured: full-opt −1.5 dB, opacity 2e-2 −4.8 dB).
Aggressive steps ship **OFF by default**.

| # | Change | Why | Components (file / function) | VRAM | Closeness gained | Risk |
|---|--------|-----|------------------------------|------|------------------|------|
| **1** | **fp16 difix** — set `dtype="float16"` on the existing filler | Roughly halves the measured 8.4 GB peak with one parameter; the cheapest headroom we have, on the backbone already shipped | `DiffusionFiller(dtype="float16")` (diffusion.py:119,159) | **~½ of [measured] 8.4 GB** | None toward ArtiFixer, but frees the budget every later step spends | **Low** — supported flag; A/B on quality |
| **2** | **Multi-step difix** — run `num_inference_steps=N>1` from a noised init so there is a real denoise trajectory | The actual fix for the opacity-mix pillar: SDEdit needs multiple steps; a single τ=199 step cannot denoise. This is the prerequisite Step 1 in earlier drafts skipped | `__call__` `num_inference_steps`/`timesteps` (pipeline_difix.py:839,1021); the loop (1053-1083) already iterates `timesteps` | small, staged | **Real** — unlocks generative fill (not just routing) on the existing backbone | **Medium** — changes output character; gate on held-out PSNR |
| **3** | **Wire the opacity-mix init (after Step 2)** — VAE-encode `z_deg`, `O_z = downscale_to_latent(alpha, …)`, `z_mix = O_z·z_deg + (1−O_z)·ε`, noised to the start timestep | With a multi-step trajectory in place, `z_mix` becomes a genuine SDEdit init; without Step 2 it is noise injection with no trajectory to remove it (would DEGRADE output) | `mask.downscale_to_latent` (mask.py:32-40, UNUSED); **re-signature** `prepare_latents` to accept `latents`/`noise`/`timestep` (none today — pipeline_difix.py:625) and stop `__call__` clobbering the kwarg (pipeline_difix.py:1024); reinstate `add_noise` (675-678) | ~0 over Step 2 | **Real** in empty cells (perceptual only — see §6) | **Medium** — depends on Step 2; behind the gate |
| **4** | **Fix the reference-pose bug** — orchestrator.py:380 must pass `camera=ref_view.camera`, not `camera=cam` | Today `references[0].camera` carries the degraded pose → relative pose is identity → any pose-aware filler (SEVA/PRoPE/Plücker) gets garbage. One line; prerequisite for every pose-conditioned path | orchestrator.py:380; `ref_view` is a `RealView` with `.camera` (capture_images.py:44) | 0 | Enables the camera pillar at all | **Low** — one-line correctness fix; current diffusion filler reads `.rgb` only, so no behaviour change today, but it unblocks Steps 6+ |
| **5** | **Single-pass `rounds=1`** — optionally add explicit pre-enumeration of all gap poses (separate change to `n_gap_poses`/the pose generator) | Get the non-progressive spirit cheaply; note `rounds=1` is one pass but still only `n_gap_poses` per round — enumerate-all is a distinct change | `fill_gaps_scene` `rounds=1` (orchestrator.py:534); pose pre-enumeration in `synthesize_near_view_poses` (174-235); keep `cap_max_factor=1.05`, `densify=False` (497,604-605) | ~0 | Workflow-shape closeness; cheaper loop | **Low** |
| **6** | **Stand up a posed filler behind the `ViewFiller` seam** — **SEVA first** (survey's pick, released multi-view weights), or a single-image posed inpainter (FLUX/SD3.5 + ControlNet) matching the one-image-in contract; render-then-fix + opacity-mix init + hard recomposite | The genuine new-backbone step. SEVA is the proven 24 GB surrogate; a single-image posed inpainter matches `ViewFiller.fill` better than any video DiT. A Wan-1.3B stack is a candidate but its camera/multi-view/mask come from unintegrated add-ons | NEW `fillers/seva.py` (or inpainter) impl `ViewFiller.fill` + `free()/unload()` (mirror diffusion.py:179-191); inject via `view_filler` (orchestrator.py:507); K poses via `synthesize_near_view_poses`; M refs in `_fill_gap_views` (356-385) — **requires Step 4** | smoke-test gate ([survey] SEVA 24 GB "unmeasured", survey:194); staged-then-freed | **High** in the regime it fits — but does **not** beat difix_ref on small near-camera holes (survey:26-30) | **Medium-High** — gate behind held-out PSNR; geometry stays frozen unless gaps are far/empty |
| **7** | **Gap-far densify opt-in** — for genuine far/empty gaps, set geometry LRs > 0, `densify=True`, `restrict_to_gaps=True` so new surface lands only at gap gaussians | The only way invented far-gap content *lands* in 3D; scaffolding exists but is OFF | distiller densify flags (distiller.py:85,127,230); `gap_index`/`fill_gap_gain` set from `cov_r.gap_gaussian_mask` (orchestrator.py:447-451) | bounded by `cap_max` (keep the explicit `cap_max_factor` set in the fill path, do not default to 1.15 blind) | Unlocks the regime ArtiFixer is *for* | **High** — ships OFF; footgun: means/scales/quats grads are NOT gap-masked (only sh/opacities are) — do not unfreeze geometry under gap-restriction naively |
| **8** | **Multi-ref PRoPE fine-tune (faithful)** — generalize the UNet `num_views=2` literal to `1+N` (vendored-file surgery, see §3), then PRoPE swap + LoRA retrain on posed pairs | The last train-required pillar; turns appearance-borrowing into geometric triangulation | edit the hardcoded `num_views` literal + thread N through every block forward (unet_2d_condition.py:75-76,128); `prope_dot_product_attention` drop-in; refs already `Sequence[RenderResult]` (protocols.py:53-55) — requires Step 4 | train tens of GPU-hrs/1 card; PRoPE inference ~0 | Incremental — refs placed geometrically | **Medium** — only after 1–7 land |

**Do Steps 1, 2, 4 first.** fp16 is free headroom, multi-step difix is the real generative unlock on
the backbone you already ship, and the reference-pose fix is a one-line correctness bug that unblocks
every pose-aware path. Only then reach for a new backbone (Step 6), and only target the far/empty
regime (Steps 7–8) when that — not our single-floor interior — is the actual goal.

---

## 5. What stays out of reach at 24 GB

Blunt list, with mitigations.

| Out of reach | Why | Mitigation |
|--------------|-----|------------|
| **The bidirectional 14 B Wan teacher, productionized** | No quantized release, no offload script, no community loader for its per-block raymap/opacity/PRoPE modules. 4-bit DiT quantizers + CPU offload *might* fit a base load, but the custom modules are unsupported and unmeasured | Mine ArtiFixer's *mechanisms* (opacity-mix, render-then-fix camera, single-pass fit), not its weights. If you want a video backbone, SEVA is the survey's pick |
| **Native 14 B-scale empty-region generation** | A 1.3 B-class model hallucinates blurrier content in fully-empty far-from-camera regions — the precise regime the 14 B scale was built for | Restrict heavy machinery to exactly that regime; keep cheap multi-step difix for near-camera residuals. On our single-floor interior the loss is largely moot (gaps are occluded surface, not empty volume) |
| **DMD-distilled 14 B speed parity end-to-end** | The released causal-AR student is built on the 14 B teacher | Adopt Self-Forcing (Apache-2.0, on Wan-1.3B) for 4-step speed, OR few-step-distill difix_ref itself; drop rolling-KV we don't need at <20 views |
| **Commercial productization of NVIDIA / SEVA weights** | ArtiFixer, difix, and SEVA are NonCommercial | Ship an Apache-2.0 stack (Wan-1.3B + Self-Forcing + Fun-Camera) or LTX (OpenRAIL-M). **Verify every license string against the actual HF/repo files before shipping** — they are asserted here without source links |
| **A faithful temporal O_z and per-block opacity injection** | Wan-VAE is spatiotemporal; faithful O_z pools spatially AND temporally, and ArtiFixer pairs the mix with per-block injection we drop | Minor at <20 views (no long clips). Our 2D-spatial `downscale_to_latent` captures the load-bearing part; per-block injection is dropped for render-then-fix |
| **Filling a structurally-empty voxel with frozen geometry** | `coverage3d` defines a gap as occupied-but-unseen; frozen-opacity fill is consistent for occluded surface but cannot raise gaussians where none exist | Keep geometry frozen/densify off by default; expose the narrow gap-local densify opt-in (Step 7), gated by the PSNR backstop |

---

## 6. Honest closeness ceiling

Per-tier estimate of how much of ArtiFixer's *unique* capability you recover. "Unique" = what
ArtiFixer adds **over our shipped difix_ref baseline** (which already reproduces the Difix3D+ inner
loop verbatim) — opacity-aware generation, posed multi-view-consistent gen, single-pass fit. These
percentages are judgment calls, not measurements.

| Tier | % of ArtiFixer's unique capability recovered | Where it lands | Where "close" becomes unprovable |
|------|---------------------------------------------|----------------|----------------------------------|
| **Tier 0** (real 14 B) | **~0% at 24 GB** — it *is* ArtiFixer in principle, but it is unbuilt and unmeasured at 24 GB, so the *effective* recovery is near zero. The "~95–100%" headline in earlier drafts was misleading; it only applies to hardware you don't have | Everything, if you could load it | N/A — unloadable as built |
| **Tier 1** (Wan-1.3B family) | **~40–60%, with caveats** — single-pass fit and (multi-step) opacity-mix are reachable; but PRoPE needs a train (NOT present), multi-frame consistency needs released weights wired + verified (NOT present here), and base Wan-1.3B ships no camera/mask. So "all 6 pillars present" is false — several are *scaffoldable*, not working | A candidate carrier, NOT a proven upgrade over difix_ref or SEVA | The empty-region win rests on perceptual/distributional metrics (FID/LPIPS/user study), not PSNR — no GT in the empty regime, so "as good as ArtiFixer there" is not measurable on our scenes |
| **Tier 2 — SEVA** | **~45–60%** — native Plücker + multi-view + 0..N refs, hosts opacity-mix; missing causal-AR, native mask, cheap single-step routing; NonCommercial. **This is the survey's recommended surrogate** | Closest non-Wan multi-view-native carrier | Same GT-free ceiling; **plus** SEVA's 24 GB fit is [survey] unmeasured — "fits" is unprovable until the smoke test |
| **Tier 2 — CogVideoX-5B / LTX-2B** | **~25–45%** — video consistency + opacity-mix + single-pass; camera control needs a trained adapter (CogVideoX) or is absent (LTX). License-clean | A commercial-clean carrier *if* you fund the camera-adapter train | LTX's consistency is temporal, not multi-view-geometric — render-then-fix is the only camera path, so "multi-view consistent" is neither guaranteed nor directly measurable GT-free |

**The structural unprovability:** ArtiFixer's headline advantage is *empty-region* generation, and
the empty regime has **no ground truth** — PSNR is undefined there. Any ≈1–3 dB margin is
**observed-region-only**. So at every tier, "how close are we *really*" in the regime that matters can
only be argued perceptually (FID/LPIPS/user study), never proven with a reconstruction metric on our
own captures. The no-regression gate proves *we did no harm* in the observed regime; it cannot prove
*we matched ArtiFixer* in the empty one. Watch **per-eval PSNR, not just the strided mean** — a local
regression near gaps can slip a strided gate.
