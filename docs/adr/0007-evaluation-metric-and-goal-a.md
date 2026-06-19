# 0007. Evaluation metric & Goal A (plausibility)

- Status: accepted
- Date: 2026-06-19

## Context

"Improve the gaussian" must be measurable. But the regions we improve are
under-observed — there is no real ground truth there. Any novel-view image of
those regions is a **diffusion-model hallucination**, not captured truth, so
PSNR against it measures conformity to a prior, not fidelity. The original
"Tier-4 = Δ-PSNR on held-out" is therefore invalid for the regions that matter.

This forces a prior question: what does "improve" mean when supervision is
generated? We resolve it as **Goal A (plausibility)**, not Goal B (truth).

## Decision

**Goal:** a more *plausible, artifact-free, renderably coherent* splat for
navigation — not geometric truth.

**Tiered metrics:**
- **Tier 1 (now):** floor-plane coverage % (navigable cells within radius `r`
  that are sampled) + mean novelty. Trivial. Drives termination tuning.
- **Tier 2 (now):** pose-space coverage — `(position cell × viewing-direction
  bin)` grid. Cheap; adds angular diversity.
- **Tier 3 (later):** ray/gaussian coverage — fraction of gaussians seen from
  ≥ `k` distinct directions (needs renderer alpha/opacity).
- **Tier 4 (deferred, redefined):** after retraining 3DGS with proposed views:
  - **Observed regions** (real holdout exists): Δ-PSNR/SSIM/LPIPS as a
    **regression guard** (`≥ 0`, "don't break what works"), *not* a goal.
  - **Unobserved regions** (no real GT): no-reference IQA (CLIP-IQA/MUSIQ),
    artifact/floater reduction, **cross-view consistency** (train on diffusion
    views at `P_train`, re-render at held-out `P_test` freshly generated), and
    human evaluation.

**Baselines** at equal pose budget `P`: `uniform grid` (lower bound), `random`
(sanity), `farthest-point` (the bar to beat).

**Success criterion:** the VLM agent beats `farthest-point` on the Goal-A
signals, judged primarily via Tier-4 (once available) — **never on Tier-1
alone**, because the cheap positional metric is exactly what a dumb sampler
maximises and would falsely negative the VLM's semantic value.

## Consequences

- ✅ Honest about the limit of generated supervision; avoids false PSNR claims.
- ✅ Tier-1/2 are implementable now and unblock termination + agent iteration.
- ✅ `farthest-point` gives a concrete, cheap bar to beat.
- ⚠️ Goal-A metrics (no-reference IQA, human eval) are noisier and more
  expensive to trust than PSNR; cross-view consistency is our best automated
  proxy but still measures coherence-with-the-prior, not truth.
- ⚠️ If Goal B (truth) ever becomes required, diffusion supervision becomes a
  liability and real captures in target regions become mandatory — supersede.
