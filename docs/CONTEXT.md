# Project Context

## What this is

**gaussian-robot** explores a **3D Gaussian Splat** with a **Vision-Language
Model (VLM)** agent in order to propose **new camera poses** in under-sampled
regions. Those poses are the deliverable: they are turned into novel views
(currently via a generative/diffusion model) and used to **densify** the splat
so it renders more coherently — the goal being **plausible, artifact-free**
rendering for downstream robot navigation (Goal A), not geometric truth.

## Resolved design (the grilling, in one place)

| # | Decision | Choice |
|---|---|---|
| Q1 | Control granularity | **Local incremental controller** (one discrete action per step) + **multi-seed restarts** from training poses for global coverage. Global waypoint planning and frontier-teleport deferred. |
| Q2 | Action set | Egocentric verbs `forward back turn_left turn_right look_up look_down stop`. **System-owned** magnitudes: step = `0.03 × AABB diagonal`, turn/pitch = 30°. Floor-plane translation, pitch is view-only. One action/step. |
| Q3 | Observation encoding | **3-panel** message: RGB view + colormapped depth + **body-fixed** top-down map (forward=up; blue dots = sampled poses, green polyline = current trail, red arrow = current pose/heading) + a fixed task prompt + one live state line. |
| Q4 | Termination | Two-level, OR-composed `StopPolicy`s. Per-walk: step budget / **coverage plateau (primary)** / bounds-or-degenerate-render. VLM `stop` is **demoted** to a plateau-counter vote (cannot end a walk alone). Session: coverage target / pose budget / seed exhaustion / diminishing returns. |
| Q5 | Metric & goal | **Goal A (plausibility).** Tier 1 (floor coverage % + novelty) & Tier 2 (pose-space `(cell × direction-bin)` coverage) now; Tier 3 (ray/gaussian coverage) later; Tier 4 (retrain + eval) deferred — observed regions use Δ-PSNR as a **regression guard only**; unobserved regions use no-reference IQA + cross-view consistency + human eval (no real GT exists there). Baseline bar: `farthest-point` sampling at matched budget. |
| Q6 | Output filtering | Global over union of trajectories: quality drop (degenerate renders) → position **novelty dedup** (greedy farthest-point, `r_keep`) → budget cap `P`. Position-only for v1. |

## Ubiquitous language

| Term | Meaning |
|------|---------|
| **Scene** | A reconstructed space; concretely a `SplatScene`. |
| **Splat** | A 3D Gaussian Splatting reconstruction. |
| **Gaussian** | One ellipsoidal primitive (mean + covariance + colour/opacity). |
| **Pose** | 6-DoF position + orientation (`Pose`). OpenCV camera convention: +Z forward, +X right, +Y down; world is +Y up. |
| **Camera** | Pose + intrinsics (`Camera`). Fully determines a rendered view. |
| **Render / View** | Turning a `Camera` into an image (RGB, ±depth) — the VLM's input. |
| **Action** | One discrete egocentric verb (`Action`) the VLM emits per step. |
| **ActionSpace** | The system-owned magnitudes (`step`, `Δrot`) applied to actions. |
| **Observation** | The 3-panel + text payload sent to the VLM each step (`Observation`). |
| **Walk** | One local-control episode from a seed pose. |
| **Session** | A collection of walks across multiple seeds; produces the deliverable. |
| **Coverage state** | The accumulated set of sampled poses; drives novelty, plateau, metrics (`CoverageState`). |
| **Novelty** | Min floor-plane distance from a pose to any sampled pose. |
| **StopPolicy** | A terminating condition; many are OR-composed. |
| **Decision** | The VLM's per-step output: an `Action` (+ optional raw text). |
| **Deliverable** | The filtered set of new poses (`filters`). |
| **VLM** | Vision-Language Model — `Qwen/Qwen3.5-9B` served via vLLM (OpenAI-compatible API). |

## Pipeline

```
 for each seed pose in training set:
   walk:
     render(camera) ──▶ Observation (rgb+depth+map+prompt)
                                   │
                                   ▼
                              VLM ──▶ Action
                                   │
     apply_action(pose, action) ◀───┘
     update coverage, check StopPolicies
   until per-walk stop
 until session stop (coverage target / budget / seeds / diminishing returns)

 union(trajectories) ──▶ filters ──▶ Deliverable: set of new poses
                                            │
                                  (later) render/diffuse at each pose
                                            ──▶ densify 3DGS
```

## Open / deferred (future ADRs)

- **Concrete renderer backend** (gsplat vs other). Blocked: needs scene data.
- **Concrete VLM client** (Qwen3.5-9B on vLLM). Blocked: needs GPU + weights.
- **Diffusion novel-view generator** that supplies training images at proposed poses.
- **Tier 3** ray/gaussian coverage (needs renderer alpha/opacity).
- **Tier 4** retrain + eval protocol (needs the 3DGS training pipeline).
- Angular dedup, action chunking, vertical motion, frontier-teleport, multi-magnitude actions — all cheap upgrades once Tier-1/2 metrics are live.

## Non-goals

- Building a SLAM / reconstruction pipeline — we *consume* splats.
- Training gaussians from scratch — we densify existing ones.
- Geometric truth (Goal B) — we pursue plausibility (Goal A).
