# Survey: generating & auto-evaluating robot tasks from real 3DGS scenes

*Deep-research synthesis, 2026-06-25. 23 primary sources, 74/75 extracted claims
survived 3-vote adversarial verification. Scope: task formalisms, open-vocab 3D
object extraction, LLM task-generation, auto-eval — for symbolic pick/place on
real 3D-Gaussian-Splat office scenes at ~200k-scene scale.*

## TL;DR — recommended stack

| Layer | Pick | Why |
|---|---|---|
| **Task schema** | **BDDL-flavoured symbolic goal predicates** (our ADR-0010), position-predicates verified **geometrically** vs extracted object centres | BDDL is pure state-logic (no action symbols), semantic goals satisfiable by many states; but our "ground truth" is extracted positions, so position predicates reduce to a distance check |
| **Tolerance** | **~0.15 m** placement (Habitat), **partial credit per-predicate** (ALFRED Goal-Condition Success) | These are the canonical, reused numbers/metrics |
| **Perception** | **objects natively from 3DGS** — GaussianGraph (instances+relations) or OpenGaussian/OpenSplat3D (instances) + **HOV-SG-style room hierarchy** for "location Y"; **Clio** task-driven filtering to cap cost | Matches our 3DGS input; hierarchy resolves "bring to room Y"; Clio keeps only task-relevant objects |
| **Task-gen** | **GRS-style**: propose tasks **only over the detected object inventory** (closed set) + a solvability "router" loop | Structural anti-hallucination + feasibility, on *real* scenes |
| **Auto-eval** | our `grab`/`drop` `CarryEvent` vs extracted centres, ε≈0.15 m, partial-credit per predicate + SPL for the navigation half | Closes the loop on ADR-0010 |

**Biggest bottleneck (cost & quality): open-vocab 3D perception on glass/reflective
office surfaces.** 3DGS is metrically *unreliable* exactly there, and that error
flows straight into wrong object centres → noisy auto-eval labels. This is the pixel
to de-risk first.

---

## 1. Task formalism — adopt symbolic predicates, verify positions geometrically

Two families, both auto-checkable:

- **Geometric goals.** Habitat 2.0 / Rearrangement Challenge specifies each target as
  `(initial COM, desired COM)` and scores success as **placement within 0.15 m of the
  desired centre-of-mass** (orientation ignored) + a `stop` action [Habitat2.0;
  Habitat-2022]. AI2-THOR Visual Rearrangement instead checks **3D bbox IoU > 0.5** plus
  **openness within 20 %** against a walkthrough reference [AI2-THOR-rearr]. Simple, but
  no semantics.
- **Symbolic predicates.** **BDDL** (BEHAVIOR-1K) is first-order logic over objects:
  an `:objects` scope, fully-grounded `:init`, and a `:goal` logical expression — and,
  unlike PDDL, it **omits action symbols** (pure *state* representation), which is exactly
  what symbolic pick/place verification wants [BEHAVIOR-1K; BDDL-docs]. OmniGibson checks
  the goal **every step** via `bddl.activity.evaluate_goal_conditions`, returning per-predicate
  satisfied/unsatisfied [predicate_goal]. Vocabulary spans spatial (`onTop`, `inside`,
  `nextTo`) and state predicates (`toggled`, `cooked`…). **ALFRED** uses PDDL goal-conditions
  with two metrics worth copying: binary **Task Success** and partial-credit **Goal-Condition
  Success** (ratio of satisfied conditions; ~2.55 conditions/task) [ALFRED].

**Verdict.** Base the schema on **BDDL-style predicates** (human-readable + machine-checkable),
but because our ground truth is *extracted* object positions (not a simulator), a position
predicate like `at(obj, region)` is verified **geometrically** against the extracted centre —
i.e. Habitat's 0.15 m rule. Report **partial credit per predicate** (ALFRED). This is precisely
what ADR-0010 proposes; the research validates it and pins the tolerance.

⚠️ Caveat: every one of these benchmarks lives at **10¹–10³ scenes** (BEHAVIOR-1K: 1000
activities / 50 scenes; ALFRED: 120 scenes). **None operates at 200k.** The schema transfers;
the *perception+gen* throughput is the novel scaling problem.

## 2. Perception — open-vocab 3D objects (the crux)

**General scene graphs from posed RGB-D:**
- **ConceptGraphs** (ICRA 2024) — the canonical baseline: SAM per frame → CLIP per region →
  project to point cloud → multi-view fuse into 3D instances → LVLM captions + LLM-derived
  relations. Output is object instances with 3D positions + CLIP embeddings + relations,
  serialisable to text for an LLM planner. **Flat** graph (no hierarchy) is its limit
  [ConceptGraphs].
- **HOV-SG** (RSS 2024) — adds the **floor→room→object hierarchy** ConceptGraphs lacks,
  75 % smaller than dense maps, ~55 % real-robot nav success. Best when "location Y" must
  resolve to a *named room* [HOV-SG].
- **Clio** (MIT, RA-L 2024) — **task-driven**: given the task list, an Information-Bottleneck
  clustering keeps only the task-relevant objects at the right granularity, real-time on-board.
  The lever for **controlling cost at 200k scale** [Clio].

**Objects natively from 3DGS (our input modality):**
- **GaussianGraph** (Mar 2025) — builds an open-vocab **scene graph (instances + relations +
  attributes) directly on Gaussians**; closest single match to what task-gen + predicate-checking
  need [GaussianGraph].
- **OpenGaussian** (NeurIPS 2024) — **point-level** instances via SAM masks + CLIP + a two-stage
  codebook; strong instances but **no relations** [OpenGaussian].
- **OpenSplat3D** (Jun 2025) — open-vocab 3D **instance** segmentation on 3DGS, tested on
  ScanNet++ (real indoor) [OpenSplat3D]. **Semantic Gaussians** — *training-free* 2D→3D feature
  projection (cheaper to attach semantics) [SemanticGaussians]. **LangSplatV2** — CLIP *field* at
  450+ FPS, but field-level (no instances); pair with an instance method [LangSplatV2].

**Confirmed failure mode on office glass/reflective surfaces (quantified):**
- 3DGS gives high visual quality but **introduces surface artifacts that degrade metric
  reliability on transparent surfaces**; **2DGS is more metrically reliable** on glass (at higher
  cost) [Transparent-Sensors2025].
- Vanilla 3DGS **fails on reflective/specular surfaces** (can't capture high-frequency speculars →
  floaters/wrong geometry); Ref-Gaussian partially fixes appearance but adds no semantics
  [Ref-Gaussian].
- All CLIP/SAM-based extractors above inherit this: corrupted Gaussians near glass → mislocalised/
  mislabelled objects → **wrong goal centres** for auto-eval.

## 3. Task generation — ground it in the detected inventory

- **GenSim / GenSim2** (ICLR 2024 / 2024) — tasks as **executable sim code** (running it *is* a
  feasibility check). But the authors **explicitly document hallucination / lack of geometric
  grounding**, needing manual filtering [GenSim]; GenSim2 anchors to a concrete articulated-object
  inventory to stay grounded [GenSim2].
- **RoboGen** (ICML 2024) — propose-generate-learn loop, auto-generates supervision; **but builds
  scenes/assets from LLM priors**, not from a real reconstruction's inventory — wrong fit for our
  real-scene goal [RoboGen].
- **Holodeck** (CVPR 2024) — environment generation from an asset catalog under spatial
  constraints; informs *constraining proposals to a finite inventory*, not task verification
  [Holodeck].
- **GRS** (CVPRW 2025) — **the closest template**: from a real RGB-D image, SAM2 + VLM describe
  objects → match to assets (F1 0.89 with GPT-4o) → **generate tasks only over the recognised
  objects** (structural anti-hallucination) + a **"router" loop** that iteratively fixes the
  task/test until solvable [GRS].

**Verdict.** Follow GRS: the LLM proposes tasks over the **closed set of extracted objects**
(can't invent what isn't there), each gated by a **solvability/feasibility check** — for us, reuse
the `_line_of_sight_clear`/coverage reachability already in the codebase.

## 4. Auto-evaluation — hybrid predicate + geometric

- **Geometric tolerances:** Habitat **0.15 m** COM placement + `stop` [Habitat2.0]; AI2-THOR
  **IoU > 0.5** + openness < 0.2 [AI2-THOR-rearr]; ObjectNav success rate + **SPL** (stop within a
  geodesic distance of the target) [Habitat-Nav2023].
- **Symbolic:** ALFRED binary Task Success + **partial-credit Goal-Condition Success** [ALFRED];
  BEHAVIOR/OmniGibson `PredicateGoal` = **all-or-nothing conjunction** of predicates + per-predicate
  `goal_status` for partial progress [predicate_goal].

**Verdict for us:** symbolic goal predicates where position-predicates are checked geometrically
vs extracted centres (ε ≈ 0.15 m ≈ our `2·step`), reported as **partial credit per predicate**
(ALFRED), with **SPL** for the navigation half. This is the ADR-0010 rule, refined.

---

## Sources

- **BEHAVIOR-1K / BDDL** — arxiv.org/html/2403.09227v1 ; behavior.stanford.edu/getting_started/important_concepts.html ; behavior.stanford.edu/omnigibson/reference/termination_conditions/predicate_goal.html ; 2025 challenge: svl.stanford.edu/b1kbeta/challenge/overview.html
- **ALFRED** — arxiv.org/pdf/1912.01734 (+ ar5iv mirror)
- **Habitat 2.0** — ar5iv.labs.arxiv.org/html/2106.14405 ; **Rearrange 2022** — aihabitat.org/challenge/2022_rearrange/ ; **Nav/ObjectNav 2023** — aihabitat.org/challenge/2023/
- **AI2-THOR Rearrangement** — github.com/allenai/ai2thor-rearrangement ; **Rearrangement framing** — arxiv.org/pdf/2011.01975
- **ConceptGraphs** — concept-graphs.github.io ; **Clio** — arxiv.org/abs/2404.13696 ; **HOV-SG** — hovsg.github.io ; **Functional 3D scene graphs** — arxiv.org/html/2503.19199v1
- **OpenGaussian** — arxiv.org/abs/2406.02058 ; **OpenSplat3D** — arxiv.org/abs/2506.07697 ; **GaussianGraph** — arxiv.org/pdf/2503.04034 ; **Semantic Gaussians** — arxiv.org/abs/2403.15624 ; **LangSplatV2** — arxiv.org/pdf/2507.07136
- **Transparent-surface 3DGS metric study** — mdpi.com/1424-8220/25/14/4410 ; **Reflective Gaussian Splatting** — ICLR 2025
- **GenSim** — arxiv.org/abs/2310.01361 ; **GenSim2** — arxiv.org/abs/2410.03645 ; **RoboGen** — arxiv.org/abs/2311.01455 ; **Holodeck** — yueyang1996.github.io/holodeck ; **GRS** — arxiv.org/html/2410.15536v1

*Note: one claim (AI2-THOR walkthrough perturbation details) was flagged as partly imprecise in
verification; treated as low-weight. Recency: a 2026 "RoboGene" preprint appeared in search but is
newer/less established than the four core generators and is not relied on here.*
