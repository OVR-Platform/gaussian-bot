export const meta = {
  name: 'splat-enhance-study',
  description: 'Study how to introduce generative novel-view enhancement of blurry/under-observed 3DGS areas into the gaussian-robot stack within 24GB VRAM',
  phases: [
    { title: 'Understand', detail: 'paper deep-read + our-stack map + lightweight-analog lit survey + 24GB-prior survey' },
    { title: 'Design', detail: '4 distinct candidate architectures (judge panel)' },
    { title: 'Verify', detail: 'adversarial VRAM/consistency/integration/obtainability check per candidate' },
    { title: 'Synthesize', detail: 'recommended phased design + VRAM budget + milestone-0 experiment' },
  ],
}

// ---------- shared grounding ----------
const CONTEXT = `
PROJECT: "gaussian-robot" — navigate a robot INSIDE a pre-trained 3D Gaussian Splat scene by
rendering views from poses and feeding them to a VLM. Pluggable, protocol-based architecture.

OUR ACTUAL STACK (verified by reading the code):
- Renderer: src/gaussian_robot/backends/gsplat_renderer.py uses gsplat.rasterization with
  render_mode="RGB+D", currently wrapped in torch.no_grad(). It loads a GaussianCloud dataclass
  holding means(N,3), quats(N,4), scales(N,3), opacities(N,), sh_coeffs(N,K,3) as torch CUDA
  tensors — i.e. the FULL differentiable 3DGS parameter set is already resident in GPU memory.
  Each render returns RenderResult(rgb uint8 HxWx3, depth HxW float32, alpha HxW float32).
  The ALPHA channel = accumulated opacity = a per-pixel coverage/transmittance signal (this is
  directly analogous to the reference paper's rendered opacity map O).
- NO gaussian optimizer/trainer exists yet. In this repo the word "densify" is a NAVIGATION
  mission mode (explore to find under-reconstructed regions and PROPOSE NEW CAPTURE POSES), NOT
  gaussian optimization. So "enhancing the splat" requires ADDING a 3DGS fine-tune/optimization
  loop. gsplat (>=1.0) supports training + MCMC/default densification strategies.
- Coverage infra: metrics/coverage3d.py builds a voxel occupancy grid (opacity-weighted) + a
  "seen" mask via ray-casting capture-camera frustums, exposing gap_mask / gap_centers() =
  exposed-surface voxels that NO capture camera saw (roofs, behind-object pockets). metrics/
  coverage.py does floor + pose-space coverage and treats a render with alpha>=thr as a
  high-quality observation. These are our "WHICH areas to enhance" triggers.
- Depth prior: depth/estimator.py wraps Depth Anything 3 (DA3-BASE) monocular depth, pluggable.
- VLM: vlm/qwen.py via a local vLLM OpenAI-compatible server (Qwen3-VL family).
- Scenes: pre-trained 3DGS .ply + cameras.json (capture poses) or COLMAP. We LOAD, never train.
- Conventions (ADR-0002): +Y up, OpenCV camera axes, world->camera rotation. cameras.json is in
  the same world frame as the PLY.

HARD CONSTRAINT: a SINGLE 24GB-VRAM GPU. The whole runtime may also need the renderer, the depth
model, and possibly the VLM (vLLM) resident — so the enhancement stage must fit in 24GB, likely
by STAGING (load/free models) rather than co-residence. Plan for staging explicitly.

REFERENCE PAPER (what the user wants to emulate IN SPIRIT, arxiv 2603.00492v2 "ArtiFixer"):
- Enhances/extends 3DGS by training a flow-matching VIDEO diffusion model (Wan 2.1 T2V, 14B) that
  maps degraded 3DGS renders -> clean artifact-free novel views. Key idea = OPACITY MIXING:
  z_mix = O_z * z_deg + (1-O_z) * noise, so low-opacity (uncovered) pixels get generative fill,
  high-opacity pixels are preserved. Camera control via Plucker raymaps; reference views via
  cross-attention. A causal auto-regressive student is distilled (DMD) for fast long-video gen.
  Variants: ArtiFixer (render from generator), ArtiFixer3D (generate views once, then DISTILL
  back into 3DGS via standard 3DGS optimization), ArtiFixer3D+ (re-apply AR model as post).
- COST: trained on 128 H100 GPUs, ~15,000 GPU-hours. 14B model. THIS TRAINING IS OUT OF REACH.
  The PORTABLE ideas are: (a) opacity/coverage map as the where-to-enhance signal [we have alpha
  + gap_mask], (b) generative fill of low-coverage regions, (c) DISTILL generated views back
  into the gaussians via a 3DGS optimization loop [ArtiFixer3D], (d) reference-view + camera
  conditioning for consistency. The MODEL must be swapped for a 24GB-feasible generative prior.

DELIVERABLE OF THIS STUDY: a concrete, staged design for introducing "coverage/blur-driven
generative novel-view enhancement that gets distilled back into the original splat" into THIS
stack, fitting 24GB, reusing our alpha/coverage3d/depth seams, and being buildable incrementally.
`

// ---------- schemas ----------
const PAPER_SCHEMA = {
  type: 'object', additionalProperties: true,
  required: ['portable_ideas', 'requires_heavy_scale', 'distill_to_3d_procedure', 'coverage_signal_mapping', 'web_access_ok'],
  properties: {
    portable_ideas: { type: 'array', items: { type: 'object', additionalProperties: true,
      properties: { idea: {type:'string'}, why_it_helps: {type:'string'}, adaptation_for_24gb: {type:'string'} } } },
    requires_heavy_scale: { type: 'array', items: { type: 'object', additionalProperties: true,
      properties: { component: {type:'string'}, why_infeasible: {type:'string'}, lightweight_substitute: {type:'string'} } } },
    distill_to_3d_procedure: { type: 'string', description: 'Exact ArtiFixer3D distill-back-into-gaussians loop, step by step' },
    coverage_signal_mapping: { type: 'string', description: 'How they pick regions, mapped onto OUR alpha map + coverage3d gap_mask' },
    consistency_mechanisms: { type: 'array', items: {type:'string'} },
    key_risks_when_downscaled: { type: 'array', items: {type:'string'} },
    web_access_ok: { type: 'boolean' },
  },
}

const STACK_SCHEMA = {
  type: 'object', additionalProperties: true,
  required: ['has_gaussian_optimizer', 'differentiable_params', 'coverage_signals', 'recommended_seam'],
  properties: {
    modules: { type:'array', items: { type:'object', additionalProperties:true,
      properties: { path:{type:'string'}, role:{type:'string'}, relevance:{type:'string'} } } },
    has_gaussian_optimizer: { type:'object', additionalProperties:true,
      properties: { present:{type:'boolean'}, note:{type:'string'} } },
    differentiable_params: { type:'string', description:'what is already differentiable / in GPU memory and what it would take to enable an optimizer' },
    coverage_signals: { type:'array', items:{type:'string'} },
    depth_prior: { type:'string' },
    scene_io: { type:'string' },
    recommended_seam: { type:'string', description:'where a new enhance/ module + 3DGS finetune loop should plug in, by file/protocol' },
    reusable_assets: { type:'array', items:{type:'string'} },
    gaps_or_unknowns: { type:'array', items:{type:'string'} },
  },
}

const LIT_SCHEMA = {
  type:'object', additionalProperties:true,
  required:['methods','closest_analog','recommendation','web_access_ok'],
  properties: {
    methods: { type:'array', items: { type:'object', additionalProperties:true, properties: {
      name:{type:'string'}, year:{type:'string'}, one_line:{type:'string'},
      generative_prior:{type:'string'}, vram_inference:{type:'string'},
      trainable_or_tunable_on_24gb:{type:'string'}, distills_to_3d:{type:'boolean'},
      multiview_consistency_approach:{type:'string'}, weights_available:{type:'string'},
      closeness_to_our_need:{type:'string'} } } },
    closest_analog: { type:'string' },
    recommendation: { type:'string' },
    web_access_ok: { type:'boolean' },
  },
}

const PRIOR_SCHEMA = {
  type:'object', additionalProperties:true,
  required:['generative_priors','gsplat_finetune','recommended_prior','web_access_ok'],
  properties: {
    generative_priors: { type:'array', items: { type:'object', additionalProperties:true, properties: {
      name:{type:'string'}, kind:{type:'string', description:'image-inpaint | single-step-restore | short-video | other'},
      params:{type:'string'}, vram_inference_24gb:{type:'string'}, lora_or_finetune_on_24gb:{type:'string'},
      multiview_consistent:{type:'string'}, camera_controllable:{type:'string'}, license_weights:{type:'string'} } } },
    gsplat_finetune: { type:'object', additionalProperties:true, properties: {
      vram_estimate_single_scene:{type:'string'}, densification_api:{type:'string'},
      mixed_precision_notes:{type:'string'}, how_to_constrain_to_gap_regions:{type:'string'} } },
    recommended_prior: { type:'string' },
    web_access_ok: { type:'boolean' },
  },
}

const CANDIDATE_SCHEMA = {
  type:'object', additionalProperties:true,
  required:['name','architecture','pipeline_steps','plugs_into','trigger_signal','generative_prior','vram_budget','total_vram_gb','fits_24gb','first_experiment'],
  properties: {
    name:{type:'string'}, stance:{type:'string'},
    architecture:{type:'string', description:'detailed prose: how it works end to end'},
    pipeline_steps:{type:'array', items:{type:'string'}},
    plugs_into:{type:'array', items:{type:'string'}, description:'OUR modules/files/protocols it hooks into'},
    trigger_signal:{type:'string', description:'how it detects blurry/under-observed areas using our alpha/coverage3d/depth'},
    generative_prior:{type:'string'},
    needs_training_or_tuning:{type:'string'},
    vram_budget:{type:'array', items:{type:'object', additionalProperties:true, properties:{component:{type:'string'}, gb:{type:'number'}, staged:{type:'boolean'}}}},
    total_vram_gb:{type:'number', description:'peak concurrent VRAM after staging'},
    fits_24gb:{type:'boolean'},
    multiview_consistency_plan:{type:'string'},
    expected_quality:{type:'string'},
    risks:{type:'array', items:{type:'string'}},
    first_experiment:{type:'string'},
  },
}

const VERDICT_SCHEMA = {
  type:'object', additionalProperties:true,
  required:['candidate_name','vram_recomputed_gb','vram_fits_24gb','keep_or_kill','score_0_10'],
  properties: {
    candidate_name:{type:'string'},
    vram_recomputed_gb:{type:'number', description:'your independent recompute of peak VRAM'},
    vram_fits_24gb:{type:'boolean'},
    vram_notes:{type:'string'},
    consistency_verdict:{type:'string', description:'will it be multiview-consistent or just per-frame hallucination? be skeptical'},
    integration_verdict:{type:'string', description:'does it fit our seams without a rewrite?'},
    obtainability_verdict:{type:'string', description:'are weights actually available + runnable without 128-H100 training? license ok?'},
    keep_or_kill:{type:'string', enum:['keep','keep-with-changes','kill']},
    required_changes:{type:'array', items:{type:'string'}},
    score_0_10:{type:'number'},
  },
}

const SYNTH_SCHEMA = {
  type:'object', additionalProperties:true,
  required:['recommended_primary','recommended_fallback','rationale','module_layout','enhancement_loop','vram_budget_final','milestone0_experiment','roadmap','open_questions','risks'],
  properties: {
    recommended_primary:{type:'string'},
    recommended_fallback:{type:'string'},
    rationale:{type:'string'},
    module_layout:{type:'array', items:{type:'object', additionalProperties:true, properties:{path:{type:'string'}, purpose:{type:'string'}}}},
    enhancement_loop:{type:'array', items:{type:'string'}, description:'end-to-end steps: detect -> sample views -> generate -> distill -> re-evaluate'},
    vram_budget_final:{type:'array', items:{type:'object', additionalProperties:true, properties:{component:{type:'string'}, gb:{type:'number'}, staged:{type:'boolean'}}}},
    total_vram_gb:{type:'number'},
    milestone0_experiment:{type:'string', description:'the SMALLEST end-to-end thing to validate the idea on our scene'},
    roadmap:{type:'array', items:{type:'object', additionalProperties:true, properties:{phase:{type:'string'}, deliverable:{type:'string'}, effort:{type:'string'}}}},
    open_questions:{type:'array', items:{type:'string'}},
    risks:{type:'array', items:{type:'object', additionalProperties:true, properties:{risk:{type:'string'}, mitigation:{type:'string'}}}},
    whats_missing:{type:'string', description:'self-critique: what this study did NOT resolve and should be checked next'},
  },
}

// ---------- PHASE 1: UNDERSTAND (barrier — all four feed the design panel) ----------
phase('Understand')
const webNote = 'You have web access via deferred tools: FIRST call ToolSearch with query "select:WebSearch,WebFetch" to load them, then use them. If web fails, proceed from the embedded CONTEXT + your knowledge and set web_access_ok=false.'

const understanding = await parallel([
  () => agent(
    `${CONTEXT}\n\nTASK: Deep-read the reference paper (arxiv 2603.00492v2 "ArtiFixer", https://arxiv.org/html/2603.00492v2). ${webNote}\n`+
    `Focus on extracting what is PORTABLE to a single-24GB-GPU per-scene setting versus what fundamentally needs the 128-H100 training run. I especially need: (1) the precise ArtiFixer3D "distill generated views back into the gaussians" procedure (what loss, what views, how many, how the 3DGS optimization is run); (2) the opacity-mixing trick mapped onto OUR rendered alpha map and coverage3d gap_mask; (3) the consistency mechanisms (reference views, Plucker raymaps, causal AR) and which are even needed if we only generate a handful of views per scene; (4) honest risks when this is radically downscaled (mode collapse, hallucinated geometry, multiview inconsistency). Return structured output.`,
    { label:'paper-deepread', phase:'Understand', schema: PAPER_SCHEMA }
  ),
  () => agent(
    `${CONTEXT}\n\nTASK: Produce a precise map of OUR codebase for the purpose of adding a splat-enhancement (generative-fill + distill-back) capability. Read the real files under src/gaussian_robot/: backends/gsplat_renderer.py, render/base.py, render/camera.py, metrics/coverage3d.py, metrics/coverage.py, depth/estimator.py, splat/loaders.py, splat/scene.py, splat/capture_poses.py, session.py (esp. the densify deliverable around lines 600-860), nav/explorer.py, config.py. `+
    `Confirm/correct: is there ANY gaussian optimization today? exactly what differentiable params are resident and what minimal change enables an optimizer (grad on GaussianCloud tensors + Adam + gsplat strategy)? what coverage/blur signals can drive region selection? where (which file/protocol) should a new enhance/ module + 3DGS finetune loop plug in so it respects the pluggable architecture? Inventory reusable assets (alpha maps, gap_centers, capture_poses as reference views, depth prior). Return structured output. Do NOT design the solution — just map the ground truth.`,
    { label:'stack-map', phase:'Understand', schema: STACK_SCHEMA }
  ),
  () => agent(
    `${CONTEXT}\n\nTASK: Survey the LITERATURE for lightweight, 24GB-feasible analogs of "generative enhancement of 3DGS/NeRF in under-observed/blurry regions, distilled back into 3D". ${webNote}\n`+
    `Cover at least: Difix3D+ (NVIDIA, single-step SD-Turbo diffusion fixer for 3DGS artifacts), 3DGS-Enhancer (video-diffusion view consistency), Deceptive-NeRF/Deceptive-3DGS, GANeRF, NeRFLiX, DiffBIR / StableSR (general restoration), ReconFusion / CAT3D (sparse-view diffusion priors), Instruct-GS2GS, RealmDreamer, SDEdit-style 3D editing, and any 2024-2026 "diffusion prior + per-scene 3DGS distillation" work. For each: generative prior used, inference VRAM, whether it is trainable/tunable on a single 24GB GPU, whether it distills into 3D, its multiview-consistency approach, weight availability/license, and closeness to our exact need. Identify the CLOSEST analog we should base our design on and why. Return structured output.`,
    { label:'lit-survey', phase:'Understand', schema: LIT_SCHEMA }
  ),
  () => agent(
    `${CONTEXT}\n\nTASK: Survey concrete 24GB-VRAM-feasible GENERATIVE PRIORS we could actually run/finetune as the "fixer/filler", plus the mechanics + memory of a per-scene gsplat fine-tune. ${webNote}\n`+
    `Generative priors to size up (params, inference VRAM at ~512-1024px on 24GB, LoRA/full finetune feasibility on 24GB, multiview consistency, camera controllability, license + weight availability): SD1.5/SD2 inpainting, SDXL inpainting, SD-Turbo / SDXL-Turbo / SDXL-Lightning (single/few-step), Difix's SD-Turbo backbone, Stable Video Diffusion (SVD/SVD-XT), CogVideoX-2B, Wan2.1-1.3B, AnimateDiff, MoGe/depth-warp+classical-inpaint (a no-big-model baseline). `+
    `For the gsplat fine-tune: realistic VRAM for optimizing a ~1-3M-gaussian single scene with Adam + densification on 24GB; the gsplat densification/strategy API; mixed precision notes; and how to CONSTRAIN optimization to only the gap/low-alpha regions (mask the loss, freeze far gaussians, or add gaussians only near gap_centers). Recommend the single best prior for our constraints. Return structured output.`,
    { label:'prior-survey', phase:'Understand', schema: PRIOR_SCHEMA }
  ),
])

const [paper, stack, lit, priors] = understanding
const u = JSON.stringify({ paper, stack, lit, priors }, null, 1)
log('Understand phase complete — designing candidate architectures')

// ---------- PHASE 2+3: DESIGN (judge panel) -> VERIFY (adversarial), pipelined per candidate ----------
const STANCES = [
  { name:'A: Minimal single-image fixer + distill-back (Difix-style)',
    stance:'Lowest risk. A single-step image diffusion restorer (SD-Turbo class, optionally LoRA-tuned on our own renders) takes a degraded render from a sampled novel pose near a coverage gap, the alpha map gates how much it changes (opacity-mixing analog), and the cleaned images become pseudo-GT to fine-tune the gaussians with a masked photometric loss. Use capture cameras as reference/anchor frames. No video model. Favor robustness + buildability over maximal consistency.' },
  { name:'B: Short multiview-consistent video diffusion + distill-back',
    stance:'Consistency-first. A small short-video diffusion prior (SVD / CogVideoX-2B / Wan-1.3B class) generates a SHORT orbit/sweep clip through the gap region so frames are mutually consistent, then distill the clip into the gaussians. Address camera control + reference conditioning within 24GB (LoRA, low frame count, staged loading). Be explicit about how 24GB is respected.' },
  { name:'C: No-big-model geometric baseline (depth-warp + inpaint + masked finetune)',
    stance:'Cheapest + most controllable. Use Depth Anything 3 + rendered depth to warp existing observed pixels into novel poses, fill only the disoccluded/low-alpha holes with a classical or small inpainter, and fine-tune gaussians on the warped+inpainted pseudo-views with strict masking. Minimal hallucination, no large generative model. This is the strong baseline every fancier method must beat.' },
  { name:'D: Score-distillation (SDS) directly into gaussians at gap regions',
    stance:'Most integrated. Skip explicit pseudo-images: run SDS/variational score distillation from a 24GB-feasible 2D diffusion prior as an extra gradient on the gaussians, applied only in gap/low-alpha regions, alongside the photometric loss on real captures. Per-scene optimization, no dataset. Be explicit about SDS instability, VRAM of backprop-through-UNet, and how to bound it to gaps.' },
]

const reviewed = await pipeline(
  STANCES,
  // STAGE 1 — design
  (s) => agent(
    `${CONTEXT}\n\n=== PHASE-1 RESEARCH FINDINGS (paper, our stack map, lightweight-analog literature, 24GB priors) ===\n${u}\n\n`+
    `TASK: Design ONE concrete candidate architecture for introducing generative novel-view enhancement of blurry/under-observed areas into gaussian-robot, then distilling it back into the original splat. Your assigned stance:\n"${s.name}" — ${s.stance}\n\n`+
    `Be specific and buildable: name the exact generative prior + weights, the exact trigger (which of our signals: per-pixel alpha, coverage3d gap_centers/gap_mask, depth disagreement, blur metric), how novel poses are sampled, the distill-back loss + how it is masked to only touch under-observed gaussians, what (if anything) must be trained/LoRA-tuned and on what data, and a precise VRAM budget broken down by component WITH staging (what is loaded when) so the PEAK fits 24GB. Include a minimal first experiment. Respect our pluggable protocols and conventions. Return structured output.`,
    { label:`design:${s.name.slice(0,1)}`, phase:'Design', schema: CANDIDATE_SCHEMA }
  ),
  // STAGE 2 — adversarial verify
  (cand, s) => agent(
    `${CONTEXT}\n\nYou are an adversarial reviewer. Try to REFUTE the feasibility of this candidate for enhancing a 3DGS scene on a SINGLE 24GB GPU. Default to skepticism.\n\n`+
    `=== CANDIDATE (stance "${s.name}") ===\n${JSON.stringify(cand, null, 1)}\n\n`+
    `=== PHASE-1 FINDINGS for cross-checking VRAM + weight availability + consistency ===\n${u}\n\n`+
    `Independently RECOMPUTE peak VRAM (model weights + activations + KV/latents + the gsplat optimizer state for a ~1-3M-gaussian scene + Adam moments + densification headroom) and state whether it truly fits 24GB after staging. Then judge: (1) multiview consistency — will distilling these generated views SHARPEN the splat or inject floaters/contradictions? (2) integration — does it fit our seams (gsplat_renderer GaussianCloud, coverage3d, depth, capture_poses) without a rewrite? (3) obtainability — are the weights actually downloadable + runnable WITHOUT a 128-H100 training run, and is the license usable for a proprietary project? Give keep/keep-with-changes/kill, required changes, and a 0-10 score. Return structured output.`,
    { label:`verify:${s.name.slice(0,1)}`, phase:'Verify', schema: VERDICT_SCHEMA }
  ),
)

const candidates = reviewed.map((r) => r).filter(Boolean)
// pair designs with verdicts by index (pipeline preserves order)
const paired = STANCES.map((s, i) => ({ stance: s, verdict: reviewed[i] }))
log('Candidates designed + adversarially verified — synthesizing recommendation')

// ---------- PHASE 4: SYNTHESIZE ----------
phase('Synthesize')
// rebuild candidate+verdict pairs from the pipeline: re-run is unnecessary; we pass both arrays.
const designsJson = JSON.stringify(reviewed, null, 1)
const synth = await agent(
  `${CONTEXT}\n\n=== PHASE-1 RESEARCH FINDINGS ===\n${u}\n\n`+
  `=== PHASE-2/3: FOUR ADVERSARIALLY-VERIFIED CANDIDATE VERDICTS (one per stance A/B/C/D) ===\n${designsJson}\n\n`+
  `TASK: Synthesize the recommendation for HOW to introduce blurry/under-observed-area generative enhancement that distills back into the original splat, on a single 24GB GPU, into THIS codebase.\n`+
  `Pick a recommended PRIMARY approach and a FALLBACK (you may graft the best parts across candidates — e.g. use the geometric baseline as the safety net under a learned fixer). Give: the rationale (why this beats the others given 24GB + consistency + buildability + license), the concrete new module_layout (files under src/gaussian_robot/, respecting pluggable protocols), the end-to-end enhancement_loop steps (detect gaps via alpha/coverage3d -> sample novel poses -> generate/fix views -> distill into gaussians with masked loss -> re-evaluate coverage), a FINAL staged VRAM budget that fits 24GB, the smallest milestone-0 experiment to validate the idea on our existing scene, a phased roadmap, open questions, and risk/mitigation pairs. Also self-critique what this study did NOT resolve. Be decisive and specific. Return structured output.`,
  { label:'synthesis', phase:'Synthesize', schema: SYNTH_SCHEMA, effort:'xhigh' }
)

return { paper, stack, lit, priors, candidates: reviewed, synthesis: synth }
