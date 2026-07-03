export const meta = {
  name: 'difix-gap-and-artifixer-24gb',
  description: 'Find how to close the gap with the official Difix3D+ recipe, and scope an ArtiFixer-style fill at 24GB using alternative diffusion backbones — adversarially verified.',
  whenToUse: 'Research+design pass for stronger 3DGS generative gap-fill on a single 24GB GPU.',
  phases: [
    { title: 'Map', detail: 'read official Difix3D trainer + our impl -> gap analysis' },
    { title: 'Survey', detail: 'fan out: video-diffusion, multiview-NVS, 3DGS-enhancers, ArtiFixer mechanisms' },
    { title: 'Verify', detail: 'adversarially verify 24GB inference feasibility + availability per candidate' },
    { title: 'Synthesize', detail: 'two ranked plans + recommendation + completeness critic' },
  ],
}

const REPO = '/mnt/wd/gaussian-bot'
const WEB = 'You have web access: use WebSearch and WebFetch (load them via ToolSearch with "select:WebSearch,WebFetch" if not already loaded). Cite every load-bearing claim with a URL or file:line.'

const OFFICIAL_SCHEMA = { type:'object', additionalProperties:false, properties:{
  pose_selection:{type:'string', description:'how pseudo/novel views are chosen (pose interpolation? near training? how far does it push?)'},
  progressive_schedule:{type:'string', description:'how many stages, how spatial extent grows, iters per stage, when Difix is re-applied'},
  optimized_params:{type:'string', description:'which 3DGS params are optimized during distillation'},
  geometry_moves:{type:'boolean', description:'does it move/add geometry (means/scales/quats + densification) or freeze it?'},
  densification:{type:'string'}, lrs:{type:'string'}, loss:{type:'string', description:'photometric loss terms (L2/L1/LPIPS/SSIM/mask?)'},
  reference_selection:{type:'string', description:'how the clean reference view(s) are picked for difix_ref'},
  realtime_enhancer:{type:'string', description:'is Difix also used as a final post-render enhancer?'},
  citations:{type:'array', items:{type:'string'}},
}, required:['pose_selection','progressive_schedule','optimized_params','geometry_moves','loss','citations'] }

const OURS_SCHEMA = { type:'object', additionalProperties:false, properties:{
  pose_selection:{type:'string'}, progressive_schedule:{type:'string'}, optimized_params:{type:'string'},
  geometry_moves:{type:'boolean'}, densification:{type:'string'}, lrs:{type:'string'}, loss:{type:'string'},
  reference_handling:{type:'string'}, gate:{type:'string'}, citations:{type:'array', items:{type:'string'}},
}, required:['pose_selection','optimized_params','geometry_moves','loss','citations'] }

const GAP_SCHEMA = { type:'object', additionalProperties:false, properties:{
  divergences:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    aspect:{type:'string'}, official:{type:'string'}, ours:{type:'string'},
    likely_cost:{type:'string', description:'how this divergence likely costs fill quality on a 24GB single-floor interior'},
    fix:{type:'string', description:'concrete change to our repo (file/function) to close it'},
    leverage:{type:'string', enum:['high','medium','low']}, risk:{type:'string', enum:['high','medium','low']},
  }, required:['aspect','official','ours','fix','leverage','risk']}},
  summary:{type:'string'},
}, required:['divergences','summary'] }

const CANDIDATES_SCHEMA = { type:'object', additionalProperties:false, properties:{
  candidates:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    name:{type:'string'}, family:{type:'string'}, params:{type:'string'}, base_model:{type:'string'},
    public_weights:{type:'boolean'}, weights_repo:{type:'string'}, public_code:{type:'boolean'}, code_url:{type:'string'},
    conditioning:{type:'array', items:{type:'string'}, description:'reference-view / camera-pose / opacity-mask / depth / text'},
    claimed_min_vram:{type:'string'}, license:{type:'string'},
    relevance:{type:'string', description:'why it fits the ArtiFixer/novel-view-fill role for 3DGS'},
  }, required:['name','params','public_weights','public_code','conditioning']}},
  notes:{type:'string'},
}, required:['candidates'] }

const MECH_SCHEMA = { type:'object', additionalProperties:false, properties:{
  opacity_mixing:{type:'string', description:'exactly how ArtiFixer mixes input-RGB latent with noise via rendered opacity maps'},
  camera_control:{type:'string'}, bidirectional_vs_ar:{type:'string'}, distillation:{type:'string', description:'DMD distillation of the bidirectional teacher into the causal AR student'},
  conditioning_inputs:{type:'string'}, what_a_smaller_backbone_must_replicate:{type:'array', items:{type:'string'}},
  citations:{type:'array', items:{type:'string'}},
}, required:['opacity_mixing','camera_control','distillation','what_a_smaller_backbone_must_replicate','citations'] }

const VERDICT_SCHEMA = { type:'object', additionalProperties:false, properties:{
  name:{type:'string'}, fits_24gb:{type:'string', enum:['yes','marginal','no']},
  inference_vram_estimate:{type:'string', description:'weights (fp16/bf16) + activations for a few ~512px frames; show the arithmetic'},
  evidence:{type:'array', items:{type:'string'}}, blockers:{type:'array', items:{type:'string'}},
  conditioning_fit:{type:'string', description:'does it actually support reference-view / camera / opacity conditioning we need?'},
  offload_options:{type:'string', description:'CPU offload / quantization / tiling that could make it fit, and the speed cost'},
  recommended:{type:'boolean'}, how_to_use_for_gapfill:{type:'string'},
}, required:['name','fits_24gb','recommended'] }  // minimal required so a compact retry still validates

const PLAN_SCHEMA = { type:'object', additionalProperties:false, properties:{
  title:{type:'string'},
  steps:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    change:{type:'string'}, rationale:{type:'string'}, components:{type:'string', description:'files/functions or new modules'},
    risk:{type:'string', enum:['high','medium','low']}, expected_effect:{type:'string'},
  }, required:['change','rationale','expected_effect','risk']}},
  overall_feasibility:{type:'string'}, vram_budget:{type:'string'}, open_risks:{type:'array', items:{type:'string'}},
}, required:['title','steps','overall_feasibility'] }

const CRITIC_SCHEMA = { type:'object', additionalProperties:false, properties:{
  missing:{type:'array', items:{type:'string'}}, overlooked_models:{type:'array', items:{type:'string'}},
  unverified_claims:{type:'array', items:{type:'string'}}, verdict:{type:'string'},
}, required:['missing','unverified_claims','verdict'] }

// ---------------- Map ----------------
phase('Map')
const [official, ours] = await parallel([
  () => agent(`Extract the EXACT 3DGS gap-fill algorithm of the OFFICIAL Difix3D+ (NVIDIA, CVPR 2025). ${WEB}
Fetch and read these raw files (raw.githubusercontent.com/nv-tlabs/Difix3D/main/<path>):
  - examples/gsplat/simple_trainer_difix3d.py  (the actual 3DGS progressive trainer — READ CAREFULLY)
  - examples/gsplat/utils.py and examples/gsplat/datasets/traj.py (pose/trajectory handling)
  - src/pipeline_difix.py (the __call__ signature + reference handling)
Also read the paper: research.nvidia.com/labs/toronto-ai/difix3d/ and arxiv.org/html/2503.01774v1 for the progressive-update schedule and losses.
Determine precisely: (1) how pseudo/novel views are SELECTED and HOW FAR they are pushed from training cams; (2) the PROGRESSIVE schedule — number of stages, how the reconstructed extent grows, iters per stage, cadence of re-applying Difix; (3) WHAT 3DGS params are optimized during distillation and crucially whether GEOMETRY MOVES / DENSIFIES or is frozen, with LRs; (4) the loss (L1/L2/LPIPS/SSIM, masked?); (5) reference-view selection for difix_ref; (6) whether Difix is also used as a real-time post enhancer. Cite file:line / section.`, { label:'read:official-difix3d', phase:'Map', schema: OFFICIAL_SCHEMA }),
  () => agent(`Inventory OUR repo's current gap-fill so it can be compared to the official Difix3D+. Read these files under ${REPO}:
  - src/gaussian_robot/enhance/orchestrator.py  (fill_gaps_scene, _progressive_distill, synthesize_near_view_poses, _fill_gap_views, the PSNR gate)
  - src/gaussian_robot/enhance/distiller.py  (GaussianDistiller: which params it optimizes, the densify flag, default LRs, the loss, MCMC)
  - src/gaussian_robot/enhance/fillers/diffusion.py  (DiffusionFiller: difix_ref, reference handling, recomposite)
Report precisely: pose selection (and the perturb/dolly magnitude), progressive schedule (rounds, perturb growth, iters/round, best-round selection), which params move (is geometry frozen? densification on/off?), LRs, the loss, reference handling, and the held-out gate. Cite file:line.`, { label:'read:our-impl', phase:'Map', schema: OURS_SCHEMA }),
])
const gap = await agent(`Produce a precise GAP ANALYSIS between the official Difix3D+ recipe and ours. For every meaningful divergence, state official vs ours, the likely quality cost on a 24GB single-floor INTERIOR scene (gaps are occupied-but-unseen surface; little valid content far from training views), the concrete fix in our repo, and rank by leverage and risk. Be skeptical: some of our simplifications (frozen geometry, no densify, conservative dolly) were deliberate to avoid a measured -1.5 to -4.8 dB regression — say where matching the official recipe would REINTRODUCE that risk vs genuinely help.
OFFICIAL: ${JSON.stringify(official)}
OURS: ${JSON.stringify(ours)}`, { label:'gap-analysis', phase:'Map', schema: GAP_SCHEMA })

// ---------------- Survey ----------------
phase('Survey')
const survey = await parallel([
  () => agent(`Survey VIDEO-DIFFUSION models that could play the ArtiFixer generative role but FIT A SINGLE 24GB GPU at inference (RTX 4090). ${WEB}
ArtiFixer itself is Wan2.1-T2V-14B (16.9B params, A100-80GB) — too big. Find SMALLER alternatives and camera-controllable variants: Wan2.1-1.3B, CogVideoX-2B/5B, Stable Video Diffusion (SVD/SVD-XT), LTX-Video, Mochi-1, AnimateDiff, and any camera-control add-ons (MotionCtrl, CameraCtrl, CamI2V, ViewCrafter, SVD-MV). For each: params, base arch, public weights+code (HF repo / github), supported conditioning (reference image / camera pose / opacity-mask / depth / text), claimed minimum inference VRAM, license. Be exhaustive; prefer ones with released camera-control weights.`, { label:'survey:video-diffusion', phase:'Survey', schema: CANDIDATES_SCHEMA }),
  () => agent(`Survey MULTIVIEW / NOVEL-VIEW-SYNTHESIS diffusion models suited to generating consistent novel views to repair 3DGS, that fit 24GB at inference. ${WEB}
Cover: Zero123 / Zero123-XL / Stable-Zero123, Zero123++, SyncDreamer, MVDream, Wonder3D, EscherNet, SV3D, CAT3D / ReconFusion (and any open reimplementations), ViewCrafter, GenWarp, Free3D, EpiDiff. For each: params, public weights+code, conditioning (reference image(s) / explicit camera pose / depth / mask), claimed inference VRAM, license, and how directly it supports "given this scene + a target camera, render a plausible view." Flag which give true camera-pose control (vs fixed orbit).`, { label:'survey:multiview-nvs', phase:'Survey', schema: CANDIDATES_SCHEMA }),
  () => agent(`Survey work that DIRECTLY uses generative priors to enhance/repair 3DGS or NeRF (the same problem class as Difix3D+ / ArtiFixer), with public code, runnable on 24GB. ${WEB}
Cover: 3DGS-Enhancer, Difix3D+ itself, GaussianObject, FSGS, SparseGS, Deceptive-NeRF/3DGS, DiffusioNeRF, ReconFusion, Nerfbusters, GANeRF, RaDe-GS, and any "diffusion pseudo-view distillation for 3DGS" papers from 2024-2026. For each: the generative backbone used, whether weights+code are public, the distillation mechanism (how 2D fixes go back into 3D), and VRAM. Identify the most reusable building blocks for OUR 24GB pipeline.`, { label:'survey:3dgs-enhancers', phase:'Survey', schema: CANDIDATES_SCHEMA }),
  () => agent(`Extract ArtiFixer's CORE TECHNICAL MECHANISMS precisely, so a smaller backbone could replicate them. ${WEB}
Read the project page research.nvidia.com/labs/sil/projects/artifixer/ and the model card huggingface.co/nvidia/ArtiFixer, plus any arxiv. Detail: (1) the OPACITY-MIXING strategy — exactly how it encodes input RGB to latent and mixes with Gaussian noise using rendered opacity maps (the SDEdit-like init), at which timestep, and why; (2) camera-control conditioning signal; (3) bidirectional video teacher vs causal auto-regressive student; (4) the DMD distillation; (5) all conditioning inputs (opacity, camera, reference views, text). Then list what a 24GB-feasible backbone MUST replicate to get most of the benefit, and what can be dropped.`, { label:'survey:artifixer-mechanisms', phase:'Survey', schema: MECH_SCHEMA }),
])

const pools = [survey[0], survey[1], survey[2]].filter(Boolean)
const mech = survey[3]
const seen = new Set(); const candidates = []
for (const pool of pools) for (const c of (pool.candidates || [])) {
  const k = (c.name || '').toLowerCase().trim()
  if (!k || seen.has(k)) continue
  seen.add(k); candidates.push(c)
}
log(`survey found ${candidates.length} distinct candidate models`)
if (candidates.length > 16) { log(`capping ${candidates.length} -> 16 for verify (others noted in survey output)`); candidates.length = 16 }

// ---------------- Verify (adversarial, per candidate) ----------------
phase('Verify')
const verdicts = (await parallel(candidates.map(c => () =>
  agent(`Adversarially verify whether ${c.name} (${c.params||'?'} params, family ${c.family||'?'}) can run INFERENCE on a SINGLE 24GB GPU (RTX 4090) for the splat gap-fill / novel-view-repair role, AND whether it supports the conditioning we need (a clean reference view and/or an explicit target camera pose, ideally an opacity/inpaint mask). ${WEB}
Default to fits_24gb='no' or 'marginal' unless concrete evidence shows it fits. Show the VRAM arithmetic: weights in fp16/bf16 + VAE + text encoder + activations for the realistic task (a handful of ~512px frames or one image). Note offload/quantization/tiling that could rescue it and the speed cost. IMPORTANT for valid structured output: keep EVERY string field to a SINGLE LINE under ~80 words — no literal newlines, tabs, or backslashes inside any field (they break JSON serialization). Surveyed metadata to verify (don't trust blindly): ${JSON.stringify(c)}`,
    { label:`verify:${(c.name||'cand').slice(0,28)}`, phase:'Verify', schema: VERDICT_SCHEMA })
))).filter(Boolean)

const feasible = verdicts.filter(v => v.fits_24gb !== 'no')
const recommended = feasible.filter(v => v.recommended)
log(`verify: ${feasible.length}/${verdicts.length} fit 24GB (>=marginal); ${recommended.length} recommended`)

// ---------------- Synthesize ----------------
phase('Synthesize')
const [planA, planB] = await parallel([
  () => agent(`PLAN A — CLOSE THE GAP WITH THE OFFICIAL DIFIX3D+ RECIPE, on 24GB, KEEPING the no-regression guarantee. Using the gap analysis, give a ranked, concrete change-list to our repo (file/function level) that would produce a STRONGER fill while staying safe on a single-floor interior. Distinguish changes that are pure wins from changes that reintroduce the measured regression risk (frozen-geometry was deliberate). Include: localized densification near gaps vs full, LPIPS/perceptual loss, more anchors, pose/trajectory strategy, multi-reference conditioning, real-time enhancer pass. Each step: change, rationale, components, risk, expected effect.
GAP ANALYSIS: ${JSON.stringify(gap)}`, { label:'plan-A:close-difix-gap', phase:'Synthesize', schema: PLAN_SCHEMA }),
  () => agent(`PLAN B — APPROXIMATE ARTIFIXER ON 24GB with an alternative diffusion backbone. Using ArtiFixer's mechanisms and ONLY the VERIFIED-feasible models, pick the single best concrete backbone (and a fallback) and design how to wire it: camera-pose conditioning, the opacity-mixing/SDEdit init from rendered opacity, multi-frame consistency, and distillation back into our 3DGS GaussianDistiller. Give a staged plan (smallest useful first), a frank feasibility verdict for 24GB, VRAM budget, and open risks. If nothing beats difix_ref meaningfully at 24GB, SAY SO and explain why.
ARTIFIXER MECHANISMS: ${JSON.stringify(mech)}
FEASIBLE MODELS (verified): ${JSON.stringify(recommended.length ? recommended : feasible)}`, { label:'plan-B:artifixer-24gb', phase:'Synthesize', schema: PLAN_SCHEMA }),
])

const report = await agent(`Write a clear, decision-oriented MARKDOWN report titled "Closing the Difix3D+ gap & ArtiFixer at 24GB". Sections: (1) TL;DR recommendation — which track to pursue first and why; (2) Gap vs official Difix3D+ — the ranked fixes (Plan A); (3) ArtiFixer-style at 24GB — best backbone + wiring + feasibility (Plan B); (4) A staged roadmap with checkpoints; (5) What stays the same (the no-regress gate, frozen-geometry caution). Be concrete and honest about expected dB gains and risks. Use tables where they help.
GAP: ${JSON.stringify(gap)}
PLAN A: ${JSON.stringify(planA)}
PLAN B: ${JSON.stringify(planB)}
VERDICTS: ${JSON.stringify(verdicts)}`, { label:'synthesis-report', phase:'Synthesize' })

const critic = await agent(`Completeness critic. Review the report below for: modalities/models overlooked (NAME them specifically, e.g. a camera-control diffusion or a 3DGS-enhancer we missed), VRAM claims that are asserted but not arithmetic-backed, and assumptions that are likely wrong for a single-floor interior at 24GB. Be specific and adversarial.
REPORT: ${report}`, { label:'completeness-critic', phase:'Synthesize', schema: CRITIC_SCHEMA })

return {
  gap_analysis: gap,
  candidates_verified: verdicts.length,
  feasible_models: feasible.map(v => ({ name:v.name, fits:v.fits_24gb, rec:v.recommended })),
  plan_A: planA,
  plan_B: planB,
  report,
  critic,
}
