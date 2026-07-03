export const meta = {
  name: 'wire-diffusion-filler',
  description: 'Wire a real generative ViewFiller (diffusion, geometric fallback) into the splat-enhancement pipeline and run it on the office scene',
  phases: [
    { title: 'Probe', detail: 'env+weights feasibility, Difix/SD-Turbo API, code seams' },
    { title: 'Build filler', detail: 'enhance/fillers: diffusion + geometric ViewFiller' },
    { title: 'Wire distill', detail: 'gap poses + masked gap-localized distillation in the orchestrator' },
    { title: 'Run & verify', detail: 'run on the office scene; before/after at gap views; no global regression' },
  ],
}

const REPO = '/mnt/wd/gaussian-bot'
const SCENE = '/mnt/archive/datasets/ufficio360-35a39133-e1f2-4426-86d4-a3d7a00614ee-PIC'
const PY = `${REPO}/.venv/bin/python`

const CONTEXT = `
REPO: ${REPO}  (run python as ${PY}; lint ${REPO}/.venv/bin/ruff; types ${REPO}/.venv/bin/mypy; tests ${PY} -m pytest)
GPU: single 24GB RTX 4090 (cuda:0), torch 2.11.0+cu128, gsplat 1.5.3 installed. NO diffusers/peft/transformers-pipelines installed yet.
SCENE (read-only, NEVER overwrite originals): PLY ${SCENE}/gaussian_pointcloud_30000_original.ply (1.5M gaussians, SH deg 3);
  COLMAP model ${SCENE}/sparse/0 (cameras.bin/images.bin); training images ${SCENE}/images (perspective views; COLMAP names end .jpg but files are .png — extension swap already handled). Camera_id 1 = SIMPLE_PINHOLE 2048^2, f=1024 (the perspective training rig). Splat renders at the real COLMAP poses match the real photos at ~31 dB, so the frame is correct.
ALL enhancement output goes to a NEW file under ${REPO}/data/enhanced/ (gitignored). Input PLY is read-only.

ALREADY BUILT (src/gaussian_robot/), all passing mypy-strict + ruff (line-length 100, double quotes, from __future__ import annotations):
- backends/gsplat_renderer.py: rasterize_gaussians(means,quats,scales,opacities,sh_coeffs,sh_degree,camera) -> GradRender(rgb (H,W,3) float UNCLAMPED, depth (H,W), alpha (H,W) accumulated-opacity, info dict). GsplatRenderer(cloud).render(camera)->RenderResult(rgb uint8, depth, alpha). load_gaussian_cloud(path, device). GaussianCloud holds means/quats/scales(activated exp)/opacities(activated sigmoid)/sh_coeffs/sh_degree/bounds/full_bounds/density_bounds.
- enhance/distiller.py: GaussianDistiller(cloud, *, device, lrs, cap_max_factor, noise_lr, refine_*, freeze_means_iters, ssim_weight, densify). Holds gsplat-correct PRE-ACTIVATION ParameterDict (means, log-scales, quats, logit-opacities, sh) + one Adam per attribute + MCMCStrategy. .render(camera)->GradRender from live params; .fit(views: Sequence[SupervisionView], iters); .to_cloud(); .save_ply(path); .num_gaussians; .peak_vram_gb(). densify=False => pure Adam, fixed N, no MCMC.
- enhance/protocols.py: SupervisionView(camera: Camera, target_rgb: np.ndarray (H,W,3) float[0,1], mask: np.ndarray|None (H,W) in [0,1], 1=synthesized/supervise, None=full anchor). ViewFiller Protocol: fill(degraded: RenderResult, references: Sequence[RenderResult]) -> SupervisionView. SplatDistiller Protocol.
- enhance/mask.py: coverage_mask(alpha, tau_lo=0.5, feather=0.15)->M (1 where alpha low); downscale_to_latent(alpha,(h,w))->O_z (max-pool, for opacity mixing).
- enhance/capture_images.py: load_colmap_views(model_dir, images_dir, camera_id=1)->list[RealView(camera, image_path, name, camera_id)]; load_image(path,w,h)->(H,W,3) float[0,1]; scale_intrinsics(intr,factor); camera_fovs(views)->(hfov,vfov).
- enhance/orchestrator.py: enhance_scene(ply, colmap_model_dir, images_dir, out_ply, *, device, camera_id=1, downscale=0.5, iters, max_anchor, eval_stride=12, gap_grid=32, densify=False, freeze_geometry=True, ssim_weight=0, lrs) -> EnhanceReport. Currently it: loads cloud, loads real views, build_coverage3d(...).gap_centers() (found 2991 gaps on this scene), splits anchor/held-out, and runs an ANCHORED frozen-geometry colour/opacity polish on real images (held-out 30.99->31.52 dB). It does NOT yet fill gaps with novel content.
- splat/ply_writer.py: write_gaussian_ply(path, means, quats, scales, opacities, sh_coeffs) — inverse of the loader.
- metrics/coverage3d.py: build_coverage3d(means, opacities, cam_pos, cam_rot, hfov, vfov, lo, hi, grid=32) -> Coverage3D; .gap_centers() -> (K,3) world XYZ of occupied-but-unseen surface voxels.
- session.look_at(origin, target, up_axis) -> (3,3) world->camera rotation (OpenCV axes). render/camera.py: Camera(pose: Pose(position,rotation world->camera), intrinsics: CameraIntrinsics(fx,fy,cx,cy,width,height)).

THE GOAL (what makes this not a toy): a real generative ViewFiller that, at the coverage-gap poses, turns the splat's DEGRADED render (low-alpha holes) into a CLEAN target image using a generative prior, then distills those targets back into the gaussians so under-observed regions actually gain plausible content — the ArtiFixer3D idea at 24GB. Opacity-mixing: only the low-alpha (masked M) pixels are synthesized; high-alpha pixels are taken verbatim from the real render (hard recomposite enhanced = M*generated + (1-M)*render).

CRITICAL LESSON (do not repeat): a naive global re-fit or unconstrained global MCMC densification with thin supervision SCATTERS floaters / "colors everywhere" and regresses held-out views. Enhancement must (a) supervise gap views with a real filler AND keep ~50% real-anchor views to prevent drift, (b) confine geometry change to gap regions (seed/grow gaussians near gap_centers and/or mask the loss to M), (c) ALWAYS verify on held-out REAL views that the rest of the scene does not regress.

LICENSE: research/internal R&D is fine (Difix/SD-Turbo non-commercial OK for now).
`

const PROBE_SCHEMA = {
  type: 'object', additionalProperties: true,
  required: ['filler_mode', 'single_forward_ok', 'notes'],
  properties: {
    filler_mode: { type: 'string', enum: ['difix', 'sdturbo', 'sd2inpaint', 'geometric-only'], description: 'the diffusion option that actually loads+runs here, or geometric-only if none' },
    working_model_id: { type: 'string', description: 'HF model id that loaded, or "" ' },
    install_commands_that_worked: { type: 'array', items: { type: 'string' } },
    env_conflicts: { type: 'string', description: 'any package conflicts; whether torch/gsplat/numpy/vllm were left intact' },
    single_forward_ok: { type: 'boolean', description: 'did a single img2img/restore forward run on cuda at ~512px' },
    vram_gb: { type: 'number' },
    minimal_snippet: { type: 'string', description: 'minimal working python to load model + run one fix on an (H,W,3) uint8 render -> cleaned (H,W,3)' },
    notes: { type: 'string' },
  },
}

// ---------- PHASE 1: PROBE (parallel — independent) ----------
phase('Probe')
const probes = await parallel([
  () => agent(
    `${CONTEXT}\n\nTASK (ENVIRONMENT + WEIGHTS PROBE — do real work with Bash): Determine which single-step image generative prior we can ACTUALLY load and run on this box for the ViewFiller, in priority order: (1) Difix nvidia/difix, (2) stabilityai/sd-turbo img2img, (3) stabilityai/stable-diffusion-2-inpainting. `+
    `Steps: check HF reachability; try installing into the PROJECT venv (${PY} -m pip install ...) the minimal deps (diffusers, peft, safetensors, accelerate; pin huggingface-hub if it conflicts) WITHOUT downgrading torch/gsplat/numpy/vllm — if an install would break those, STOP and record it. Then actually LOAD the highest-priority model that works and run ONE forward on a dummy 512x512 render on cuda:0, measuring peak VRAM. `+
    `If NONE load/run (no net, OOM, conflicts), set filler_mode="geometric-only". Return the structured result incl. a MINIMAL working snippet (load + fix one (H,W,3) uint8 image -> cleaned (H,W,3) uint8) for whatever worked. Be fast and decisive; cap any single download attempt and report failures rather than hanging.`,
    { label: 'env-probe', phase: 'Probe', schema: PROBE_SCHEMA }
  ),
  () => agent(
    `${CONTEXT}\n\nTASK (DIFIX/SD-TURBO INTEGRATION RECON — web + read): Find the exact, minimal recipe to use a single-step SD-Turbo-class image restorer as a 3DGS artifact "fixer": how Difix3D (nv-tlabs/Difix3D, nvidia/difix) takes a degraded render (+ optional reference view) and outputs a clean image in ONE step; the SDEdit opacity-mixing realization (encode render -> latent z_deg; mix z = O_z*z_deg + (1-O_z)*noise using the downscaled alpha; partial-denoise from the matching timestep; decode; hard recomposite enhanced = M*gen + (1-M)*render). Give concrete diffusers API calls (pipeline class, scheduler, how to start denoising from an intermediate timestep / strength). Keep it to what is needed to implement enhance/fillers/diffusion.py. Return a concise but concrete spec (prose + code sketch).`,
    { label: 'difix-recon', phase: 'Probe' }
  ),
])
const probe = probes[0] || { filler_mode: 'geometric-only', single_forward_ok: false, notes: 'probe failed', minimal_snippet: '' }
const difixSpec = probes[1] || 'see CONTEXT opacity-mixing description'
log(`Probe done: filler_mode=${probe.filler_mode}, single_forward_ok=${probe.single_forward_ok}`)

const probeJson = JSON.stringify(probe, null, 1)

// ---------- PHASE 2: BUILD FILLER ----------
phase('Build filler')
const fillerReport = await agent(
  `${CONTEXT}\n\n=== ENV/WEIGHTS PROBE RESULT ===\n${probeJson}\n\n=== DIFFUSION FIX SPEC ===\n${difixSpec}\n\n`+
  `TASK (WRITE CODE, then make it pass ruff+mypy): Implement the ViewFiller(s) under src/gaussian_robot/enhance/fillers/.\n`+
  `1) enhance/fillers/__init__.py — exports.\n`+
  `2) enhance/fillers/geometric.py — a GeometricFiller(ViewFiller): given a degraded RenderResult at a gap pose, it returns SupervisionView whose target_rgb = the render itself with low-alpha holes filled by a cheap classical inpaint (e.g. OpenCV inpaint via cv2 if available, else a blurred-neighbourhood fill), and mask M = coverage_mask(alpha). This is the no-weights fallback and must ALWAYS work.\n`+
  `3) enhance/fillers/diffusion.py — a DiffusionFiller(ViewFiller) using filler_mode="${probe.filler_mode}" / model "${probe.working_model_id || ''}" (lazy/staged load like DA3DepthEstimator; .to(cuda) on first use, expose a free()/unload). It must: VAE-encode the degraded render, opacity-mix with O_z=downscale_to_latent(alpha,...), SDEdit partial-denoise, decode, and HARD-RECOMPOSITE enhanced = M*generated + (1-M)*render so trusted pixels are untouched. mask M = coverage_mask(alpha). If filler_mode=="geometric-only", make diffusion.py import-safe but raise a clear error on use (so geometric is the path).\n`+
  `Constraints: respect the ViewFiller Protocol (fill(degraded: RenderResult, references) -> SupervisionView); from __future__ import annotations; line-length 100; double quotes; full type annotations (mypy strict). Do NOT edit other modules. After writing, run: ${REPO}/.venv/bin/ruff check src/gaussian_robot/enhance/fillers/ && ${REPO}/.venv/bin/ruff format src/gaussian_robot/enhance/fillers/ && ${REPO}/.venv/bin/mypy . — iterate until BOTH pass. Report the files created, the API, and the exact ruff+mypy PASS output.`,
  { label: 'build-filler', phase: 'Build filler' }
)
log('Filler built')

// ---------- PHASE 3: WIRE GAP-LOCALIZED DISTILLATION ----------
phase('Wire distill')
const wireReport = await agent(
  `${CONTEXT}\n\n=== PROBE ===\n${probeJson}\n\n=== FILLER BUILD REPORT ===\n${fillerReport}\n\n`+
  `TASK (WRITE CODE, then ruff+mypy+pytest): Add a gap-FILL path to the enhancement so under-observed regions gain real content, WITHOUT breaking the rest.\n`+
  `Add to enhance/orchestrator.py a function fill_gaps_scene(...) (or extend enhance_scene with mode="fill") that:\n`+
  `1) loads cloud + real views; build_coverage3d(...).gap_centers().\n`+
  `2) SYNTHESIZES gap camera poses: for a small batch of gap centers (e.g. 8-16), place a camera in navigable space looking at the gap via session.look_at (use the perspective intrinsics, downscaled). \n`+
  `3) For each gap pose: render degraded (GsplatRenderer or rasterize_gaussians), build the SupervisionView via the chosen ViewFiller (DiffusionFiller if probe says so, else GeometricFiller).\n`+
  `4) DISTILL: GaussianDistiller with densify=True but a MODEST cap (cap_max_factor ~1.05) so a few gaussians can be added near gaps; supervise a 50/50 mix of (a) the filled gap views (mask=M, only synthesized pixels) and (b) real ANCHOR views (mask=None) to prevent global drift; gentle LRs. Keep iters modest (e.g. 400-800).\n`+
  `5) write the result to a NEW ply under data/enhanced/.\n`+
  `Also ensure config.py 'enhance' mode still validates. Do NOT overwrite the input PLY (guard out!=in). After writing, run ${REPO}/.venv/bin/ruff check . && ${REPO}/.venv/bin/ruff format <changed files> && ${REPO}/.venv/bin/mypy . && ${PY} -m pytest -q — iterate until all pass. Report the new API + the PASS output. Do NOT run the heavy scene job (that is the next phase).`,
  { label: 'wire-distill', phase: 'Wire distill' }
)
log('Gap-fill distillation wired')

// ---------- PHASE 4: RUN & VERIFY ----------
phase('Run & verify')
const runReport = await agent(
  `${CONTEXT}\n\n=== PROBE ===\n${probeJson}\n\n=== WIRE REPORT ===\n${wireReport}\n\n`+
  `TASK (ACTUALLY RUN IT on the office scene and report honest numbers): Use the fill_gaps_scene path just built. Run on a SMALL scope to keep within time/VRAM: downscale 0.5, ~8-12 gap poses, ~400-600 iters, on cuda:0. Reads ${SCENE}/gaussian_pointcloud_30000_original.ply read-only; write the enhanced cloud to ${REPO}/data/enhanced/gaussian_pointcloud_30000_original_gapfill.ply.\n`+
  `Then VERIFY honestly: (a) render the ORIGINAL and the GAP-FILLED clouds at the synthesized gap poses and report whether holes got filled (alpha increased in the masked regions; show mean alpha before/after in M); (b) render both at the held-out real eval views (every 12th cam-1 view) and report mean PSNR-to-real before vs after — this MUST NOT regress more than ~0.5 dB (else report FAIL: global breakage). (c) report peak VRAM and final gaussian count. Save a few before/after PNGs (original | gap-filled | real-if-any) under ${REPO}/data/enhanced/gapfill_compare/. \n`+
  `Be brutally honest: if the diffusion filler degraded the scene or produced floaters, say so with numbers; if geometric-only ran, say that. Report exact commands run and their output. Do not claim success unless held-out held AND gap alpha increased.`,
  { label: 'run-verify', phase: 'Run & verify' }
)

return { probe, filler: fillerReport, wire: wireReport, run: runReport }
