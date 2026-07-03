"""Run the Difix3D+ gap-fill on the office scene and verify HONESTLY.

Pipeline under test = reference-conditioned official DifixPipeline (nvidia/difix_ref) on
near-training-view poses, distilled with geometry FROZEN, gated so it cannot regress.

Steps:
  1. fill_gaps_scene -> writes the gap-filled ply (or RAISES if it would regress held-out PSNR).
  2. Re-derive the SAME deterministic near-view gap poses + references, render ORIGINAL and FILLED
     clouds at them, report mean alpha in the masked hole region before/after (did gaps gain
     coverage?), and save orig/filled/reference/Difix-target PNGs.
  3. Render ORIGINAL and FILLED at held-out real eval views; report mean PSNR-to-real before vs
     after (the regression guard) + per-view.

Everything reads the original ply READ-ONLY; the only writes are the new ply + PNGs.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from gaussian_robot.backends.gsplat_renderer import GsplatRenderer, load_gaussian_cloud
from gaussian_robot.enhance.capture_images import (
    camera_fovs,
    load_colmap_views,
    load_image,
    scale_intrinsics,
)
from gaussian_robot.enhance.mask import coverage_mask
from gaussian_robot.enhance.orchestrator import (
    _infer_up_axis,
    fill_gaps_scene,
    synthesize_near_view_poses,
)
from gaussian_robot.metrics.coverage3d import build_coverage3d
from gaussian_robot.render.camera import Camera

PLY = "/mnt/archive/datasets/ufficio360-35a39133-e1f2-4426-86d4-a3d7a00614ee-PIC/gaussian_pointcloud_30000_original.ply"
COLMAP = "/mnt/archive/datasets/ufficio360-35a39133-e1f2-4426-86d4-a3d7a00614ee-PIC/sparse/0"
IMAGES = "/mnt/archive/datasets/ufficio360-35a39133-e1f2-4426-86d4-a3d7a00614ee-PIC/images"
OUT = "/mnt/wd/gaussian-bot/data/enhanced/gaussian_pointcloud_30000_original_gapfill.ply"
CMP = "/mnt/wd/gaussian-bot/data/enhanced/gapfill_compare"
DEVICE = "cuda:0"

CAMERA_ID = 1
DOWNSCALE = 0.5
FILLER = "difix"
ITERS = 300          # per round
ROUNDS = int(os.environ.get("ROUNDS", "3"))
PERTURB_START = float(os.environ.get("PERTURB_START", "0.15"))  # dolly 15% toward the gap
PERTURB_STEP = float(os.environ.get("PERTURB_STEP", "0.2"))  # grow the dolly each round
N_GAP_POSES = 10
MAX_ANCHOR = 48
EVAL_STRIDE = 12
GAP_GRID = 32
REGRESSION_TOL_DB = float(os.environ.get("REGRESSION_TOL_DB", "0.3"))
# Gentle polish LRs (match the M0 no-regress config); opacity/sh only, geometry frozen.
LRS = {"means": 0.0, "scales": 0.0, "quats": 0.0, "opacities": 5e-3, "sh": 2.5e-4}

os.makedirs(CMP, exist_ok=True)


def _save_png(path: str, img01: np.ndarray) -> None:
    from PIL import Image

    u8 = (np.clip(img01, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(u8).save(path)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else float(-10.0 * np.log10(mse))


def main() -> None:
    torch.zeros(1, device=DEVICE)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # ----- 1. RUN (raises if it would regress held-out PSNR) -----
    report = fill_gaps_scene(
        PLY, COLMAP, IMAGES, OUT,
        device=DEVICE, camera_id=CAMERA_ID, downscale=DOWNSCALE, filler=FILLER,
        iters=ITERS, rounds=ROUNDS, perturb_start=PERTURB_START, perturb_step=PERTURB_STEP,
        n_gap_poses=N_GAP_POSES, max_anchor=MAX_ANCHOR, eval_stride=EVAL_STRIDE, gap_grid=GAP_GRID,
        regression_tol_db=REGRESSION_TOL_DB, lrs=LRS,
    )
    run_secs = time.time() - t0
    peak_after_run = float(torch.cuda.max_memory_allocated()) / 1e9
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # ----- 2. RE-DERIVE the same near-view gap poses + references (deterministic) -----
    orig = load_gaussian_cloud(PLY, device=DEVICE)
    views = load_colmap_views(COLMAP, IMAGES, camera_id=CAMERA_ID)

    means_np = orig.means.detach().cpu().numpy()
    opac_np = orig.opacities.detach().cpu().numpy().reshape(-1)
    cam_pos = np.array([v.camera.pose.position for v in views], dtype=np.float64)
    cam_rot = np.array([v.camera.pose.rotation for v in views], dtype=np.float64)
    hfov, vfov = camera_fovs(views)
    lo, hi = (
        orig.density_bounds
        if orig.density_bounds is not None
        else (orig.full_bounds.min, orig.full_bounds.max)
    )
    cov = build_coverage3d(means_np, opac_np, cam_pos, cam_rot, hfov, vfov, lo, hi, grid=GAP_GRID)
    gap_centers = cov.gap_centers()
    resolved_up = _infer_up_axis(cam_pos)
    gap_intrinsics = scale_intrinsics(views[0].camera.intrinsics, DOWNSCALE)
    gap_pairs = synthesize_near_view_poses(
        views, gap_centers, gap_intrinsics, up_axis=resolved_up, n_poses=N_GAP_POSES
    )
    rw, rh = gap_intrinsics.width, gap_intrinsics.height

    # Render ORIGINAL at the (round-0) gap poses; mask M from the original degraded alpha. (The raw
    # Difix outputs are dumped separately by show_novel_views.py — kept out of here to save VRAM.)
    r_orig = GsplatRenderer(orig)
    gap_orig_alpha, gap_masks, gap_orig_rgb, gap_ref_rgb = [], [], [], []
    for cam, ref_view in gap_pairs:
        rr = r_orig.render(cam)
        a = np.asarray(rr.alpha, dtype=np.float32)
        m = coverage_mask(torch.as_tensor(a), tau_lo=0.5, feather=0.15).cpu().numpy().astype(np.float32)
        gap_orig_alpha.append(a)
        gap_masks.append(m)
        gap_orig_rgb.append(rr.rgb.astype(np.float32) / 255.0)
        gap_ref_rgb.append(load_image(ref_view.image_path, rw, rh))
    del r_orig

    filled = load_gaussian_cloud(OUT, device=DEVICE)
    r_fill = GsplatRenderer(filled)
    gap_fill_alpha, gap_fill_rgb = [], []
    for cam, _ in gap_pairs:
        rr = r_fill.render(cam)
        gap_fill_alpha.append(np.asarray(rr.alpha, dtype=np.float32))
        gap_fill_rgb.append(rr.rgb.astype(np.float32) / 255.0)

    per_pose, hole_total, wb, wa = [], 0, 0.0, 0.0
    for i, (a0, a1, m) in enumerate(zip(gap_orig_alpha, gap_fill_alpha, gap_masks, strict=True)):
        hole = m >= 0.5
        n = int(hole.sum())
        mb = float(a0[hole].mean()) if n else float("nan")
        ma = float(a1[hole].mean()) if n else float("nan")
        per_pose.append({"pose": i, "hole_px": n, "alpha_in_M_before": mb, "alpha_in_M_after": ma})
        if n:
            hole_total += n
            wb += mb * n
            wa += ma * n
    del r_fill

    # ----- 3. Held-out real eval views: PSNR-to-real before vs after -----
    eval_views = views[::EVAL_STRIDE]

    def _cam_and_img(v):  # type: ignore[no-untyped-def]
        cam = Camera(pose=v.camera.pose, intrinsics=scale_intrinsics(v.camera.intrinsics, DOWNSCALE))
        return cam, load_image(v.image_path, cam.intrinsics.width, cam.intrinsics.height)

    eval_cams_imgs = [_cam_and_img(v) for v in eval_views]
    r_orig2, r_fill2 = GsplatRenderer(orig), GsplatRenderer(filled)
    psnr_before, psnr_after, eval_imgs = [], [], []
    for ci, (cam, tgt) in enumerate(eval_cams_imgs):
        pb = r_orig2.render(cam).rgb.astype(np.float32) / 255.0
        pa = r_fill2.render(cam).rgb.astype(np.float32) / 255.0
        psnr_before.append(_psnr(pb, tgt))
        psnr_after.append(_psnr(pa, tgt))
        eval_imgs.append((ci, pb, pa, tgt))

    result = {
        "scope": {
            "filler": FILLER, "downscale": DOWNSCALE, "iters": ITERS, "n_gap_poses": N_GAP_POSES,
            "max_anchor": MAX_ANCHOR, "eval_stride": EVAL_STRIDE, "gap_grid": GAP_GRID,
            "regression_tol_db": REGRESSION_TOL_DB, "geometry": "frozen", "densify": False,
        },
        "report": {
            "n_views": report.n_views, "n_anchor": report.n_anchor, "n_eval": report.n_eval,
            "gap_count": report.gap_count, "n_gap_poses": report.n_gap_poses,
            "n_gaussians_before": report.n_gaussians_before,
            "n_gaussians_after": report.n_gaussians_after, "filler": report.filler,
            "report_psnr_before": report.psnr_before, "report_psnr_after": report.psnr_after,
            "rounds_run": report.rounds_run, "per_round_psnr": report.per_round_psnr,
            "out_ply": report.out_ply,
        },
        "gap_alpha": {
            "agg_alpha_in_M_before": wb / hole_total if hole_total else float("nan"),
            "agg_alpha_in_M_after": wa / hole_total if hole_total else float("nan"),
            "hole_px_total": hole_total, "per_pose": per_pose,
        },
        "heldout_real": {
            "resolved_up_axis": resolved_up, "n_eval": len(eval_cams_imgs),
            "psnr_before_mean": float(np.mean(psnr_before)),
            "psnr_after_mean": float(np.mean(psnr_after)),
            "psnr_delta_mean": float(np.mean(psnr_after) - np.mean(psnr_before)),
            "psnr_before_per": [float(x) for x in psnr_before],
            "psnr_after_per": [float(x) for x in psnr_after],
        },
        "perf": {"peak_vram_gb": peak_after_run, "run_secs": run_secs},
    }
    print("RESULT_JSON_BEGIN")
    print(json.dumps(result, indent=2))
    print("RESULT_JSON_END")

    for i in range(min(4, len(gap_pairs))):
        comp = np.concatenate([gap_orig_rgb[i], gap_fill_rgb[i], gap_ref_rgb[i]], axis=1)
        _save_png(f"{CMP}/gap{i:02d}_orig_filled_ref.png", comp)
        _save_png(f"{CMP}/gap{i:02d}_mask.png", np.repeat(gap_masks[i][..., None], 3, axis=2))
    for ci, pb, pa, tgt in eval_imgs[:3]:
        _save_png(f"{CMP}/eval{ci:02d}_orig_filled_real.png", np.concatenate([pb, pa, tgt], axis=1))


if __name__ == "__main__":
    main()
