"""Dump the 10 novel views: BEFORE (splat render at the near-view gap pose) and AFTER
(the RAW Difix diffusion output — not the recomposite, not the render). Deterministic re-derivation
of the same poses fill_gaps_scene uses."""

from __future__ import annotations

import os

import numpy as np
import torch

from gaussian_robot.backends.gsplat_renderer import GsplatRenderer, load_gaussian_cloud
from gaussian_robot.enhance.capture_images import (
    camera_fovs,
    load_colmap_views,
    load_image,
    scale_intrinsics,
)
from gaussian_robot.enhance.fillers import DiffusionFiller
from gaussian_robot.enhance.mask import coverage_mask
from gaussian_robot.enhance.orchestrator import _infer_up_axis, synthesize_near_view_poses
from gaussian_robot.metrics.coverage3d import build_coverage3d

PLY = "/mnt/archive/datasets/ufficio360-35a39133-e1f2-4426-86d4-a3d7a00614ee-PIC/gaussian_pointcloud_30000_original.ply"
COLMAP = "/mnt/archive/datasets/ufficio360-35a39133-e1f2-4426-86d4-a3d7a00614ee-PIC/sparse/0"
IMAGES = "/mnt/archive/datasets/ufficio360-35a39133-e1f2-4426-86d4-a3d7a00614ee-PIC/images"
OUT = "/mnt/wd/gaussian-bot/data/enhanced/novel_views"
DEVICE, DOWNSCALE, N, GRID = "cuda:0", 0.5, 10, 32
os.makedirs(OUT, exist_ok=True)


def save(path, img01):
    from PIL import Image
    Image.fromarray((np.clip(img01, 0, 1) * 255).astype(np.uint8)).save(path)


def main():
    cloud = load_gaussian_cloud(PLY, device=DEVICE)
    views = load_colmap_views(COLMAP, IMAGES, camera_id=1)
    means = cloud.means.detach().cpu().numpy()
    opac = cloud.opacities.detach().cpu().numpy().reshape(-1)
    cpos = np.array([v.camera.pose.position for v in views], np.float64)
    crot = np.array([v.camera.pose.rotation for v in views], np.float64)
    hf, vf = camera_fovs(views)
    lo, hi = (cloud.density_bounds if cloud.density_bounds is not None
              else (cloud.full_bounds.min, cloud.full_bounds.max))
    cov = build_coverage3d(means, opac, cpos, crot, hf, vf, lo, hi, grid=GRID)
    up = _infer_up_axis(cpos)
    intr = scale_intrinsics(views[0].camera.intrinsics, DOWNSCALE)
    pairs = synthesize_near_view_poses(views, cov.gap_centers(), intr, up_axis=up, n_poses=N)

    r = GsplatRenderer(cloud)
    filler = DiffusionFiller(device=DEVICE)
    filler.load()
    montage_rows = []
    print(f"{'pose':>4} {'hole_px':>8} {'mean|difix-render| in hole':>28}")
    for i, (cam, ref_view) in enumerate(pairs):
        rr = r.render(cam)
        render01 = rr.rgb.astype(np.float32) / 255.0
        ref_img = load_image(ref_view.image_path, intr.width, intr.height)
        # RAW diffusion output (no recomposite): call the pipeline path directly.
        render_u8 = np.ascontiguousarray(rr.rgb).astype(np.uint8)
        ref_u8 = (np.clip(ref_img, 0, 1) * 255).astype(np.uint8)
        with torch.no_grad():
            difix01 = filler._difix(render_u8, ref_u8)  # pure generated, (H,W,3) float[0,1]
        a = np.asarray(rr.alpha, np.float32)
        m = coverage_mask(torch.as_tensor(a), tau_lo=0.5, feather=0.15).cpu().numpy()
        hole = m >= 0.5
        n = int(hole.sum())
        d = float(np.abs(difix01[hole] - render01[hole]).mean()) if n else float("nan")
        print(f"{i:>4} {n:>8} {d:>28.4f}")
        save(f"{OUT}/novel{i:02d}_BEFORE_render.png", render01)
        save(f"{OUT}/novel{i:02d}_AFTER_difix.png", difix01)
        # side-by-side for quick viewing: BEFORE | AFTER
        montage_rows.append(np.concatenate([render01, difix01], axis=1))
    filler.free()
    # one tall contact sheet of all 10 (left=render, right=difix)
    save(f"{OUT}/ALL_before_render_VS_after_difix.png", np.concatenate(montage_rows, axis=0))
    print(f"\nwrote {len(pairs)} novel views (BEFORE/AFTER) to {OUT}")


if __name__ == "__main__":
    main()
