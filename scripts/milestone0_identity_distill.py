"""Milestone-0 acceptance gate for splat enhancement (docs/research/splat-enhancement-study.md).

Proves the load-bearing new code on a REAL scene, with no diffusion and no warp:

  1. PLY round-trip: write the loaded cloud, reload it, assert tensor-equality and that
     re-rendering matches the original at > 45 dB PSNR (validates the pre-activation
     log/logit inversion + SH repack — study trap #1).
  2. Identity distill: render targets from the splat itself, run a short masked distill loop,
     and assert the re-render does NOT regress and peak VRAM stays under the 14 GB budget
     (validates grad render + `_meta`-into-MCMC + per-attribute Adam + staging).

Usage:
    uv run python scripts/milestone0_identity_distill.py --ply /path/to/point_cloud.ply
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from gaussian_robot.backends.gsplat_renderer import (
    GaussianCloud,
    GsplatRenderer,
    load_gaussian_cloud,
)
from gaussian_robot.enhance.distiller import GaussianDistiller
from gaussian_robot.enhance.mask import coverage_mask
from gaussian_robot.enhance.protocols import SupervisionView
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose
from gaussian_robot.splat.ply_writer import write_gaussian_ply

ROUND_TRIP_PSNR = 45.0
NO_REGRESS_PSNR = 35.0
VRAM_CEILING_GB = 14.0


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> Pose:
    """World->camera Pose (OpenCV axes) with the camera at ``eye`` looking at ``target``."""
    z = target - eye
    z = z / (np.linalg.norm(z) + 1e-12)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:  # looking along up: pick another reference
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    rot = np.stack([x, y, z], axis=0)
    return Pose(position=eye.astype(np.float64), rotation=rot.astype(np.float64))


def _orbit_cameras(
    cloud: GaussianCloud, n: int = 4, res: int = 512, fov_deg: float = 60.0
) -> list[Camera]:
    lo = cloud.bounds.min
    hi = cloud.bounds.max
    center = (lo + hi) / 2.0
    radius = 1.2 * float(np.max(hi - lo)) + 1e-3
    f = res / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    intr = CameraIntrinsics(fx=f, fy=f, cx=res / 2.0, cy=res / 2.0, width=res, height=res)
    up = np.array([0.0, 1.0, 0.0])
    cams = []
    for i in range(n):
        ang = 2.0 * math.pi * i / n
        eye = center + radius * np.array([math.cos(ang), 0.15, math.sin(ang)])
        cams.append(Camera(pose=_look_at(eye, center, up), intrinsics=intr))
    return cams


def _psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0) ** 2))
    return 99.0 if mse <= 1e-12 else -10.0 * math.log10(mse)


def _check_round_trip(cloud: GaussianCloud, cams: list[Camera], device: str) -> bool:
    print("\n[1/2] PLY writer round-trip")
    with tempfile.TemporaryDirectory() as td:
        path = write_gaussian_ply(
            Path(td) / "rt.ply",
            cloud.means,
            cloud.quats,
            cloud.scales,
            cloud.opacities,
            cloud.sh_coeffs,
        )
        reloaded = load_gaussian_cloud(path, device=device)

    diffs = {
        "means": float((cloud.means - reloaded.means).abs().max()),
        "scales": float((cloud.scales - reloaded.scales).abs().max()),
        "opacities": float((cloud.opacities - reloaded.opacities).abs().max()),
        "sh": float((cloud.sh_coeffs - reloaded.sh_coeffs).abs().max()),
    }
    print(f"      max abs tensor diff: {diffs}")

    orig_r = GsplatRenderer(cloud)
    new_r = GsplatRenderer(reloaded)
    psnrs = [_psnr_uint8(orig_r.render(c).rgb, new_r.render(c).rgb) for c in cams]
    worst = min(psnrs)
    print(f"      re-render PSNR (min over {len(cams)} views): {worst:.2f} dB")
    ok = worst > ROUND_TRIP_PSNR and max(diffs.values()) < 1e-3
    print(f"      -> {'PASS' if ok else 'FAIL'} (need > {ROUND_TRIP_PSNR} dB, diff < 1e-3)")
    return ok


def _check_identity_distill(
    cloud: GaussianCloud, cams: list[Camera], device: str, iters: int
) -> bool:
    print("\n[2/2] Identity distill (no diffusion, no warp)")
    renderer = GsplatRenderer(cloud)
    views = []
    for cam in cams:
        res = renderer.render(cam)
        alpha = torch.as_tensor(res.alpha, device=device)
        mask = coverage_mask(alpha).cpu().numpy()
        views.append(
            SupervisionView(camera=cam, target_rgb=res.rgb.astype(np.float32) / 255.0, mask=mask)
        )

    dist = GaussianDistiller(cloud, device=device, freeze_means_iters=min(100, iters // 2))
    dist.reset_peak_vram()
    dist.fit(views, iters=iters)

    psnrs = []
    for view in views:
        out = dist.render(view.camera).rgb.detach().clamp(0.0, 1.0).cpu().numpy()
        psnrs.append(
            _psnr_uint8((out * 255).astype(np.uint8), (view.target_rgb * 255).astype(np.uint8))
        )
    worst = min(psnrs)
    vram = dist.peak_vram_gb()
    print(f"      anchor re-render PSNR (min): {worst:.2f} dB")
    print(f"      gaussians: {cloud.means.shape[0]} -> {dist.num_gaussians}")
    print(f"      peak VRAM: {vram:.2f} GB")
    ok = worst > NO_REGRESS_PSNR and vram < VRAM_CEILING_GB
    print(
        f"      -> {'PASS' if ok else 'FAIL'} (need > {NO_REGRESS_PSNR} dB, < {VRAM_CEILING_GB} GB)"
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Milestone-0 identity distill gate")
    parser.add_argument("--ply", required=True, help="Path to a 3DGS point_cloud.ply")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--views", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: gsplat rasterization needs a CUDA GPU.")
        return 2

    print(f"Loading {args.ply} on {args.device} ...")
    cloud = load_gaussian_cloud(args.ply, device=args.device)
    print(f"  {cloud.means.shape[0]} gaussians, SH degree {cloud.sh_degree}")
    cams = _orbit_cameras(cloud, n=args.views)

    ok1 = _check_round_trip(cloud, cams, args.device)
    ok2 = _check_identity_distill(cloud, cams, args.device, args.iters)

    print(f"\nMILESTONE-0: {'PASS ✅' if ok1 and ok2 else 'FAIL ❌'}")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
