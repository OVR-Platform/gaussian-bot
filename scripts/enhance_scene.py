"""SUPERSEDED by ``uv run gaussian-robot enhance`` (ADR-0011) — kept for research reference.

The supported path is the CLI (`--rounds-mode` covers this script's regime with a filler; this
script remains the filler-less anchored polish). Run the splat-enhancement loop on a scene (docs/research/splat-enhancement-study.md).

Anchored, densification-controlled refinement driven by the backbone coverage signal.
Reads the input PLY read-only and writes a NEW ply.

    uv run python scripts/enhance_scene.py \
        --ply  /path/scene/gaussian_pointcloud_30000_original.ply \
        --model /path/scene/sparse/0 --images /path/scene/images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gaussian_robot.enhance.orchestrator import enhance_scene


def main() -> int:
    ap = argparse.ArgumentParser(description="Splat enhancement (anchored, densify-controlled)")
    ap.add_argument("--ply", required=True, help="Input 3DGS PLY (READ-ONLY)")
    ap.add_argument(
        "--model", required=True, help="COLMAP sparse model dir (cameras.bin/images.bin)"
    )
    ap.add_argument("--images", required=True, help="Dir with the training images")
    ap.add_argument(
        "--out",
        default=None,
        help="Output PLY (NEW file); default data/enhanced/<stem>_enhanced.ply",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--camera-id", type=int, default=1, help="COLMAP camera_id of the training rig")
    ap.add_argument("--downscale", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument(
        "--densify", action="store_true", help="Enable MCMC densification (default off)"
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: gsplat rasterization needs a CUDA GPU.")
        return 2

    in_path = Path(args.ply)
    out_path = (
        Path(args.out) if args.out else Path("data/enhanced") / f"{in_path.stem}_enhanced.ply"
    )

    print(f"Enhancing {in_path.name} (read-only) -> {out_path}")
    report = enhance_scene(
        args.ply,
        args.model,
        args.images,
        out_path,
        device=args.device,
        camera_id=args.camera_id,
        downscale=args.downscale,
        iters=args.iters,
        densify=args.densify,
    )
    print(
        f"\n  registered views: {report.n_views}  (anchor {report.n_anchor}, held-out eval {report.n_eval})"
    )
    print(f"  coverage gaps (poses-to-enhance): {report.gap_count}")
    print(f"  gaussians: {report.n_gaussians}  densify={args.densify}")
    print(f"  held-out real-view PSNR: {report.psnr_before:.2f} dB  ->  {report.psnr_after:.2f} dB")
    print(f"  per-view before: {[round(x, 1) for x in report.per_eval_before]}")
    print(f"  per-view after:  {[round(x, 1) for x in report.per_eval_after]}")
    print(f"  wrote: {report.out_ply}")
    regressed = report.psnr_after < report.psnr_before - 0.5
    print(f"\n  {'REGRESSED ❌ (scene got worse)' if regressed else 'scene held / improved ✅'}")
    return 1 if regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
