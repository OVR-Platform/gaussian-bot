"""SUPERSEDED by ``uv run gaussian-robot enhance`` (ADR-0011) — kept for research reference.

The CLI drives the same fill through the robot-driven path; use ``--rounds-mode`` for this
script's rounds regime. Run the generative Difix3D+ gap-FILL loop on a scene (docs/research/).

Unlike ``enhance_scene.py`` (the M0 anchored, densify-controlled refinement), this drives the
reference-conditioned diffusion filler: each round renders near-view gap poses DEGRADED from the
current cloud, cleans them with ``nvidia/difix_ref`` (single-step, reference-mixing), and distils
the cleaned content back into the gap gaussians — with a held-out PSNR gate so the result never
regresses. Reads the input PLY read-only and writes a NEW ply.

    uv run python scripts/fill_gaps.py \
        --ply    /mnt/archive/datasets/ufficioNikon/gaussian_pointcloud_30000_original.ply \
        --model  /mnt/archive/datasets/ufficioNikon/sparse/0 \
        --images /mnt/archive/datasets/ufficioNikon/images \
        --filler difix --filler-dtype float16

``--filler-dtype float16`` roughly halves the Difix VRAM peak (~8.4 GB -> ~4-5 GB); validate the
held-out PSNR, since some VAEs are unstable in fp16. ``--filler geometric`` runs the no-weights
fallback (no download, useful for a fast end-to-end smoke test).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gaussian_robot.enhance.orchestrator import fill_gaps_scene


def main() -> int:
    ap = argparse.ArgumentParser(description="Generative Difix3D+ gap-fill (reference-conditioned)")
    ap.add_argument("--ply", required=True, help="Input 3DGS PLY (READ-ONLY)")
    ap.add_argument(
        "--model", required=True, help="COLMAP sparse model dir (cameras.bin/images.bin)"
    )
    ap.add_argument("--images", required=True, help="Dir with the training images")
    ap.add_argument(
        "--out", default=None, help="Output PLY (NEW file); default data/enhanced/<stem>_filled.ply"
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--camera-id", type=int, default=1, help="COLMAP camera_id of the training rig")
    ap.add_argument("--downscale", type=float, default=0.5)
    ap.add_argument("--filler", default="difix", choices=["difix", "geometric", "identity"])
    ap.add_argument(
        "--filler-dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Difix precision; float16 halves VRAM (validate quality). Ignored by geometric.",
    )
    ap.add_argument("--iters", type=int, default=300, help="Distiller iters per round")
    ap.add_argument("--rounds", type=int, default=3, help="Progressive rounds (1 = single pass)")
    ap.add_argument("--n-gap-poses", type=int, default=12)
    ap.add_argument("--max-anchor", type=int, default=64)
    ap.add_argument("--eval-stride", type=int, default=12)
    ap.add_argument("--gap-grid", type=int, default=32)
    ap.add_argument(
        "--regression-tol-db",
        type=float,
        default=0.3,
        help="Revert/stop if a round drops more than this below the best held-out PSNR.",
    )
    ap.add_argument(
        "--restrict-to-gaps", action="store_true", help="Gap-local SH/opacity gradients"
    )
    ap.add_argument("--ssim-weight", type=float, default=0.0)
    ap.add_argument(
        "--ref-select",
        default="visible",
        choices=["visible", "nearest"],
        help="Reference picking: 'visible' projects the gap into each view and prefers the one "
        "that frames it; 'nearest' is the legacy translation argmin.",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: gsplat rasterization needs a CUDA GPU.")
        return 2

    in_path = Path(args.ply)
    out_path = Path(args.out) if args.out else Path("data/enhanced") / f"{in_path.stem}_filled.ply"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Gap-fill {in_path.name} (read-only) -> {out_path}")
    print(
        f"  filler={args.filler} dtype={args.filler_dtype} rounds={args.rounds} iters={args.iters}"
    )
    report = fill_gaps_scene(
        args.ply,
        args.model,
        args.images,
        out_path,
        device=args.device,
        camera_id=args.camera_id,
        downscale=args.downscale,
        filler=args.filler,
        filler_dtype=args.filler_dtype,
        iters=args.iters,
        rounds=args.rounds,
        n_gap_poses=args.n_gap_poses,
        max_anchor=args.max_anchor,
        eval_stride=args.eval_stride,
        gap_grid=args.gap_grid,
        regression_tol_db=args.regression_tol_db,
        restrict_to_gaps=args.restrict_to_gaps,
        ssim_weight=args.ssim_weight,
        ref_select=args.ref_select,
    )

    print(
        f"\n  registered views: {report.n_views}  "
        f"(anchor {report.n_anchor}, held-out eval {report.n_eval})"
    )
    print(f"  coverage gaps: {report.gap_count}   gap poses filled/round: {report.n_gap_poses}")
    print(f"  gaussians: {report.n_gaussians_before} -> {report.n_gaussians_after}")
    print(
        f"  rounds run: {report.rounds_run}   per-round PSNR: {[round(x, 3) for x in report.per_round_psnr]}"
    )
    print(
        f"  fill diag: mask_frac={report.fill_mask_frac:.3f}  delta={report.fill_delta:.4f}"
        f"{'  ⚠ near-no-op fill (empty mask / weak reference)' if report.fill_delta < 1e-3 else ''}"
    )
    print(f"  held-out real-view PSNR: {report.psnr_before:.3f} dB  ->  {report.psnr_after:.3f} dB")
    print(f"  per-view before: {[round(x, 1) for x in report.per_eval_before]}")
    print(f"  per-view after:  {[round(x, 1) for x in report.per_eval_after]}")
    if torch.cuda.is_available():
        print(f"  peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"  wrote: {report.out_ply}")
    regressed = report.psnr_after < report.psnr_before - 0.5
    print(f"\n  {'REGRESSED ❌ (scene got worse)' if regressed else 'scene held / improved ✅'}")
    return 1 if regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
