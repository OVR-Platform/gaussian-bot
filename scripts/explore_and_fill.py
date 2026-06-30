"""Unified pipeline: the robot walks the scene, marks poses to improve, then Difix fills them.

This wires the two halves of the project together. A head-less densify session
(``session.build_session`` → ``Explorer.run_session``) explores the splat and auto-marks the
under-observed viewpoints it reaches; those marks become the ``select_centers`` of the
reference-conditioned Difix3D+ gap-fill (``orchestrator.fill_gaps_scene``), so the fill targets
exactly the gaps the navigator flagged. The input PLY is read-only; a NEW ply is written.

Finally it renders a before/after fly-through of the SAME walked path through the original vs
enhanced cloud, side by side, to a GIF.

    uv run python scripts/explore_and_fill.py \
        --ply    /mnt/archive/datasets/ufficio360-.../gaussian_pointcloud_30000_original.ply \
        --model  /mnt/archive/datasets/ufficio360-.../sparse/0/ \
        --images /mnt/archive/datasets/ufficio360-.../images \
        --filler difix --filler-dtype float16 \
        --rounds 5 --iters 800 --n-gap-poses 64 --max-anchor 96 \
        --num-seeds 6 --max-steps 40

``--real-vlm`` drives the walk with the Qwen/vLLM policy (needs the server up); the default is a
head-less coverage/scripted policy. ``--no-movie`` skips the before/after GIF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gaussian_robot.enhance.before_after import render_before_after_gif
from gaussian_robot.enhance.explore_fill import explore_and_fill


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Robot-driven Difix3D+ gap-fill + before/after movie")
    ap.add_argument("--ply", required=True, help="Input 3DGS PLY (READ-ONLY)")
    ap.add_argument("--model", required=True, help="COLMAP sparse model dir (cameras.bin/images.bin)")
    ap.add_argument("--images", required=True, help="Dir with the training images")
    ap.add_argument("--out", default=None, help="Output PLY (NEW); default data/enhanced/<stem>_filled.ply")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--camera-id", type=int, default=1)
    ap.add_argument("--downscale", type=float, default=0.5)
    # robot session
    ap.add_argument("--num-seeds", type=int, default=6, help="Walks launched across the scene")
    ap.add_argument("--max-steps", type=int, default=40, help="Steps per walk")
    ap.add_argument("--real-vlm", action="store_true", help="Drive the walk with Qwen/vLLM (needs server)")
    ap.add_argument("--no-aerial", action="store_true", help="Disable the aerial gap-survey walk")
    ap.add_argument("--up-axis", default="auto")
    # fill
    ap.add_argument("--filler", default="difix", choices=["difix", "geometric", "identity"])
    ap.add_argument("--filler-dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n-gap-poses", type=int, default=12)
    ap.add_argument("--max-anchor", type=int, default=64)
    ap.add_argument("--eval-stride", type=int, default=12)
    ap.add_argument("--gap-grid", type=int, default=32)
    ap.add_argument("--regression-tol-db", type=float, default=0.3)
    ap.add_argument("--restrict-to-gaps", action="store_true")
    ap.add_argument("--ssim-weight", type=float, default=0.0)
    ap.add_argument(
        "--aggressive",
        action="store_true",
        help="Non-conservative one-shot: unfreeze geometry, raise LRs, gate OFF, up-weight fills. "
        "Makes the change visible but can corrupt geometry (use --progressive instead).",
    )
    ap.add_argument(
        "--progressive",
        action="store_true",
        help="Faithful Difix3D+: many small consistent steps, accumulated pseudo-views, "
        "densification ON. The recommended way to get a real, stable enhancement.",
    )
    ap.add_argument("--steps", type=int, default=12, help="Progressive: number of growth steps")
    ap.add_argument("--iters-per-step", type=int, default=150, help="Progressive: distill iters/step")
    # movie
    ap.add_argument("--no-movie", action="store_true", help="Skip the before/after GIF")
    ap.add_argument("--movie-out", default=None, help="Before/after GIF path; default <out>.before_after.gif")
    ap.add_argument("--movie-max-frames", type=int, default=240)
    return ap


def main() -> int:
    args = _build_argparser().parse_args()

    if not torch.cuda.is_available():
        print("ERROR: gsplat rasterization needs a CUDA GPU.")
        return 2

    in_path = Path(args.ply)
    out_path = Path(args.out) if args.out else Path("data/enhanced") / f"{in_path.stem}_filled.ply"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Explore→fill {in_path.name} (read-only) -> {out_path}")
    print(f"  robot: seeds={args.num_seeds} max_steps={args.max_steps} real_vlm={args.real_vlm}")
    print(f"  fill:  filler={args.filler} dtype={args.filler_dtype} rounds={args.rounds} iters={args.iters}")

    report = explore_and_fill(
        args.ply,
        args.model,
        args.images,
        out_path,
        device=args.device,
        camera_id=args.camera_id,
        downscale=args.downscale,
        num_seeds=args.num_seeds,
        max_steps=args.max_steps,
        use_real_vlm=args.real_vlm,
        up_axis=args.up_axis,
        aerial_survey=not args.no_aerial,
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
        aggressive=args.aggressive,
        progressive=args.progressive,
        steps=args.steps,
        iters_per_step=args.iters_per_step,
    )

    fill = report.fill
    assert fill is not None
    print(f"\n  robot walks: {report.n_walks}/{report.n_seeds} seeds   marks (poses to improve): {report.n_marks}")
    print(f"  trajectory poses: {len(report.trajectory)}   up_axis: {report.up_axis}")
    print(f"  coverage gaps: {fill.gap_count}   gap poses filled/round: {fill.n_gap_poses}")
    print(f"  gaussians: {fill.n_gaussians_before} -> {fill.n_gaussians_after}")
    print(f"  rounds run: {fill.rounds_run}   per-round PSNR: {[round(x, 3) for x in fill.per_round_psnr]}")
    print(f"  fill diag: mask_frac={fill.fill_mask_frac:.3f}  delta={fill.fill_delta:.4f}"
          f"{'  ⚠ near-no-op fill (empty mask / weak reference)' if fill.fill_delta < 1e-3 else ''}")
    print(f"  held-out real-view PSNR: {fill.psnr_before:.3f} dB  ->  {fill.psnr_after:.3f} dB")
    dev = args.device if args.device.startswith("cuda") else None
    print(f"  peak VRAM: {torch.cuda.max_memory_allocated(dev) / 1e9:.2f} GB")
    print(f"  wrote: {fill.out_ply}")

    if not args.no_movie:
        movie_out = Path(args.movie_out) if args.movie_out else out_path.with_suffix(".before_after.gif")
        print(f"\n  rendering before/after path movie -> {movie_out}")
        torch.cuda.reset_peak_memory_stats()
        info = render_before_after_gif(
            report.trajectory,
            args.ply,
            fill.out_ply,
            movie_out,
            device=args.device,
            up_axis=report.up_axis,
            marks=report.marks,
            max_frames=args.movie_max_frames,
        )
        if info.get("ok"):
            print(f"  movie: {info['n_frames']} frames, {info['n_marks']} marks  -> {info['out_gif']}")
        else:
            print(f"  movie skipped: {info.get('error')}")

    regressed = fill.psnr_after < fill.psnr_before - 0.5
    print(f"\n  {'REGRESSED ❌ (scene got worse)' if regressed else 'scene held / improved ✅'}")
    return 1 if regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
