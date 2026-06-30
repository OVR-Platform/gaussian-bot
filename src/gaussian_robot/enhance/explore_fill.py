"""Robot-driven gap-fill: let the navigator pick the poses, then Difix-fill them.

This is the seam the user asked for — it unifies the two halves of the project:

- The **robot** (``session.build_session`` → ``Explorer.run_session``) walks the scene in
  *densify* mode and auto-marks the under-observed viewpoints it reaches (``WalkResult.marks`` —
  "poses to improve"). This is the same coverage-driven exploration the densify deliverable is
  built from, run head-less here (scripted/coverage policy by default, real VLM optional).
- The **fill** (``orchestrator.fill_gaps_scene``) takes those marks as ``select_centers`` and
  runs the progressive, reference-conditioned Difix3D+ loop *narrowed to the gaps the robot
  flagged* — instead of spreading across every gap in the scene. The held-out PSNR gate still
  guards every round, and the input PLY stays read-only (a NEW ply is written).

The walked trajectory is returned alongside the report so a caller can render the
before/after fly-through (``enhance.before_after.render_before_after_gif``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gaussian_robot.enhance.orchestrator import (
    FillReport,
    fill_gaps_progressive,
    fill_gaps_scene,
)
from gaussian_robot.render.camera import Pose


@dataclass
class ExploreFillReport:
    """Outcome of an explore→fill run: what the robot found and what the fill did."""

    n_seeds: int
    n_walks: int
    n_marks: int  # robot "poses to improve" (deduped marks across all walks)
    up_axis: str
    trajectory: list[Pose] = field(default_factory=list)  # full walked path, for the movie
    marks: list[Pose] = field(default_factory=list)  # the marked vantage points
    fill: FillReport | None = None
    out_ply: str = ""


def _free_session_renderer() -> None:
    """Drop the session's module-cached renderer + empty the CUDA cache.

    ``session._load_renderer`` keeps the original cloud resident in a module global so the UI can
    reuse it; here the fill is about to load its own cloud + Difix, so release the duplicate first.
    """
    from gaussian_robot import session as _session  # noqa: PLC0415

    _session._cached_renderer = None  # noqa: SLF001
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def explore_and_fill(
    ply_path: str | Path,
    colmap_model_dir: str | Path,
    images_dir: str | Path,
    out_ply: str | Path,
    *,
    device: str = "cuda:0",
    camera_id: int | None = 1,
    downscale: float = 0.5,
    # --- robot session ---
    num_seeds: int = 6,
    max_steps: int = 40,
    use_real_vlm: bool = False,
    up_axis: str = "auto",
    aerial_survey: bool = True,
    # --- fill ---
    filler: str = "difix",
    filler_dtype: str = "float16",
    iters: int = 300,
    rounds: int = 3,
    n_gap_poses: int = 12,
    max_anchor: int = 64,
    eval_stride: int = 12,
    gap_grid: int = 32,
    regression_tol_db: float = 0.3,
    restrict_to_gaps: bool = False,
    ssim_weight: float = 0.0,
    aggressive: bool = False,
    lrs: dict[str, float] | None = None,
    gate: bool | None = None,
    fill_weight: float | None = None,
    # --- progressive Difix3D+ (recommended) ---
    progressive: bool = False,
    steps: int = 12,
    iters_per_step: int = 150,
) -> ExploreFillReport:
    """Run the robot densify walk, then Difix-fill the gaps it marked. Writes a NEW ply.

    The robot session renders the splat read-only to navigate; the fill reads the same PLY
    read-only and writes ``out_ply``. ``out_ply`` must differ from ``ply_path``.

    ``aggressive=True`` is the non-conservative preset: it UNFREEZES geometry and raises the
    opacity/SH/scale learning rates so the splat actually adopts the Difix-sharpened targets, turns
    the no-regression ``gate`` off (ships the final round), and up-weights the fills. This makes the
    enhancement visible at the cost of the held-out guarantee. Explicit ``lrs``/``gate``/
    ``fill_weight`` override the preset.
    """
    from gaussian_robot.config import RunConfig  # noqa: PLC0415
    from gaussian_robot.session import build_session  # noqa: PLC0415

    if Path(out_ply).resolve() == Path(ply_path).resolve():
        raise ValueError("out_ply must differ from ply_path; refusing to overwrite the original")

    # 1) Robot densify session — coverage-driven exploration that marks poses to improve.
    config = RunConfig(
        mode="densify",
        ply_path=str(ply_path),
        poses_path=str(colmap_model_dir),
        cuda_device=device,
        up_axis=up_axis,
        use_real_vlm=use_real_vlm,
        use_capture_pose_seeds=True,
        coverage_3d=True,
        aerial_survey=aerial_survey,
        num_seeds=num_seeds,
        max_steps=max_steps,
    )
    explorer, seeds, coverage = build_session(config)
    results = explorer.run_session(seeds, coverage, requested_seeds=num_seeds)
    resolved_up = explorer.scene.up_axis

    trajectory = [p for r in results for p in r.poses]
    marks = [m for r in results for m in r.marks]
    select_centers = (
        np.array([m.position for m in marks], dtype=np.float64) if marks else None
    )

    # Resolve the aggressive preset (explicit args win over it).
    if aggressive:
        # Unfreeze geometry so blurry gaussians can shrink/move toward the sharp targets (the
        # frozen-geometry path can only recolour them, never sharpen). High opacity/SH LRs let the
        # fills actually take. Gate off ships the final round even if held-out PSNR dips.
        if lrs is None:
            lrs = {"means": 2e-5, "scales": 3e-3, "quats": 1e-3, "opacities": 3e-2, "sh": 1e-2}
        if gate is None:
            gate = False
        if fill_weight is None:
            fill_weight = 2.0
    resolved_gate = True if gate is None else gate
    resolved_fill_weight = 1.0 if fill_weight is None else fill_weight

    # 2) Free the duplicate cloud the session cached, then Difix-fill the marked gaps.
    _free_session_renderer()
    if progressive:
        # Faithful Difix3D+: many small consistent steps, accumulated pseudo-views, densification.
        fill = fill_gaps_progressive(
            ply_path,
            colmap_model_dir,
            images_dir,
            out_ply,
            device=device,
            camera_id=camera_id,
            downscale=downscale,
            filler=filler,
            filler_dtype=filler_dtype,
            steps=steps,
            iters_per_step=iters_per_step,
            n_gap_poses=n_gap_poses,
            max_anchor=max_anchor,
            eval_stride=eval_stride,
            gap_grid=gap_grid,
            ssim_weight=ssim_weight,
            lrs=lrs,
            # Progressive frontier: march from the real cameras toward the robot's flagged targets,
            # perturb growing each step, fixing only the currently-recoverable (well-covered) frames.
            target_centers=(
                np.array([m.position for m in marks], dtype=np.float64) if marks else None
            ),
        )
        return ExploreFillReport(
            n_seeds=len(seeds), n_walks=len(results), n_marks=len(marks),
            up_axis=resolved_up, trajectory=trajectory, marks=marks,
            fill=fill, out_ply=fill.out_ply,
        )
    fill = fill_gaps_scene(
        ply_path,
        colmap_model_dir,
        images_dir,
        out_ply,
        device=device,
        camera_id=camera_id,
        downscale=downscale,
        filler=filler,
        filler_dtype=filler_dtype,
        iters=iters,
        rounds=rounds,
        n_gap_poses=n_gap_poses,
        max_anchor=max_anchor,
        eval_stride=eval_stride,
        gap_grid=gap_grid,
        regression_tol_db=regression_tol_db,
        restrict_to_gaps=restrict_to_gaps,
        ssim_weight=ssim_weight,
        select_centers=select_centers,
        lrs=lrs,
        gate=resolved_gate,
        fill_weight=resolved_fill_weight,
    )

    return ExploreFillReport(
        n_seeds=len(seeds),
        n_walks=len(results),
        n_marks=len(marks),
        up_axis=resolved_up,
        trajectory=trajectory,
        marks=marks,
        fill=fill,
        out_ply=fill.out_ply,
    )
