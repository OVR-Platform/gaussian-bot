"""End-to-end splat enhancement loop, wired to the existing backbone.

This connects the pieces rather than reinventing them:

- WHERE to enhance comes from the backbone's coverage signal — ``build_coverage3d`` over the
  splat + its real capture cameras yields ``gap_centers()`` (occupied-but-unseen surface),
  the same "poses to enhance" the densify deliverable is built from.
- Supervision is anchored on the REAL training images (``capture_images``), so the fine-tune
  cannot drift away from observed geometry.
- The distill runs with densification OFF (``GaussianDistiller(densify=False)``) — a localized,
  anchored refinement, NOT the unconstrained global MCMC that scattered floaters in the first
  Milestone-0 smoke-test.
- The result is written to a NEW ply; the input is read-only.

The generative/geometric ViewFiller (turning a gap pose's degraded render into a *clean* novel
target) is the one remaining pluggable slot — this module supervises real anchors today and
exposes the gap poses where a filler will act next.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from gaussian_robot.backends.gsplat_renderer import (
    GaussianCloud,
    GsplatRenderer,
    load_gaussian_cloud,
)
from gaussian_robot.enhance.capture_images import (
    RealView,
    camera_fovs,
    load_colmap_views,
    load_image,
    scale_intrinsics,
)
from gaussian_robot.enhance.distiller import GaussianDistiller
from gaussian_robot.enhance.protocols import SupervisionView
from gaussian_robot.metrics.coverage3d import build_coverage3d
from gaussian_robot.render.base import RenderResult
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose
from gaussian_robot.session import look_at

if TYPE_CHECKING:
    from gaussian_robot.enhance.protocols import ViewFiller


@dataclass
class EnhanceReport:
    n_views: int
    n_anchor: int
    n_eval: int
    n_gaussians: int
    gap_count: int
    psnr_before: float
    psnr_after: float
    out_ply: str
    per_eval_before: list[float] = field(default_factory=list)
    per_eval_after: list[float] = field(default_factory=list)


@dataclass
class FillReport:
    """Outcome of a gap-FILL run (novel content distilled into under-observed regions).

    The held-out PSNR pair is the regression guard: ``psnr_after`` must not fall materially
    below ``psnr_before`` (the rest of the scene must not drift while gaps gain content).
    """

    n_views: int
    n_anchor: int
    n_eval: int
    gap_count: int
    n_gap_poses: int
    n_gaussians_before: int
    n_gaussians_after: int
    filler: str
    psnr_before: float
    psnr_after: float
    out_ply: str
    rounds_run: int = 1
    per_round_psnr: list[float] = field(default_factory=list)
    per_eval_before: list[float] = field(default_factory=list)
    per_eval_after: list[float] = field(default_factory=list)
    # Last-round fill diagnostics: ``fill_mask_frac`` is the mean fraction of each gap frame the
    # filler was allowed to change; ``fill_delta`` is the mean abs change it actually introduced
    # over the raw render. ``fill_delta ≈ 0`` flags a no-op fill (empty mask / useless reference).
    fill_mask_frac: float = 0.0
    fill_delta: float = 0.0


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else float(-10.0 * np.log10(mse))


def _scaled_camera(view: RealView, downscale: float) -> Camera:
    return Camera(
        pose=view.camera.pose, intrinsics=scale_intrinsics(view.camera.intrinsics, downscale)
    )


def _eval_psnr(
    dist: GaussianDistiller, views: list[Camera], targets: list[np.ndarray]
) -> list[float]:
    out = []
    for cam, tgt in zip(views, targets, strict=True):
        pred = dist.render(cam).rgb.detach().clamp(0.0, 1.0).cpu().numpy()
        out.append(_psnr(pred, tgt))
    return out


def _infer_up_axis(cam_pos: np.ndarray) -> str:
    """Guess the world up axis from the spread of capture-camera positions.

    Indoor capture rigs vary far more in the floor plane than in height, so the world axis
    with the *smallest* positional spread is the up axis. Sign is chosen so that "up" points
    away from the camera cluster's lower half (cameras sit above the floor).
    """
    if cam_pos.shape[0] < 2:
        return "y"
    spread = cam_pos.std(axis=0)
    up_idx = int(np.argmin(spread))
    median = float(np.median(cam_pos[:, up_idx]))
    mean = float(cam_pos[:, up_idx].mean())
    sign = "-" if mean < median else "+"
    return f"{sign}{'xyz'[up_idx]}"


def synthesize_gap_poses(
    gap_centers: np.ndarray,
    cam_pos: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    up_axis: str,
    n_poses: int,
    standoff_frac: float = 0.6,
    neighbours: int = 4,
    rng_seed: int = 0,
) -> list[Camera]:
    """Place ``n_poses`` cameras in navigable space, each looking at a coverage gap.

    For each chosen gap centre, the camera is placed by backing off from the gap toward the
    centroid of its ``neighbours`` nearest real capture cameras — guaranteed-navigable vantage
    points that already observed nearby geometry — at ``standoff_frac`` of that distance. This
    keeps the synthetic camera out of solid geometry (it sits between a real camera and the gap)
    while framing the under-observed region. The real perspective ``intrinsics`` are reused so
    the FOV matches the training rig.
    """
    cameras: list[Camera] = []
    if gap_centers.shape[0] == 0 or cam_pos.shape[0] == 0:
        return cameras

    rng = np.random.default_rng(rng_seed)
    n_take = min(n_poses, gap_centers.shape[0])
    # Spread the picks across the gap set (deterministic, well-distributed) rather than taking
    # the first K, which would cluster in one corner of the voxel scan order.
    pick = np.unique(np.linspace(0, gap_centers.shape[0] - 1, n_take).astype(np.int64))
    k = min(neighbours, cam_pos.shape[0])

    for gi in pick:
        gap = gap_centers[gi].astype(np.float64)
        d2 = ((cam_pos - gap) ** 2).sum(axis=1)
        nearest = cam_pos[np.argsort(d2)[:k]]
        vantage = nearest.mean(axis=0)
        # A small deterministic jitter (scaled by the local camera spread) decorrelates poses
        # that share neighbours, so the batch frames the gap from slightly different angles.
        jitter = rng.normal(0.0, 1.0, size=3) * (nearest.std(axis=0) + 1e-6) * 0.15
        origin = gap + (vantage - gap) * standoff_frac + jitter
        rot = look_at(origin, gap, up_axis)
        cameras.append(Camera(pose=Pose(position=origin, rotation=rot), intrinsics=intrinsics))
    return cameras


def _pick_reference(
    gap: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    ref_select: str,
    angular_weight: float,
    dist_tiebreak: float = 0.25,
) -> int:
    """Index of the training view to use as the clean reference for ``gap``.

    ``ref_select="visible"`` (default) is the fix for "the reference points the wrong way": it
    projects the gap into every candidate camera and prefers the view that actually FRAMES it —
    in front of the camera, inside the image, nearest the principal point — with a mild distance
    tie-break. Difix reference-mixing can only borrow appearance the reference actually contains,
    so a view aimed at the gap beats one merely close in translation but facing elsewhere.

    Selection cascades so a reference is always returned: (1) views that see the gap in-frame,
    ranked by off-centre distance + ``dist_tiebreak``·(dist/mean_dist); else (2) front-facing
    views, ranked the same way on angular offset; else (3) the nearest by translation.

    ``ref_select="nearest"`` is the legacy translation argmin (optionally angular-penalised via
    ``angular_weight``) kept for comparison/back-compat.
    """
    d = gap[None, :] - cam_pos  # (M, 3) world rays camera -> gap
    dist = np.linalg.norm(d, axis=1)
    dscale = max(float(dist.mean()), 1e-6)

    if ref_select == "nearest":
        if angular_weight > 0.0:
            fwd = cam_rot[:, 2, :]  # world->camera: forward is the third row
            cos = np.sum(fwd * (d / np.maximum(dist, 1e-9)[:, None]), axis=1).clip(-1.0, 1.0)
            return int(np.argmin(dist * (1.0 + angular_weight * (1.0 - cos))))
        return int(np.argmin(dist))

    # Visibility-aware: project the gap into each camera (p_cam = R_w2c @ (gap - pos)).
    p_cam = np.einsum("mij,mj->mi", cam_rot, d)  # (M, 3)
    pz = p_cam[:, 2]
    in_front = pz > 1e-6
    safe_z = np.where(in_front, pz, 1.0)
    u = intrinsics.fx * p_cam[:, 0] / safe_z + intrinsics.cx
    v = intrinsics.fy * p_cam[:, 1] / safe_z + intrinsics.cy
    in_frame = in_front & (u >= 0) & (u < intrinsics.width) & (v >= 0) & (v < intrinsics.height)
    # Off-centre distance in normalized image units (0 = dead centre); the lower, the more the
    # reference squarely frames the gap.
    offcentre = np.hypot((u - intrinsics.cx) / intrinsics.width, (v - intrinsics.cy) / intrinsics.height)
    score = offcentre + dist_tiebreak * (dist / dscale)

    if in_frame.any():
        cand = np.nonzero(in_frame)[0]
        return int(cand[np.argmin(score[cand])])
    if in_front.any():
        cand = np.nonzero(in_front)[0]
        return int(cand[np.argmin(score[cand])])
    return int(np.argmin(dist))


def synthesize_near_view_poses(
    views: list[RealView],
    gap_centers: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    up_axis: str,
    n_poses: int,
    perturb_frac: float = 0.15,
    angular_weight: float = 0.0,
    ref_select: str = "visible",
) -> list[tuple[Camera, RealView]]:
    """Place NEAR-training-view poses that frame coverage gaps, each paired with its reference.

    This is the Difix3D+ regime, and it is the fix for the prior ``synthesize_gap_poses`` failure
    (which planted cameras *at* gaps → full-frame smear no artifact-fixer can repair). For each
    chosen gap a real training camera that FRAMES it becomes the clean **reference** (see
    :func:`_pick_reference`); the novel pose dollies ``perturb_frac`` of the way along the
    reference→gap ray and re-aims at the gap. So ``perturb_frac=0`` is the reference position
    re-aimed (pure rotation), and larger values move progressively closer to the gap — the lever
    the progressive loop grows each round.

    ``ref_select="visible"`` (default) picks the reference by projecting the gap into each camera
    and preferring the one that actually contains it (in-frame, in-front, near centre) — the fix
    for a reference that is close in translation but points away, leaving Difix nothing real to
    borrow. ``"nearest"`` is the legacy translation argmin.

    The offset is a fraction of the camera-to-gap *distance*, NOT of inter-camera spacing: 360 rigs
    extract many perspective views from the SAME physical centre, so camera spacing is ~0 and a
    spacing-scaled offset would collapse every pose onto the training position (silently disabling
    the perturbation). ``perturb_frac`` is clamped to ``[0, 0.9]`` so the camera stays before the gap.

    Returns ``(novel_camera, reference_view)`` pairs; the reference's on-disk image is loaded by the
    caller and handed to the :class:`ViewFiller` as ``ref_image``.
    """
    pairs: list[tuple[Camera, RealView]] = []
    if gap_centers.shape[0] == 0 or len(views) == 0:
        return pairs

    cam_pos = np.array([v.camera.pose.position for v in views], dtype=np.float64)
    cam_rot = np.array([v.camera.pose.rotation for v in views], dtype=np.float64)  # (M, 3, 3) w2c
    n_take = min(n_poses, gap_centers.shape[0])
    pick = np.unique(np.linspace(0, gap_centers.shape[0] - 1, n_take).astype(np.int64))
    frac = float(np.clip(perturb_frac, 0.0, 0.9))

    for gi in pick:
        gap = gap_centers[gi].astype(np.float64)
        ci = _pick_reference(
            gap, cam_pos, cam_rot, intrinsics, ref_select=ref_select, angular_weight=angular_weight
        )
        base = views[ci]
        pos0 = cam_pos[ci]
        to_gap = gap - pos0
        if float(np.linalg.norm(to_gap)) < 1e-6:
            continue
        origin = pos0 + to_gap * frac  # dolly a fraction of the way toward the gap
        rot = look_at(origin, gap, up_axis)
        pairs.append(
            (Camera(pose=Pose(position=origin, rotation=rot), intrinsics=intrinsics), base)
        )
    return pairs


def synthesize_marked_pairs(
    marks: list[Pose],
    views: list[RealView],
    intrinsics: CameraIntrinsics,
    *,
    angular_weight: float = 1.0,
) -> list[tuple[Camera, RealView]]:
    """Pair each ROBOT-marked pose with its best real reference — Difix runs on THESE frames.

    The mark *is* the frame to fix: the exact viewpoint the navigator flagged, not a synthesized
    novel pose dollied toward a coverage voxel. Difix cleans the splat's render AT the mark and the
    fix is distilled straight back. This is what makes the enhancement scene-agnostic and "do no
    harm": at a well-observed mark the render is already clean, so Difix returns ~identity and the
    disagreement mask is empty (nothing changes); at an under-observed mark it repairs it. Nothing
    is touched except where the robot pointed and the render is actually degraded.

    The reference is the real training view nearest the mark whose viewing direction best matches
    it (so it frames the same content) — the clean image Difix borrows appearance from.
    """
    pairs: list[tuple[Camera, RealView]] = []
    if not marks or not views:
        return pairs
    cam_pos = np.array([v.camera.pose.position for v in views], dtype=np.float64)
    cam_fwd = np.array([v.camera.pose.forward() for v in views], dtype=np.float64)
    for m in marks:
        d = cam_pos - m.position
        dist = np.linalg.norm(d, axis=1)
        dscale = max(float(dist.mean()), 1e-6)
        cos = (cam_fwd @ m.forward()).clip(-1.0, 1.0)  # view-direction alignment
        ci = int(np.argmin(dist / dscale + angular_weight * (1.0 - cos)))
        pairs.append((Camera(pose=m, intrinsics=intrinsics), views[ci]))
    return pairs


def enhance_scene(
    ply_path: str | Path,
    colmap_model_dir: str | Path,
    images_dir: str | Path,
    out_ply: str | Path,
    *,
    device: str = "cuda:0",
    camera_id: int | None = 1,
    downscale: float = 0.5,
    iters: int = 300,
    max_anchor: int = 256,
    eval_stride: int = 12,
    gap_grid: int = 32,
    densify: bool = False,
    freeze_geometry: bool = True,
    ssim_weight: float = 0.0,
    lrs: dict[str, float] | None = None,
) -> EnhanceReport:
    """Anchored, densification-controlled refinement of a splat; writes a NEW ply.

    Reads ``ply_path`` read-only and refuses to write back over it.
    """
    if Path(out_ply).resolve() == Path(ply_path).resolve():
        raise ValueError("out_ply must differ from ply_path; refusing to overwrite the original")

    cloud = load_gaussian_cloud(ply_path, device=device)
    views = load_colmap_views(colmap_model_dir, images_dir, camera_id=camera_id)
    if len(views) < eval_stride + 2:
        raise ValueError(f"too few registered views with images on disk: {len(views)}")

    # WHERE to enhance — the backbone coverage signal (occupied-but-unseen surface).
    means_np = cloud.means.detach().cpu().numpy()
    opac_np = cloud.opacities.detach().cpu().numpy().reshape(-1)
    cam_pos = np.array([v.camera.pose.position for v in views], dtype=np.float64)
    cam_rot = np.array([v.camera.pose.rotation for v in views], dtype=np.float64)
    hfov, vfov = camera_fovs(views)
    lo, hi = (
        cloud.density_bounds
        if cloud.density_bounds is not None
        else (
            cloud.full_bounds.min,
            cloud.full_bounds.max,
        )
    )
    cov = build_coverage3d(means_np, opac_np, cam_pos, cam_rot, hfov, vfov, lo, hi, grid=gap_grid)
    gap_count = int(cov.gap_centers().shape[0])

    # Deterministic split: every eval_stride-th view is held out for evaluation.
    eval_views = views[::eval_stride]
    anchor_views = [v for i, v in enumerate(views) if i % eval_stride != 0]
    if len(anchor_views) > max_anchor:
        sel = np.linspace(0, len(anchor_views) - 1, max_anchor).astype(int)
        anchor_views = [anchor_views[i] for i in sel]

    # Preload images at the working resolution.
    def _cam_and_img(v: RealView) -> tuple[Camera, np.ndarray]:
        cam = _scaled_camera(v, downscale)
        img = load_image(v.image_path, cam.intrinsics.width, cam.intrinsics.height)
        return cam, img

    anchor = [_cam_and_img(v) for v in anchor_views]
    eval_cams_imgs = [_cam_and_img(v) for v in eval_views]
    eval_cams = [c for c, _ in eval_cams_imgs]
    eval_imgs = [im for _, im in eval_cams_imgs]

    supervision = [SupervisionView(camera=c, target_rgb=im, mask=None) for c, im in anchor]

    # Safe default: freeze geometry, gentle colour/opacity LRs — an anchored polish that holds
    # held-out views. A naive global re-fit with aggressive LRs on a small anchor subset
    # OVERFITS and regresses held-out (the filler + gap-localized loss is what actually adds
    # new geometry; that is the next slot).
    if lrs is None and freeze_geometry:
        lrs = {"means": 0.0, "scales": 0.0, "quats": 0.0, "opacities": 5e-3, "sh": 2.5e-4}
    dist = GaussianDistiller(
        cloud,
        device=device,
        densify=densify,
        freeze_means_iters=0,
        ssim_weight=ssim_weight,
        lrs=lrs,
    )
    before = _eval_psnr(dist, eval_cams, eval_imgs)
    dist.fit(supervision, iters=iters)
    after = _eval_psnr(dist, eval_cams, eval_imgs)
    saved = dist.save_ply(out_ply)

    return EnhanceReport(
        n_views=len(views),
        n_anchor=len(anchor),
        n_eval=len(eval_views),
        n_gaussians=dist.num_gaussians,
        gap_count=gap_count,
        psnr_before=float(np.mean(before)),
        psnr_after=float(np.mean(after)),
        out_ply=str(saved),
        per_eval_before=before,
        per_eval_after=after,
    )


def build_filler(filler: str, *, device: str = "cuda:0", dtype: str = "float32") -> ViewFiller:
    """Construct the requested :class:`ViewFiller`.

    ``"difix"`` returns the generative :class:`DiffusionFiller` (the env probe confirmed Difix
    loads + runs at 6.8 GB on this box); ``"geometric"`` / ``"identity"`` return the no-weights
    :class:`GeometricFiller`. Imported lazily so the orchestrator module stays light and a
    missing diffusion dependency does not break the geometric path.

    ``dtype`` selects the Difix precision: ``"float32"`` (default) matches NVIDIA's reference tests;
    ``"float16"`` roughly halves the ~8.4 GB peak. fp16 is the cheapest VRAM-headroom lever, but is
    NOT guaranteed bit-for-bit free — validate output quality on a held-out view, since some
    AutoencoderKL VAEs are unstable in fp16. Ignored by the geometric path.
    """
    from gaussian_robot.enhance.fillers import DiffusionFiller, GeometricFiller  # noqa: PLC0415

    name = filler.strip().lower()
    if name == "difix":
        return DiffusionFiller(filler_mode="difix", device=device, dtype=dtype)
    if name in ("geometric", "identity"):
        return GeometricFiller()
    raise ValueError(f"unknown filler {filler!r}; expected 'difix', 'geometric' or 'identity'")


def _fill_gap_views(
    cloud: GaussianCloud,
    gap_pairs: list[tuple[Camera, RealView]],
    *,
    ref_size: tuple[int, int],
    filler: str,
    device: str,
    view_filler: ViewFiller | None,
    filler_dtype: str = "float32",
    diag_out: list[tuple[float, float]] | None = None,
) -> list[SupervisionView]:
    """Render each near-view gap pose DEGRADED, condition on its REAL reference, and fill it.

    Each ``(gap_camera, reference_view)`` pair (from :func:`synthesize_near_view_poses`) is rendered
    degraded; the reference view's on-disk image is loaded at ``ref_size`` and handed to the
    :class:`ViewFiller` as the clean ``ref_image`` (Difix3D+ reference-mixing). The filler owns the
    GPU only during this phase; an orchestrator-built filler is freed afterwards (``free()``) so the
    distiller can own the card. An injected ``view_filler`` is left untouched (caller owns it).

    When ``diag_out`` is given, each view appends ``(mask_frac, delta)``: the mean coverage mask
    (fraction of the frame the filler may change) and the mean absolute change the fill actually
    introduced over the raw render. ``delta ≈ 0`` means the round was a no-op — the symptom of a
    near-empty mask or a reference that gave the filler nothing to borrow.
    """
    renderer = GsplatRenderer(cloud)
    if view_filler is None:
        # Reclaim the distiller's reserved-but-unallocated VRAM before staging Difix — in the
        # progressive loop the (densified) distiller stays resident, so Difix must fit beside it.
        try:
            import torch as _torch  # noqa: PLC0415

            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except ImportError:
            pass
    chosen = (
        view_filler
        if view_filler is not None
        else build_filler(filler, device=device, dtype=filler_dtype)
    )
    w, h = ref_size
    gap_supervision: list[SupervisionView] = []
    for cam, ref_view in gap_pairs:
        degraded = renderer.render(cam)
        ref_img = load_image(ref_view.image_path, w, h)  # (H, W, 3) float in [0, 1]
        # The reference RenderResult must carry the REFERENCE pose, not the gap pose ``cam``: a
        # pose-aware filler (SEVA / PRoPE / Plücker) reads ``references[0].camera`` to recover the
        # ref->gap relative pose, and ``camera=cam`` would collapse it to identity. The current
        # difix filler reads ``.rgb`` only, so this is a correctness fix with no behaviour change
        # today — it unblocks any pose-conditioned filler injected via ``view_filler``.
        ref_rr = RenderResult(rgb=ref_img, camera=ref_view.camera)
        sv = chosen.fill(degraded, references=[ref_rr])
        gap_supervision.append(sv)
        if diag_out is not None:
            deg = np.asarray(degraded.rgb, dtype=np.float32)
            if deg.max() > 1.5:  # uint8 render -> normalize to match the [0,1] target
                deg = deg / 255.0
            tgt = np.asarray(sv.target_rgb, dtype=np.float32)
            mfrac = float(np.asarray(sv.mask).mean()) if sv.mask is not None else 1.0
            delta = float(np.abs(tgt - deg).mean()) if tgt.shape == deg.shape else 0.0
            diag_out.append((mfrac, delta))
    free = getattr(chosen, "free", None)
    if view_filler is None and callable(free):
        free()
    return gap_supervision


def _select_nearest_gaps(all_gaps: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Subset of ``all_gaps`` nearest to each row of ``centers`` (deduped, order-stable).

    Maps externally chosen vantage points — the robot's marked "poses to improve" — onto the
    live coverage gaps, so the fill targets the SAME surface holes the navigator flagged rather
    than spreading across every gap in the scene. Recomputed per round against the current
    coverage, so as nearby gaps are filled the mapping naturally tracks what remains. Empty
    ``centers`` (the robot found nothing) falls back to the full gap set.
    """
    if all_gaps.shape[0] == 0 or centers.shape[0] == 0:
        return all_gaps
    seen: set[int] = set()
    order: list[int] = []
    for c in centers:
        j = int(np.argmin(((all_gaps - c) ** 2).sum(axis=1)))
        if j not in seen:
            seen.add(j)
            order.append(j)
    return all_gaps[np.array(order, dtype=np.int64)]


def _progressive_distill(
    dist: GaussianDistiller,
    *,
    views: list[RealView],
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    hfov: np.ndarray,
    vfov: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    gap_grid: int,
    gap_intrinsics: CameraIntrinsics,
    resolved_up: str,
    anchor_supervision: list[SupervisionView],
    eval_cams: list[Camera],
    eval_imgs: list[np.ndarray],
    ref_size: tuple[int, int],
    n_gap_poses: int,
    iters: int,
    rounds: int,
    perturb_start: float,
    perturb_step: float,
    fill_weight: float,
    fill_gap_gain: float,
    restrict_to_gaps: bool,
    angular_weight: float,
    filler: str,
    device: str,
    view_filler: ViewFiller | None,
    filler_dtype: str,
    regression_tol_db: float,
    psnr_before: float,
    select_centers: np.ndarray | None = None,
    ref_select: str = "visible",
    gate: bool = True,
) -> tuple[list[float], list[float], int, float, float]:
    """Run the progressive fill rounds on ``dist``.

    Returns ``(per-round PSNR, best per-view PSNR, n_poses, last-round mask_frac, last-round delta)``.

    Each round (1) recomputes the coverage gaps from the CURRENT (improving) cloud so filled gaps
    drop out and the gap-gaussian index tracks the live geometry, (2) renders near-view poses whose
    offset grows with the round, (3) Difix-cleans them (fills weighted below the real anchors), and
    (4) distils opacity+SH back, restricted to gap gaussians. The best-scoring parameter state is
    retained and restored at the end; a round that drops more than ``regression_tol_db`` below the
    best reverts and stops the loop (so the result never regresses).
    """
    dist.fill_gap_gain = fill_gap_gain
    best_psnr = psnr_before
    best_snap = dist.state_snapshot()
    per_round: list[float] = []
    n_pairs = 0
    last_diag: list[tuple[float, float]] = []
    for r in range(rounds):
        perturb = perturb_start + r * perturb_step
        current = dist.to_cloud()  # snapshot of the current state to render the degraded views from
        # WHERE, recomputed on the current cloud: gap centres for pose synthesis + the gap-gaussian
        # index that confines this round's colour/opacity updates.
        means_np = current.means.detach().cpu().numpy()
        opac_np = current.opacities.detach().cpu().numpy().reshape(-1)
        cov_r = build_coverage3d(
            means_np, opac_np, cam_pos, cam_rot, hfov, vfov, lo, hi, grid=gap_grid
        )
        # Gap-restriction is OPT-IN: on a single-floor interior the gaussians that compose the
        # rendered hole at the near-view pose are NOT the gap-voxel gaussians, so restricting to
        # them zeros the fill gradient (measured). Keep it off by default; useful for larger holes.
        dist.gap_index = (
            torch.as_tensor(cov_r.gap_gaussian_mask(means_np), dtype=torch.bool, device=dist.device)
            if restrict_to_gaps
            else None
        )
        # WHERE the fill aims: the live coverage gaps, optionally narrowed to the ones the robot
        # navigator flagged (``select_centers``). With no robot input this is every gap (the
        # standalone fill_gaps behaviour).
        round_gaps = cov_r.gap_centers()
        if select_centers is not None:
            round_gaps = _select_nearest_gaps(round_gaps, select_centers)
        gap_pairs = synthesize_near_view_poses(
            views, round_gaps, gap_intrinsics, up_axis=resolved_up,
            n_poses=n_gap_poses, perturb_frac=perturb, angular_weight=angular_weight,
            ref_select=ref_select,
        )
        n_pairs = len(gap_pairs)
        round_diag: list[tuple[float, float]] = []
        gap_supervision = _fill_gap_views(
            current, gap_pairs, ref_size=ref_size, filler=filler,
            device=device, view_filler=view_filler, filler_dtype=filler_dtype,
            diag_out=round_diag,
        )
        last_diag = round_diag
        del current  # free the render snapshot before the distiller fit owns the card
        gap_supervision = [replace(sv, weight=fill_weight) for sv in gap_supervision]
        supervision = _interleave(gap_supervision, anchor_supervision)
        if not supervision:
            raise ValueError("no supervision views (no gap poses and no anchors)")
        dist.fit(supervision, iters=iters)
        post = float(np.mean(_eval_psnr(dist, eval_cams, eval_imgs)))
        per_round.append(post)
        if gate:
            if post > best_psnr:
                best_psnr = post
                best_snap = dist.state_snapshot()
            elif best_psnr - post > regression_tol_db:
                break  # regressed beyond tolerance; revert to the best state and stop

    if gate:
        dist.load_snapshot(best_snap)  # ship the best-scoring params (never worse than the input)
    # gate=False: keep the FINAL round's state as-is — the non-conservative path that lets a
    # visible enhancement ship even if it does not raise the held-out real-view PSNR.
    mask_frac = float(np.mean([d[0] for d in last_diag])) if last_diag else 0.0
    delta = float(np.mean([d[1] for d in last_diag])) if last_diag else 0.0
    return per_round, _eval_psnr(dist, eval_cams, eval_imgs), n_pairs, mask_frac, delta


def fill_gaps_scene(
    ply_path: str | Path,
    colmap_model_dir: str | Path,
    images_dir: str | Path,
    out_ply: str | Path,
    *,
    device: str = "cuda:0",
    camera_id: int | None = 1,
    downscale: float = 0.5,
    filler: str = "difix",
    iters: int = 300,
    rounds: int = 3,
    perturb_start: float = 0.15,
    perturb_step: float = 0.2,
    n_gap_poses: int = 12,
    max_anchor: int = 64,
    eval_stride: int = 12,
    gap_grid: int = 32,
    cap_max_factor: float = 1.05,
    up_axis: str = "auto",
    lrs: dict[str, float] | None = None,
    regression_tol_db: float = 0.3,
    fill_weight: float = 1.0,
    anchor_weight: float = 1.0,
    fill_gap_gain: float = 1.0,
    restrict_to_gaps: bool = False,
    ssim_weight: float = 0.0,
    angular_weight: float = 0.0,
    view_filler: ViewFiller | None = None,
    filler_dtype: str = "float32",
    select_centers: np.ndarray | None = None,
    ref_select: str = "visible",
    gate: bool = True,
) -> FillReport:
    """Progressive Difix3D+ gap-fill: distil reference-conditioned fills WITHOUT regressing.

    ``gate=False`` is the non-conservative path: it ships the FINAL round's state regardless of the
    held-out real-view PSNR (no revert-to-best, no hard regression error). Combine with unfrozen
    geometry / higher LRs (``lrs``) and a larger ``fill_weight`` to let the splat actually adopt the
    Difix-sharpened targets — at the cost of the no-regression guarantee.

    ``select_centers`` ``(K, 3)`` are external vantage points — typically the robot navigator's
    marked "poses to improve" — that narrow the fill to the coverage gaps nearest them, instead
    of spreading across every gap. ``None`` keeps the standalone behaviour (all gaps).

    The progressive 3D update of Difix3D+: ``rounds`` of (render the CURRENT cloud at near-view gap
    poses → reference-conditioned Difix clean → distil opacity+SH back), with the pose offset
    ``perturb_frac`` GROWING each round (``perturb_start + r*perturb_step``). Early rounds fix
    content close to the training views; because the cloud is already improved, later rounds can
    push the cameras farther toward the gaps while the Difix conditioning stays strong — which is
    how the spatial extent of good reconstruction grows.

    Per round:

    1. ``build_coverage3d(...).gap_centers()`` (occupied-but-unseen surface) says WHERE; the gaps
       are computed once on the original cloud and reused every round.
    2. :func:`synthesize_near_view_poses` nudges the NEAREST training camera ``perturb_frac`` of the
       local camera spacing toward each gap and pairs it with that camera's REAL image as the clean
       reference (a mostly-observed frame with a bounded hole — Difix's regime).
    3. Each pose is rendered DEGRADED from the CURRENT (distilled) cloud; the reference-conditioned
       ``difix_ref`` pipeline cleans it into a target + coverage mask ``M`` (holes only).
    4. A :class:`GaussianDistiller` with **geometry frozen, densification OFF** raises opacity +
       fixes colour (SH) of the gaussians already occupying the gap voxels, on a 50/50 mix of the
       filled views (``mask=M``) and real ANCHOR views (``mask=None``) that pin the rest.

    Safety: the held-out real PSNR is evaluated after every round; the **best-scoring** parameter
    state is retained, and if a round drops more than ``regression_tol_db`` below the best the loop
    reverts to the best and stops. The exported ply is therefore never worse than the input, so the
    final hard gate can only fail in degenerate cases. ``rounds=1`` is the single-pass behaviour.
    The input is read-only (``out_ply != ply_path``).
    """
    if Path(out_ply).resolve() == Path(ply_path).resolve():
        raise ValueError("out_ply must differ from ply_path; refusing to overwrite the original")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if fill_gap_gain != 1.0 and not restrict_to_gaps:
        # fill_gap_gain only scales the gap-restricted fill gradient; with restriction off it is a
        # silent no-op. Fail loudly rather than let a caller think they are pushing the fill harder.
        raise ValueError("fill_gap_gain != 1.0 requires restrict_to_gaps=True (it scales the "
                         "gap-restricted fill gradient; with no gap mask it has no effect)")

    cloud = load_gaussian_cloud(ply_path, device=device)
    views = load_colmap_views(colmap_model_dir, images_dir, camera_id=camera_id)
    if len(views) < eval_stride + 2:
        raise ValueError(f"too few registered views with images on disk: {len(views)}")

    # WHERE: occupied-but-unseen surface voxels (the same coverage signal the densify mode uses).
    means_np = cloud.means.detach().cpu().numpy()
    opac_np = cloud.opacities.detach().cpu().numpy().reshape(-1)
    cam_pos = np.array([v.camera.pose.position for v in views], dtype=np.float64)
    cam_rot = np.array([v.camera.pose.rotation for v in views], dtype=np.float64)
    hfov, vfov = camera_fovs(views)
    lo, hi = (
        cloud.density_bounds
        if cloud.density_bounds is not None
        else (cloud.full_bounds.min, cloud.full_bounds.max)
    )
    cov = build_coverage3d(means_np, opac_np, cam_pos, cam_rot, hfov, vfov, lo, hi, grid=gap_grid)
    gap_centers = cov.gap_centers()
    gap_count = int(gap_centers.shape[0])

    resolved_up = _infer_up_axis(cam_pos) if up_axis == "auto" else up_axis

    # Deterministic split: every eval_stride-th REAL view is held out as the regression guard.
    eval_views = views[::eval_stride]
    anchor_views = [v for i, v in enumerate(views) if i % eval_stride != 0]
    if len(anchor_views) > max_anchor:
        sel = np.linspace(0, len(anchor_views) - 1, max_anchor).astype(int)
        anchor_views = [anchor_views[i] for i in sel]

    def _cam_and_img(v: RealView) -> tuple[Camera, np.ndarray]:
        cam = _scaled_camera(v, downscale)
        img = load_image(v.image_path, cam.intrinsics.width, cam.intrinsics.height)
        return cam, img

    anchor = [_cam_and_img(v) for v in anchor_views]
    eval_cams_imgs = [_cam_and_img(v) for v in eval_views]
    eval_cams = [c for c, _ in eval_cams_imgs]
    eval_imgs = [im for _, im in eval_cams_imgs]

    gap_intrinsics = scale_intrinsics(views[0].camera.intrinsics, downscale)
    ref_size = (gap_intrinsics.width, gap_intrinsics.height)
    anchor_supervision = [
        SupervisionView(camera=c, target_rgb=im, mask=None, weight=anchor_weight)
        for c, im in anchor
    ]

    # Geometry FROZEN, densification OFF. Coverage gaps are occupied-but-unseen voxels — the
    # gaussians already exist there; the fill is achieved by raising their opacity and correcting
    # colour (SH), NOT by moving/adding geometry. Moving geometry or running MCMC over a thin
    # anchor set is precisely what regressed held-out views by ~1.5 dB in the prior design.
    # GENTLE opacity/SH LRs (the Milestone-0 no-regress polish config): aggressive opacity LRs
    # overfit the thin anchor set and *also* regress held-out (measured −4.8 dB at opac=2e-2).
    if lrs is None:
        lrs = {"means": 0.0, "scales": 0.0, "quats": 0.0, "opacities": 5e-3, "sh": 2.5e-4}
    dist = GaussianDistiller(
        cloud,
        device=device,
        densify=False,
        cap_max_factor=cap_max_factor,
        freeze_means_iters=0,
        ssim_weight=ssim_weight,  # opt-in masked-SSIM on fills; 0 (L1-only) is best on this regime
        lrs=lrs,
    )
    n_before = dist.num_gaussians
    before = _eval_psnr(dist, eval_cams, eval_imgs)
    psnr_before = float(np.mean(before)) if before else 0.0

    per_round, after, n_pairs, mask_frac, fill_delta = _progressive_distill(
        dist,
        views=views,
        cam_pos=cam_pos,
        cam_rot=cam_rot,
        hfov=hfov,
        vfov=vfov,
        lo=lo,
        hi=hi,
        gap_grid=gap_grid,
        gap_intrinsics=gap_intrinsics,
        resolved_up=resolved_up,
        anchor_supervision=anchor_supervision,
        eval_cams=eval_cams,
        eval_imgs=eval_imgs,
        ref_size=ref_size,
        n_gap_poses=n_gap_poses,
        iters=iters,
        rounds=rounds,
        perturb_start=perturb_start,
        perturb_step=perturb_step,
        fill_weight=fill_weight,
        fill_gap_gain=fill_gap_gain,
        restrict_to_gaps=restrict_to_gaps,
        angular_weight=angular_weight,
        filler=filler,
        device=device,
        view_filler=view_filler,
        filler_dtype=filler_dtype,
        regression_tol_db=regression_tol_db,
        psnr_before=psnr_before,
        select_centers=select_centers,
        ref_select=ref_select,
        gate=gate,
    )
    psnr_after = float(np.mean(after)) if after else 0.0
    # HARD GATE (the round logic already keeps psnr_after >= psnr_before; this is the backstop).
    # Skipped when gate=False (the caller opted into a possibly-regressing, more visible result).
    if gate and before and (psnr_before - psnr_after) > regression_tol_db:
        raise RuntimeError(
            f"gap-fill regressed held-out PSNR {psnr_before:.3f} -> {psnr_after:.3f} dB "
            f"(drop {psnr_before - psnr_after:.3f} > tol {regression_tol_db}); refusing to write "
            f"a worse splat. Loosen lrs / iters or raise regression_tol_db to inspect."
        )
    saved = dist.save_ply(out_ply)

    return FillReport(
        n_views=len(views),
        n_anchor=len(anchor),
        n_eval=len(eval_views),
        gap_count=gap_count,
        n_gap_poses=n_pairs,
        n_gaussians_before=n_before,
        n_gaussians_after=dist.num_gaussians,
        filler=filler if view_filler is None else type(view_filler).__name__,
        psnr_before=psnr_before,
        psnr_after=psnr_after,
        out_ply=str(saved),
        rounds_run=len(per_round),
        per_round_psnr=per_round,
        per_eval_before=before,
        per_eval_after=after,
        fill_mask_frac=mask_frac,
        fill_delta=fill_delta,
    )


# Moderate LRs for the progressive scheme: geometry MOVES (unlike the frozen no-regress polish) but
# gently, and MCMC densification — not distortion of existing gaussians — carries most of the new
# detail. The one-shot aggressive path (high SH LR + unfrozen geometry, densify OFF) produced
# rainbow artifacts precisely because existing gaussians were forced to fit inconsistent fills.
_PROGRESSIVE_LRS: dict[str, float] = {
    "means": 8e-5, "scales": 2.5e-3, "quats": 5e-4, "opacities": 2e-2, "sh": 1.5e-3,
}


@dataclass
class _FillScene:
    """Loaded scene + the deterministic anchor/eval split shared by the fill orchestrators."""

    cloud: GaussianCloud
    views: list[RealView]
    cam_pos: np.ndarray
    cam_rot: np.ndarray
    hfov: np.ndarray
    vfov: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    resolved_up: str
    n_anchor: int
    n_eval: int
    anchor_supervision: list[SupervisionView]
    eval_cams: list[Camera]
    eval_imgs: list[np.ndarray]
    gap_intrinsics: CameraIntrinsics
    ref_size: tuple[int, int]


def _load_scene_for_fill(
    ply_path: str | Path,
    colmap_model_dir: str | Path,
    images_dir: str | Path,
    *,
    device: str,
    camera_id: int | None,
    downscale: float,
    eval_stride: int,
    max_anchor: int,
) -> _FillScene:
    """Load the cloud + COLMAP views and build the anchor/held-out split + gap intrinsics."""
    cloud = load_gaussian_cloud(ply_path, device=device)
    views = load_colmap_views(colmap_model_dir, images_dir, camera_id=camera_id)
    if len(views) < eval_stride + 2:
        raise ValueError(f"too few registered views with images on disk: {len(views)}")

    cam_pos = np.array([v.camera.pose.position for v in views], dtype=np.float64)
    cam_rot = np.array([v.camera.pose.rotation for v in views], dtype=np.float64)
    hfov, vfov = camera_fovs(views)
    lo, hi = (
        cloud.density_bounds
        if cloud.density_bounds is not None
        else (cloud.full_bounds.min, cloud.full_bounds.max)
    )

    eval_views = views[::eval_stride]
    anchor_views = [v for i, v in enumerate(views) if i % eval_stride != 0]
    if len(anchor_views) > max_anchor:
        sel = np.linspace(0, len(anchor_views) - 1, max_anchor).astype(int)
        anchor_views = [anchor_views[i] for i in sel]

    def _cam_and_img(v: RealView) -> tuple[Camera, np.ndarray]:
        cam = _scaled_camera(v, downscale)
        return cam, load_image(v.image_path, cam.intrinsics.width, cam.intrinsics.height)

    anchor = [_cam_and_img(v) for v in anchor_views]
    eval_cams_imgs = [_cam_and_img(v) for v in eval_views]
    gap_intrinsics = scale_intrinsics(views[0].camera.intrinsics, downscale)
    return _FillScene(
        cloud=cloud,
        views=views,
        cam_pos=cam_pos,
        cam_rot=cam_rot,
        hfov=hfov,
        vfov=vfov,
        lo=np.asarray(lo, dtype=np.float64),
        hi=np.asarray(hi, dtype=np.float64),
        resolved_up=_infer_up_axis(cam_pos),
        n_anchor=len(anchor),
        n_eval=len(eval_views),
        anchor_supervision=[
            SupervisionView(camera=c, target_rgb=im, mask=None) for c, im in anchor
        ],
        eval_cams=[c for c, _ in eval_cams_imgs],
        eval_imgs=[im for _, im in eval_cams_imgs],
        gap_intrinsics=gap_intrinsics,
        ref_size=(gap_intrinsics.width, gap_intrinsics.height),
    )


def _recoverable_pairs(
    cloud: GaussianCloud,
    pairs: list[tuple[Camera, RealView]],
    hole_max: float,
) -> list[tuple[Camera, RealView]]:
    """Keep only candidate frames the current cloud actually COVERS (recoverable, not a smear).

    Renders each candidate's accumulated alpha from ``cloud`` and drops those whose
    ``hole_fraction`` exceeds ``hole_max`` — frames looking into still-empty space, where Difix
    would invent rather than repair. They re-qualify in a later step once the cloud improves.
    """
    from gaussian_robot.enhance.frame_quality import hole_fraction  # noqa: PLC0415

    renderer = GsplatRenderer(cloud)
    keep: list[tuple[Camera, RealView]] = []
    for cam, ref in pairs:
        rr = renderer.render(cam)
        if hole_fraction(rr.alpha) <= hole_max:
            keep.append((cam, ref))
    return keep


def _progressive_step(
    dist: GaussianDistiller,
    scene: _FillScene,
    view_filler: ViewFiller | None,
    pseudo: list[SupervisionView],
    *,
    step_idx: int,
    perturb: float,
    gstep: int,
    iters_per_step: int,
    gap_grid: int,
    select_centers: np.ndarray | None,
    n_gap_poses: int,
    ref_select: str,
    filler: str,
    device: str,
    filler_dtype: str,
    buffer_cap: int,
    frame_poses: list[Pose] | None = None,
    target_centers: np.ndarray | None = None,
    recover_hole_max: float = 0.10,
) -> tuple[list[SupervisionView], float, list[tuple[float, float]], int, int]:
    """One progressive step: re-render the current cloud, clean the recoverable views, fit.

    Modes for choosing the views to fix:
    - ``target_centers`` (the progressive frontier — RECOMMENDED): march from each real camera
      ``perturb`` of the way toward a robot-flagged target and re-aim. ``perturb`` grows each step,
      so the frame starts AT a real camera (sharp, real signal) and creeps outward. A per-frame
      **recoverability gate** then drops any candidate whose render is under-covered
      (``hole_fraction > recover_hole_max``) — a total smear with no real signal, which Difix could
      only hallucinate. Those frames are left for a later step: once densification + the earlier
      fixes improve the cloud, they become covered and pass the gate. This is "grow the sharp region
      outward from the real cameras; never fix what is still a smear".
    - ``frame_poses`` — fix exactly the robot poses (no march).
    - neither — synthesized coverage-gap poses (legacy fallback).

    Returns ``(pseudo, held_out_psnr, diag, n_pairs, gap_count)``.  When ``view_filler is None`` the
    Difix model is built and FREED inside ``_fill_gap_views`` each step — essential here: keeping it
    resident on the GPU through the densified, grad-heavy distill OOMs a 24 GB card.
    """
    current = dist.to_cloud()
    if target_centers is not None:
        cand = synthesize_near_view_poses(
            scene.views, target_centers, scene.gap_intrinsics, up_axis=scene.resolved_up,
            n_poses=target_centers.shape[0], perturb_frac=perturb, ref_select=ref_select,
        )
        gap_pairs = _recoverable_pairs(current, cand, recover_hole_max)
        n_gaps = len(gap_pairs)
    elif frame_poses is not None:
        gap_pairs = synthesize_marked_pairs(frame_poses, scene.views, scene.gap_intrinsics)
        n_gaps = len(gap_pairs)
    else:
        means_np = current.means.detach().cpu().numpy()
        opac_np = current.opacities.detach().cpu().numpy().reshape(-1)
        cov_r = build_coverage3d(
            means_np, opac_np, scene.cam_pos, scene.cam_rot, scene.hfov, scene.vfov,
            scene.lo, scene.hi, grid=gap_grid,
        )
        round_gaps = cov_r.gap_centers()
        if select_centers is not None:
            round_gaps = _select_nearest_gaps(round_gaps, select_centers)
        gap_pairs = synthesize_near_view_poses(
            scene.views, round_gaps, scene.gap_intrinsics, up_axis=scene.resolved_up,
            n_poses=n_gap_poses, perturb_frac=perturb, ref_select=ref_select,
        )
        n_gaps = int(round_gaps.shape[0])
    if not gap_pairs:  # nothing recoverable at this frontier yet — fit anchors only, hold steady
        post = float(np.mean(_eval_psnr(dist, scene.eval_cams, scene.eval_imgs)))
        return pseudo, post, [], 0, n_gaps
    diag: list[tuple[float, float]] = []
    new_fills = _fill_gap_views(
        current, gap_pairs, ref_size=scene.ref_size, filler=filler,
        device=device, view_filler=view_filler, filler_dtype=filler_dtype, diag_out=diag,
    )
    del current

    # Store accumulated fills as float16 — the buffer can hold dozens of full-frame targets, and
    # the distiller upcasts to float32 on use; float32 storage was a needless ~2x CPU-RAM cost.
    new_fills = [
        replace(
            sv,
            target_rgb=np.asarray(sv.target_rgb, dtype=np.float16),
            mask=None if sv.mask is None else np.asarray(sv.mask, dtype=np.float16),
        )
        for sv in new_fills
    ]
    pseudo = pseudo + new_fills
    if len(pseudo) > buffer_cap:
        pseudo = pseudo[-buffer_cap:]
    # Balanced batch: this step's fresh fills + a sample of older fills (~anchor count) + all
    # anchors, then a deterministic shuffle so iters < len still covers the set uniformly.
    n_new = len(new_fills)
    older = pseudo[:-n_new] if n_new else pseudo
    rng = np.random.default_rng(step_idx)
    if len(older) > len(scene.anchor_supervision):
        pick = rng.choice(len(older), size=len(scene.anchor_supervision), replace=False)
        older = [older[i] for i in pick]
    batch = _interleave(new_fills + list(older), scene.anchor_supervision)
    batch = [batch[i] for i in rng.permutation(len(batch))]
    if batch:
        dist.fit(batch, iters=iters_per_step, step_offset=gstep)
    post = float(np.mean(_eval_psnr(dist, scene.eval_cams, scene.eval_imgs)))
    return pseudo, post, diag, len(gap_pairs), n_gaps


def fill_gaps_progressive(
    ply_path: str | Path,
    colmap_model_dir: str | Path,
    images_dir: str | Path,
    out_ply: str | Path,
    *,
    device: str = "cuda:0",
    camera_id: int | None = 1,
    downscale: float = 0.5,
    filler: str = "difix",
    filler_dtype: str = "float16",
    steps: int = 12,
    iters_per_step: int = 150,
    n_gap_poses: int = 16,
    max_anchor: int = 64,
    eval_stride: int = 12,
    gap_grid: int = 32,
    perturb_start: float = 0.05,
    perturb_end: float = 0.6,
    cap_max_factor: float = 1.1,
    densify: bool = True,
    lrs: dict[str, float] | None = None,
    freeze_means_iters: int = 100,
    ssim_weight: float = 0.0,
    select_centers: np.ndarray | None = None,
    ref_select: str = "visible",
    buffer_cap: int = 80,
    view_filler: ViewFiller | None = None,
    frame_poses: list[Pose] | None = None,
    target_centers: np.ndarray | None = None,
    recover_hole_max: float = 0.10,
    gate: bool = True,
    regression_tol_db: float = 0.2,
) -> FillReport:
    """Faithful progressive Difix3D+ gap-fill: grow the reconstruction in small, consistent steps.

    ``frame_poses`` (the robot's marked viewpoints) makes this scene-agnostic: Difix runs ONLY on
    those frames, so a well-observed scene is left essentially untouched (do no harm) while flagged
    under-observed frames get repaired. With ``gate=True`` the best-scoring (held-out) state is kept
    and a step that regresses beyond ``regression_tol_db`` reverts — a hard guarantee the result is
    never worse than the input, on any scene.

    The one-shot path (synthesize far novel poses, clean them, distil once) breaks because Difix's
    per-view 2D fixes are not multi-view-consistent: pushed hard they corrupt geometry/SH (rainbow
    artifacts, large held-out regression). This routine follows the published Difix3D+ scheme:

    1. **Small steps.** ``perturb_frac`` grows linearly ``perturb_start -> perturb_end`` over
       ``steps`` increments, so each rendered novel view is only mildly degraded — Difix changes it
       little and stays consistent with the real geometry.
    2. **Re-render from the improving cloud each step**, so a fix is always relative to the latest
       (already-better) state, not the stale original.
    3. **Accumulate** every cleaned view into a growing pseudo-view buffer; the 3DGS is optimised on
       a balanced mix of those + the REAL anchors, so all prior fixes (and the real images) jointly
       constrain the geometry — no single inconsistent view can dominate.
    4. **Densification ON.** New detail is added as NEW gaussians (MCMC relocate/add) instead of
       distorting existing ones — the missing ingredient that made the one-shot aggressive run melt.

    Ships the final (consistent) state; the input PLY is read-only.
    """
    if Path(out_ply).resolve() == Path(ply_path).resolve():
        raise ValueError("out_ply must differ from ply_path; refusing to overwrite the original")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    scene = _load_scene_for_fill(
        ply_path, colmap_model_dir, images_dir, device=device, camera_id=camera_id,
        downscale=downscale, eval_stride=eval_stride, max_anchor=max_anchor,
    )

    dist = GaussianDistiller(
        scene.cloud,
        device=device,
        densify=densify,
        cap_max_factor=cap_max_factor,
        freeze_means_iters=freeze_means_iters,
        ssim_weight=ssim_weight,
        lrs=lrs or _PROGRESSIVE_LRS,
    )
    n_before = dist.num_gaussians
    before = _eval_psnr(dist, scene.eval_cams, scene.eval_imgs)
    psnr_before = float(np.mean(before)) if before else 0.0

    # Difix is built+freed PER STEP inside _fill_gap_views (view_filler=None): keeping it resident
    # on the GPU through the densified grad-heavy distill OOMs a 24 GB card. The reload is from the
    # local HF cache (~seconds). An injected view_filler (tests) stays resident — caller owns it.
    pseudo: list[SupervisionView] = []
    per_step: list[float] = []
    last_diag: list[tuple[float, float]] = []
    n_pairs = 0
    last_gaps = 0
    best_psnr = psnr_before
    best_snap = dist.state_snapshot() if gate else None
    for s in range(steps):
        perturb = perturb_start + (perturb_end - perturb_start) * (s / max(steps - 1, 1))
        pseudo, post, diag, n_pairs, last_gaps = _progressive_step(
            dist, scene, view_filler, pseudo,
            step_idx=s, perturb=perturb, gstep=s * iters_per_step, iters_per_step=iters_per_step,
            gap_grid=gap_grid, select_centers=select_centers, n_gap_poses=n_gap_poses,
            ref_select=ref_select, filler=filler, device=device, filler_dtype=filler_dtype,
            buffer_cap=buffer_cap, frame_poses=frame_poses, target_centers=target_centers,
            recover_hole_max=recover_hole_max,
        )
        last_diag = diag or last_diag
        per_step.append(post)
        vram = (
            torch.cuda.max_memory_allocated(device) / 1e9
            if torch.cuda.is_available()
            else 0.0
        )
        md = last_diag[0] if last_diag else (0.0, 0.0)
        print(
            f"  step {s + 1}/{steps}  perturb={perturb:.2f}  gaussians={dist.num_gaussians}  "
            f"frames={n_pairs}  mask={md[0]:.3f} delta={md[1]:.4f}  "
            f"held-out PSNR={post:.3f}  peakVRAM={vram:.1f}GB",
            flush=True,
        )
        if gate:
            if post > best_psnr:
                best_psnr = post
                best_snap = dist.state_snapshot()
            elif best_psnr - post > regression_tol_db:
                print(f"  gate: step regressed > {regression_tol_db}dB, reverting to best & stopping",
                      flush=True)
                break

    if gate and best_snap is not None:
        dist.load_snapshot(best_snap)  # ship the best-scoring state — never worse than the input
    after = _eval_psnr(dist, scene.eval_cams, scene.eval_imgs)
    psnr_after = float(np.mean(after)) if after else 0.0
    mask_frac = float(np.mean([d[0] for d in last_diag])) if last_diag else 0.0
    fill_delta = float(np.mean([d[1] for d in last_diag])) if last_diag else 0.0
    saved = dist.save_ply(out_ply)

    return FillReport(
        n_views=len(scene.views),
        n_anchor=scene.n_anchor,
        n_eval=scene.n_eval,
        gap_count=last_gaps,
        n_gap_poses=n_pairs,
        n_gaussians_before=n_before,
        n_gaussians_after=dist.num_gaussians,
        filler=filler if view_filler is None else type(view_filler).__name__,
        psnr_before=psnr_before,
        psnr_after=psnr_after,
        out_ply=str(saved),
        rounds_run=len(per_step),
        per_round_psnr=per_step,
        per_eval_before=before,
        per_eval_after=after,
        fill_mask_frac=mask_frac,
        fill_delta=fill_delta,
    )


def _interleave(gap: list[SupervisionView], anchor: list[SupervisionView]) -> list[SupervisionView]:
    """Round-robin two lists into a ~50/50 alternating sequence (longer list fills the tail)."""
    out: list[SupervisionView] = []
    i = j = 0
    while i < len(gap) or j < len(anchor):
        if i < len(gap):
            out.append(gap[i])
            i += 1
        if j < len(anchor):
            out.append(anchor[j])
            j += 1
    return out
