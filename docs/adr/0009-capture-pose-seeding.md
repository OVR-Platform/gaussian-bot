# 0009. Seed walks from the splat's capture poses

- Status: accepted
- Date: 2026-06-23

## Context

ADR-0003 calls for launching walks from "spread-out training poses," but the
implementation had drifted to *guessing* seed positions from the renderer's
density grid (sampling cells weighted by `sqrt(density)`) and validating each
guess with a test render. In practice seeds still landed in empty space: the
density grid is coarse, and a cell with some gaussians does not mean a *camera*
placed there sees reconstructed geometry — it may sit inside a wall, above the
floor, or facing a hole. Render validation tied to step size (`median_depth <
step`) then over-rejected the few good guesses on large scenes.

The poses the splat was reconstructed from are the ground truth for "where can a
camera stand and see real geometry?" — every one is a viewpoint that was
actually observed. We have them on disk next to every trained scene.

## Decision

When capture poses are available, use them as the **seed-candidate pool** instead
of density guesses:

1. **Source, frame-safe first.** Prefer the 3DGS `cameras.json` co-located with
   the PLY — it is emitted by the trainer in the *same world frame as the saved
   gaussians*, so positions need no alignment. Fall back to COLMAP
   `images.bin`/`images.txt`. Raw COLMAP output may be rescaled/realigned before
   training, so it is only trustworthy when its frame matches the PLY. Poses are
   converted to the repo convention (ADR-0002): `cameras.json` stores
   camera->world rotation, so `Pose.rotation` is its transpose; COLMAP stores
   world->camera directly.
2. **Spread, not rank.** Capture poses cluster (rig bursts, slow walks), so we
   pick a diverse subset by farthest-point selection on floor-plane position
   (the same machinery as ADR-0008) and **preserve that order**. We do *not*
   re-rank capture seeds by render depth: a "sees-furthest" score collapses the
   spatial spread toward whichever viewpoint happens to look across the scene.
3. **Lenient validation.** Capture seeds are real viewpoints, so we only reject
   ones that look into the void (near-zero alpha or mostly-infinite depth). The
   step-tied median-depth floor stays only for the density/grid fallback.

Seeds keep their **original capture orientation** rather than a synthetic
look-at, since that orientation is itself a known-good view.

Discovery is automatic (walk up from the PLY); `RunConfig.poses_path` overrides
it and `use_capture_pose_seeds` disables it. When no capture poses are found, the
prior density/grid seeding is used unchanged.

### Two orientation bugs surfaced by real viewpoints

Seeding from real capture views (instead of empty guesses) made two latent
orientation bugs visible — both now fixed:

1. **Renderer vertical flip.** `GsplatRenderer` flipped every render vertically
   on a false "OpenGL Y-up framebuffer" assumption. gsplat with an OpenCV
   `viewmat` already returns a top-left-origin image, so the flip inverted RGB
   *and* depth (corrupting the map projection). Removed. Verified by matching a
   rendered capture view to its ground-truth photo.
2. **Signed + auto up axis.** The capture cameras' mean world-up is `-Y` for the
   reference scene (COLMAP/3DGS reconstructions often have gravity-up along a
   negative axis). `up_axis` previously accepted only positive `x/y/z`, and
   `look_at` hard-coded `+axis`, so synthetic poses (preview, seed fallbacks,
   navigation) were upside down. `up_axis` now accepts signed values
   (`-y`, `+z`, ...); floor-plane projections still ignore the sign.

   Requiring the operator to know the sign is not intuitive, so `up_axis`
   defaults to **`"auto"`**: the up direction is inferred from the capture poses'
   averaged up vector (`infer_up_axis`) at session-build time, falling back to
   `+y` when no poses are available. An explicit `up_axis` still overrides
   detection. Up-axis detection runs even when capture-pose *seeding* is
   disabled, so the two concerns are independent.

## Consequences

- ✅ Seeds start where geometry provably exists — the original failure mode
  (seeds in empty space) is eliminated for any scene with capture poses.
- ✅ Realises ADR-0003's "spread-out training poses" as written.
- ✅ Backward compatible: synthetic scenes and PLYs without capture poses fall
  back to the density/grid path; one new test covers the no-poses case.
- ⚠️ Trusts that `cameras.json` shares the PLY frame. True for standard 3DGS
  trainers; a trainer that re-centres the PLY after dumping cameras would break
  it, and raw COLMAP dirs from a rescaled pipeline must not be pointed at via
  `poses_path`.
- ⚠️ Capture-pose density follows where the operator walked, not coverage need;
  frontier-style seeding toward under-reconstructed regions remains future work.
