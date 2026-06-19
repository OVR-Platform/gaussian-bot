# 0002. Coordinate & camera convention

- Status: accepted
- Date: 2026-06-19

## Context

Actions (Q2) manipulate a `Pose` by "move forward", "turn left", etc. This is
impossible to implement without a pinned convention for (a) the world frame and
(b) how `Pose.rotation` maps to a viewing direction. The scaffold left this
"OpenCV/OpenGL agnostic". It must be fixed to make `apply_action` correct, and
it must eventually align with whichever renderer we choose.

## Decision

- **World frame:** right-handed, **+Y up**. The navigable floor plane is **XZ**.
  `SplatScene.up_axis` (default `"y"`) names the world up axis; floor movement
  is projected onto the plane perpendicular to it.
- **Camera convention:** **OpenCV**. Camera local axes: **+Z = forward**
  (into the scene), **+X = right**, **+Y = down**.
- `Pose.rotation` is the **world→camera** rotation matrix. Therefore, in world
  space:
  - camera forward = `rotation[2, :]` (third row)
  - camera right = `rotation[0, :]` (first row)
  - world-up direction = `−rotation[1, :]`
- `Pose.heading(up_axis)` returns the **horizontal** forward (forward projected
  onto the floor plane, normalised) — what `forward`/`back` translate along.

## Consequences

- ✅ `apply_action` is well-defined and unit-testable without any renderer.
- ✅ Matches the dominant convention in 3DGS / COLMAP / OpenCV ecosystems.
- ⚠️ When a concrete renderer is chosen (ADR TBD), we must verify it uses the
  same world frame and camera axes; if not, add a transform layer rather than
  changing this convention (supersede this ADR only if truly necessary).
- ⚠️ Pitch (`look_up`/`look_down`) tilts forward out of the floor plane;
  `heading` ignores pitch by construction, so translation stays level even when
  the agent is looking up/down.
