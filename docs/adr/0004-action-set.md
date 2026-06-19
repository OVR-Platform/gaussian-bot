# 0004. Action set

- Status: accepted
- Date: 2026-06-19

## Context

The local controller (ADR-0003) needs a discrete action vocabulary the VLM can
emit reliably each step. A 9B VLM is good at qualitative direction ("turn
toward the door") but bad at metric magnitudes ("move 0.37 m").

## Decision

- **Verbs** (egocentric frame): `forward`, `back`, `turn_left`, `turn_right`,
  `look_up`, `look_down`, `stop`.
- **Magnitudes are system-owned, not VLM-owned:**
  - `step = 0.03 × AABB diagonal` (auto-scales room vs object scenes).
  - `Δrot = 30°` for turn and pitch.
  - The VLM emits only the verb; the executor applies the magnitude.
- **Translation stays on the floor plane** (perpendicular to `up_axis`); pitch
  is view-only and does not change floor position.
- **One action per step** (no action chunking); every action gets its own render.
- `stop` is a meta-action; its semantics are defined in ADR-0006 (demoted to a
  vote, does not end a walk alone).
- **Emission format:** JSON `{"action": "<verb>"}`. No `reason` field —
  Qwen3.5-9B runs in thinking mode, so `<think>` already captures rationale.

Deferred (cheap upgrades): multi-level magnitudes, vertical motion
(`rise`/`lower`), `strafe`, action chunking.

## Consequences

- ✅ Tiny verb set → reliable JSON parsing from a 9B model.
- ✅ VLM owns direction (qualitative), system owns magnitude (quantitative) —
  each does what it is good at.
- ✅ Floor-plane constraint keeps poses on a navigable surface; trivially
  reverted per-scene if vertical motion is added.
- ⚠️ Single magnitude may over/under-step for some scenes; tunable via the
  `ActionSpace` value, and multi-level is a one-field upgrade.
