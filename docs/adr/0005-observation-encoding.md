# 0005. Observation encoding

- Status: accepted
- Date: 2026-06-19

## Context

"Where would you go to explore?" requires knowing where coverage is *missing*.
A single forward RGB view shows what is there, not what is unknown; a coordinate
list is unreadable by a VLM. Coverage must be **drawn** as an image.

## Decision

Send the VLM a **3-panel message** each step plus a fixed task prompt:

1. **RGB** — forward render at the current pose.
2. **Depth** (colormapped) — we have accurate depth; de-risks `forward` and
   signals reconstruction quality.
3. **Top-down map**, **body-fixed frame** (the agent's current forward = image
   up; the map rotates with the agent each step):
   - **blue dots** = all sampled poses (training + this session),
   - **green polyline** = the current walk's trail,
   - **red arrow** = current pose + heading.

Plus a **fixed text prompt** describing the panels and action vocabulary and the
policy ("steer toward large empty regions; avoid your green trail; emit `stop`
if your surroundings are well-covered"), plus one live state line (step index,
remaining budget, scene scale: "one step ≈ 3% of the map").

## Consequences

- ✅ The VLM *sees* coverage directly, and the body-fixed frame removes the
  hardest cognitive load (mental rotation): "empty is to my left → `turn_left`."
- ✅ Green trail makes orbiting visible, attacking the stuck problem.
- ✅ Justified by ADR-0003's division of labour: global coverage is handled by
  seeds, so the map only needs to support local steering — exactly what a
  body-fixed map does. (A global-fixed map is the alternative; deferred.)
- ⚠️ We lose global layout memory across steps; a 9B cannot maintain that well
  anyway, and seeds cover it.
- ⚠️ Cost: 3 images/step × many steps × seeds = many VLM calls; mitigated by
  Qwen3.5-9B's 262K context and capping image resolution.
- ⚠️ **Key R&D risk:** if the VLM cannot correlate the RGB view with even a
  body-fixed map, the local-controller thesis wobbles. Must be tested early.
