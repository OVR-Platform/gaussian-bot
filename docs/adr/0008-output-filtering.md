# 0008. Output filtering (trajectories → deliverable poses)

- Status: accepted
- Date: 2026-06-19

## Context

The agent emits a pose every step, across many seeds. Most are redundant
(orbiting, clustering, cross-seed duplicates) and some are degenerate (wandered
into a bad-render edge). The deliverable must be a clean, bounded, well-spread
set of poses.

## Decision

Filter the **global union** of all walk trajectories through three stages:

1. **Quality drop** — discard poses whose render is degenerate (low mean opacity
   or depth mostly holes/NaN). Removes AABB-clamped and edge garbage.
2. **Novelty dedup** — greedy farthest-point decimation: keep a pose only if its
   floor-plane distance to every *kept* pose exceeds `r_keep` (the Tier-1
   coverage radius).
3. **Budget cap** — if more than `P` survive, keep the top-`P` by max-min
   distance; if fewer, keep all.

**Dedup space:** **position-only** (floor-plane) for v1. Angular (position +
viewing-direction) dedup is deferred — extra angular samples at a spot are
generally *helpful* for densifying 3DGS, so we do not aggressively cull them
early.

**Output:** each surviving pose as `(rotation, position)` plus lightweight
metadata (seed id, novelty score, render confidence) for reproducibility. On-disk
format (`transforms.json` / `cameras.json`, quaternion vs matrix) is decided
when the downstream 3DGS trainer is wired.

## Consequences

- ✅ Reuses the novelty machinery from coverage (ADR-0007) and termination
  (ADR-0006) — one knob (`r_keep`), consistent everywhere.
- ✅ Guarantees a fixed-size, well-spread deliverable regardless of how much the
  agent orbited.
- ⚠️ Quality drop depends on a render-confidence signal from the renderer; until
  a real renderer is wired, confidence can be supplied as a placeholder/all-ones.
- ⚠️ Position-only dedup may ship some near-duplicate viewpoints; revisit once
  Tier-2 pose-space coverage is observable.
