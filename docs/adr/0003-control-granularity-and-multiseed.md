# 0003. Control granularity: local controller + multi-seed restarts

- Status: accepted
- Date: 2026-06-19

## Context

The agent must cover a whole scene, but a VLM emitting one local action per step
(forward/turn) from a single seed can only explore a small bubble and tends to
orbit. We considered (A) pure local control, (B) a global waypoint planner,
(C) local control + multi-seed restarts, (D) local control + a frontier-teleport
action.

## Decision

Adopt **C**: a local incremental controller (one discrete egocentric action per
step) for steering, combined with **multiple short walks launched from spread-out
training poses** for global coverage. The session output is the union of all
walk trajectories. Global waypoint planning (B) and frontier-teleport (D) are
deferred.

## Consequences

- ✅ A 9B VLM reasons well egocentrically ("turn toward the open doorway") but
  poorly at metric coordinate arithmetic, so local discrete actions are what it
  emits reliably; B is ruled out as the primary mode.
- ✅ Every training camera is a valid, well-rendered launch point, so multi-seed
  buys global coverage without inventing a teleport semantics.
- ✅ Clean division of labour: seeds handle global coverage; the body-fixed map
  (ADR-0005) handles local steering.
- ⚠️ Requires a reasonable set of seeds; trivially obtained from the training
  cameras, but very few seeds → under-coverage.
- ⚠️ D (frontier-teleport) remains a v2 upgrade if seeds alone leave gaps that
  Tier-1/2 metrics expose.
