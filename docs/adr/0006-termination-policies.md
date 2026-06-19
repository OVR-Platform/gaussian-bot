# 0006. Termination policies

- Status: accepted
- Date: 2026-06-19

## Context

We need to know when a walk (one local episode) and a session (the whole
exploration) end. The VLM's own `stop` action is an unreliable terminator — a 9B
model may declare "done" prematurely. We want multiple composable stop signals
and an objective (measured) primary signal rather than a self-assessed one.

## Decision

Implement each stop signal as a `StopPolicy`; compose many with OR (any firing
⇒ stop). Two levels:

**Per-walk** (ends one walk):
- **Step budget** (`N=40`): safety net, guarantees termination.
- **Coverage plateau** *(primary, objective)*: novelty < `δ` for `K=5`
  consecutive steps, where novelty = min floor-plane distance to any sampled
  pose and `δ` = one step length.
- **Bounds / degenerate render**: pose clamped to AABB or render mostly empty.
- **VLM `stop` is demoted:** emitting `stop` adds `+1` to the plateau counter
  rather than ending the walk. It can *hasten* a plateau-driven stop but cannot
  kill a walk that is still finding novelty.

**Session** (ends exploration):
- **Coverage target** *(primary)*: ≥ `τ=80%` of navigable floor cells sampled
  within radius `r`.
- **Pose budget** `P=200`: deliverable size reached.
- **Seed exhaustion**: launched from all chosen seeds.
- **Diminishing returns**: last seed batch each added `< ε` coverage.

All numeric thresholds are tunable knobs, calibrated once Tier-1 metrics are live.

## Consequences

- ✅ The user's stated fear ("says done after minor degradation") is resolved:
  measured novelty, not VLM self-assessment, terminates walks.
- ✅ OR-composition makes adding/removing policies trivial and testable in
  isolation.
- ✅ Multi-seed bounds the cost of any single over-eager stop to one walk.
- ⚠️ Per-walk plateau and session coverage both depend on the novelty/coverage
  definitions (ADR-0007), so they must share `CoverageState`.
