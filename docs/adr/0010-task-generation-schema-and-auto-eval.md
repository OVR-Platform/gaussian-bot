# 0010. Task-generation schema & ground-truth auto-eval

- Status: proposed
- Date: 2026-06-25

## Context

Beyond densification (ADR-0007, *Goal A: plausibility*, no ground truth in the
regions we improve), a second product line: **generate robot tasks from a scene
and train/evaluate a VLM agent on them**, scaling to ~200k indoor 3DGS scenes.

This line is **Goal B (truth)** and it is *tractable here* precisely because we
manufacture the ground truth: if we extract where the objects actually are
(open-vocab 3D object instances from the posed capture photos), then a task like
"take the mug from the desk to the counter" has a *checkable* success condition —
unlike densification, where novel-view supervision is a diffusion hallucination.

This ADR fixes the **data contract** (so 200k scenes produce comparable data)
and the **automatic success rule**. It does *not* fix the perception method that
extracts the objects (that is the open research/eng question — see the parallel
survey) nor the task-generation policy; both feed this schema.

The agent already has the execution substrate: `task` mode with simulated
`grab`/`drop` (ADR-0004 extension) and `CarryEvent`. Today grab/drop are
unverified; this ADR makes them verifiable against the extracted ground truth.

## Decision

### 1. Scene object graph (ground truth), per scene

Extracted once from the posed photos (perception is pluggable, ADR-0001 spirit):

```jsonc
{
  "scene_id": "office_0421",
  "up_axis": "-y",
  "bounds": {"min": [...], "max": [...]},
  "objects": [
    {
      "id": "obj_03",
      "label": "mug",                 // open-vocab class
      "aliases": ["cup", "coffee mug"],
      "center": [x, y, z],            // world (same frame as the splat/cameras.json)
      "bbox": {"min": [...], "max": [...]},
      "support": "obj_07",            // resting-on relation (optional)
      "room": "kitchenette",          // coarse region label (optional)
      "confidence": 0.0,              // detector/lift confidence
      "n_views": 0                    // # capture frames it was seen in (reliability)
    }
  ],
  "regions": [{"id": "kitchenette", "centroid": [x, y, z], "bbox": {...}}],
  "provenance": {"perception": "<method@version>", "extracted_at": "<iso8601>"}
}
```

`center` is the single source of truth for auto-eval. `n_views`/`confidence`
let us drop unreliable instances before generating tasks.

### 2. Task spec (generated), referencing object/region ids

Goals are **predicates over the object graph** (PDDL/BDDL-style) so they are
verifiable and human-readable. v1 supports three task types:

```jsonc
{
  "task_id": "office_0421/t12",
  "scene_id": "office_0421",
  "type": "fetch_carry",            // goto | find | fetch_carry
  "instruction": "Take the mug from the desk and bring it to the kitchen counter.",
  "target": "obj_03",               // object to find/reach/pick
  "destination": {"region": "kitchenette"},  // or {"object": "obj_19"} or {"pose": [...]}
  "goal_predicates": [              // the formal, checkable goal
    {"pred": "holding", "args": ["agent", "obj_03"]},
    {"pred": "at", "args": ["obj_03", "kitchenette"]}
  ],
  "feasibility": {"reachable": true, "path_checked_with": "line_of_sight"},
  "difficulty": "medium",           // heuristic: distance, #rooms, occlusion
  "gen": {"by": "<llm@version>", "seed": 0}
}
```

`goto`/`find` use only the first predicate (`at agent target` / `seen target`);
`fetch_carry` uses both. The schema is **versioned** (`schema_version` at file
root) from day one.

### 3. Automatic success rule (ground-truth auto-eval)

A run produces a trajectory plus `mark`/`CarryEvent`s. Success is decided
**geometrically against the object graph**, with a tolerance `ε` derived from
the action step (e.g. `ε = 2·step`, the reach radius):

- **goto / find**: the agent's pose came within `ε` of `target.center` **and** a
  frame had `target` within the view frustum (grounded, not just nearby).
- **fetch_carry**:
  1. a `grab` `CarryEvent` fired within `ε` of `target.center` (right object), and
  2. a later `drop` `CarryEvent` fired within `ε` of the destination
     (region centroid / object center / pose), with `carrying` true in between.

Outcome per run: `{success: bool, failed_predicate, steps, path_len,
grab_err, drop_err}`. This is the training/eval label — and the regression
guard against the agent "declaring done" without doing the task.

### 4. Feasibility gate (task generation must pass it)

A generated task is kept only if: target (and destination object) exist in the
graph with `confidence`/`n_views` above threshold, and a coarse path exists
(reuse `_line_of_sight_clear` / coverage reachability). Infeasible tasks are
logged and dropped, never silently kept (cf. ADR-0006 "no silent caps").

## Consequences

- ✅ One versioned contract across 200k scenes → comparable, joinable data.
- ✅ Grab/drop become **verified**; the agent can no longer be rewarded for a
  bogus "done". Closes the loop for training and eval.
- ✅ Decouples the three moving parts (perception, task-gen, agent) behind a
  stable schema — each can improve independently.
- ✅ `goal_predicates` keep tasks human-readable *and* machine-checkable, and
  leave room to grow toward full PDDL/BDDL if needed.
- ⚠️ Auto-eval is only as good as the extracted `center`s: perception error
  feeds straight into label noise. Track `confidence`/`n_views` and consider a
  human-audited gold subset.
- ⚠️ `ε` is a blunt proxy for "did the right thing"; a too-large `ε` passes near
  misses. Tune against the gold subset.
- ⚠️ Symbolic pick/place only (no grasp physics); a task needing real
  manipulation is out of scope until the action set is extended (ADR-0004).
```
