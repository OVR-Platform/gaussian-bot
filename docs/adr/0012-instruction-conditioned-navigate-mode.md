# 0012. Instruction-conditioned navigate mode (VLN)

- Status: accepted
- Date: 2026-07-03

## Context

The stack implements autonomous *coverage* exploration (ADR-0003…0008) plus a
"task mode" substrate (mode `task`, `task_prompt`, simulated grab/drop,
`TaskStop`) that only ever ran inside `experiments/taskgen/` with privileged
scene-graph ground truth and hardcoded paths. There was no in-package way to
run a goal-conditioned episode: no CLI, no headless recorder, no reusable
episode GIF, no success signal beyond "the VLM said stop", and no blocking
vLLM startup for headless use. The VLN research note
(`docs/research/vln-embodied-exploration.md`) flags two ADRs a navigate mode
contradicts: ADR-0006 (the VLM's `stop` is demoted to a plateau vote) and
ADR-0003 (multi-seed restarts / no goal-directed planning).

## Decision

1. **One command**: `uv run gaussian-robot navigate <scene.ply> --instruction
   "…"` runs a single goal-conditioned walk against the real stack (gsplat
   renderer + Qwen on vLLM; `--start-vllm` spawns the server and *blocks* on
   readiness via `VLLMServerProcess.wait_ready`; `--demo-vlm` keeps a scripted
   client for smokes/tests). Outputs: `episode.json` (trajectory, forwards,
   actions, grab/drop, outcome), an `episode.gif`
   (`[robot view | top-down trail+goal]`, `gaussian_robot.episode`), and a
   printed outcome. Exit code 0 on success, 3 on failure.
2. **Instruction is first-class**: `Observation` gains a structured
   `instruction` field (set in task mode; still woven into the prompt — the
   Qwen client is unchanged). Recorders/evaluators no longer re-parse prompts.
3. **Success with provenance** (amends ADR-0006 *for this mode only*):
   - With `--target-xyz` (a goal position in the splat frame, e.g. from a
     scene graph), a `GoalReached` walk policy measures arrival
     (floor-plane distance ≤ `--goal-eps`, default 1.2 m per the
     task-pipeline survey) and ends the walk with reason `goal_reached` —
     composed BEFORE `TaskStop` so a simultaneous VLM stop reports the
     measured reason. Success source: `geometric`.
   - Without a target, the VLM's `stop` (`task_complete`) counts as success
     but is labelled `vlm_declared` — explicitly unverified.
   - With a target, a VLM stop away from the goal is a **failure**.
   Densify mode is untouched: `stop` stays a demoted plateau vote there.
4. **Single episode** (relaxes ADR-0003 for this mode): navigate = one walk
   from one validated capture-pose seed (`num_seeds=1`, as task mode already
   forced). Coverage machinery (aerial survey, 3D gaps, pose deliverable,
   ADR-0008 filtering) is bypassed — the deliverable is the
   trajectory + outcome.

## Consequences

- Given a splat and an instruction, an episode runs end-to-end headless and
  reports an honest outcome; the office-scene smoke is
  `scripts/navigate_smoke.py`.
- Instruction *grounding* is vision+prompt only: without `--target-xyz` the
  robot has no bearing hint and success is unverified VLM say-so. Real
  grounding needs the scene-graph pipeline (ADR-0010) to resolve
  "the blue mat" → coordinates; the CLI accepts those coordinates today.
- The research menu's bigger levers — MapGPT-style topological memory,
  FisherRF uncertainty-NBV, frontier scoring — are NOT in this mode yet; this
  is the loop they plug into (future ADRs).
- `experiments/taskgen/run_task.py` (GT-gated teachers, predicate eval)
  remains the experiment-grade harness; the CLI is the supported product
  path.
