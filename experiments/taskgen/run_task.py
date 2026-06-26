"""Run one generated task headless in task-mode, record the run, and auto-eval it.

Closes the loop end-to-end on the live system: picks a task from tasks.json, drives a real
task-mode walk (the VLM navigates by vision), captures the trajectory + grab/drop events into
run_record.json, then scores it with evaluate.py against the scene ground truth.

Run:  uv run python experiments/taskgen/run_task.py [task_index]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from gaussian_robot.config import load_config
from gaussian_robot.events import CarryEvent, StepEvent
from gaussian_robot.metrics.coverage import viewing_direction
from gaussian_robot.nav.action import Action
from gaussian_robot.session import build_session
from gaussian_robot.vlm.client import Decision

HERE = Path(__file__).parent
MAX_STEPS = 45


class OracleNav:
    """Scripted teacher policy: follows the privileged TARGET bearing in [state] deterministically.

    Generates guaranteed-successful goto demos to imitate — the trained VLM (which won't have the
    bearing) learns from these, rather than from the untuned base model that ignores the hint.
    Parses the TARGET cue the observation already exposes, so the full walk loop (terrain-follow,
    wall-capping, events) is reused unchanged.
    """

    def reset(self) -> None:
        pass

    def describe(self, observation: object) -> str:
        return "oracle"

    def act(self, observation: object) -> Decision:
        text = observation.prompt  # type: ignore[attr-defined]
        if "BLOCKED" in text:
            return Decision(action=Action.TURN_LEFT, raw_text="oracle:escape")
        m = re.search(r"TARGET (?:(\d+)° (left|right), )?([\d.]+)m", text)
        if not m:
            return Decision(action=Action.FORWARD, raw_text="oracle:noinfo")
        ang = int(m.group(1)) if m.group(1) else 0
        side, dist = m.group(2), float(m.group(3))
        if dist <= 1.0:
            return Decision(action=Action.STOP, raw_text="oracle:arrived")
        if ang > 25:
            return Decision(action=Action.TURN_LEFT if side == "left" else Action.TURN_RIGHT,
                            raw_text="oracle:turn")
        return Decision(action=Action.FORWARD, raw_text="oracle:advance")


def main() -> None:
    tasks = json.loads((HERE / "tasks.json").read_text())["tasks"]
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    pick = int(nums[0]) if nums else next(i for i, t in enumerate(tasks) if t["type"] == "goto")
    task = tasks[pick]
    print(f"running task {task['task_id']} [{task['type']}]: {task['instruction']}\n", flush=True)

    cfg = load_config().overrides(
        {"mode": "task", "task_prompt": task["instruction"], "use_real_vlm": True,
         "max_steps": MAX_STEPS, "aerial_survey": False, "coverage_3d": False}
    )
    explorer, seeds, coverage = build_session(cfg)

    # Privileged ground-truth steering hint: feed the target's known 3D centre (from the scene
    # graph) so [state] reports a bearing to it. Used to generate successful demos / as curriculum;
    # the trained VLM won't have this — it learns from the demos.
    graph = json.loads((HERE / "scene_graph.json").read_text())
    objs = {o["id"]: o for o in graph["objects"]}
    explorer.observation_builder.task_target = np.asarray(objs[task["target"]]["center"])

    if "--oracle" in sys.argv:
        explorer.vlm = OracleNav()  # scripted teacher that follows the bearing
        explorer.actions_per_query = 1  # re-decide every step
        print("[oracle teacher policy]\n", flush=True)

    run: dict = {"task_id": task["task_id"], "trajectory": [], "forwards": [],
                 "grabs": [], "drops": []}

    def sink(ev: object) -> None:
        if isinstance(ev, StepEvent):
            run["trajectory"].append([float(x) for x in ev.pose.position])
            run["forwards"].append([float(x) for x in viewing_direction(ev.pose.rotation)])
        elif isinstance(ev, CarryEvent):
            (run["grabs"] if ev.kind == "grab" else run["drops"]).append(
                [float(x) for x in ev.floor]
            )

    explorer.event_sink = sink
    result = explorer.run_walk(seeds[0].pose, coverage, walk_id="task0")
    print(f"\nwalk ended: {result.stop_reason} after {len(result.steps)} steps; "
          f"{len(run['trajectory'])} poses, {len(run['grabs'])} grabs, {len(run['drops'])} drops",
          flush=True)

    (HERE / "run_record.json").write_text(json.dumps(run, indent=1))

    # score it
    from evaluate import evaluate_task  # noqa: PLC0415

    res = evaluate_task(task, objs, run, graph["up_axis"])
    tgt = objs[task["target"]]
    end = np.asarray(run["trajectory"][-1]) if run["trajectory"] else None
    print("\n=== AUTO-EVAL ===")
    print(f"target '{tgt['label']}' center={tgt['center']}")
    if end is not None:
        print(f"final pose={[round(float(x),2) for x in end]}")
    print(f"SUCCESS={res['success']}  goal-condition-success={res['goal_condition_success']:.2f}")
    for r in res["predicates"]:
        print(f"  [{'OK ' if r['satisfied'] else 'XX '}] {r['pred']['pred']}{r['pred']['args']}"
              f"  err={r['error_m']}m")


if __name__ == "__main__":
    main()
