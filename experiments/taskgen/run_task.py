"""Run one generated task headless in task-mode, record the run, and auto-eval it.

Closes the loop end-to-end: drives a real task-mode walk (the VLM navigates by vision, steered
by a privileged TARGET bearing that follows an A* path around obstacles), captures the run, and
scores it against the scene ground truth. Stop-on-arrival ends the walk when the goal is reached
so demos are clean. Importable as run_one() for batch generation.

Run:  uv run python experiments/taskgen/run_task.py [task_index] [--oracle] [--direct]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from gaussian_robot.config import load_config
from gaussian_robot.events import CarryEvent, StepEvent
from gaussian_robot.metrics.coverage import floor_xy, viewing_direction
from gaussian_robot.nav.action import Action
from gaussian_robot.session import build_session
from gaussian_robot.vlm.client import Decision

HERE = Path(__file__).parent
MAX_STEPS = 80
STOP_REACH = 1.0  # m: auto-stop when within this floor distance of the final target (< eval 1.2)


def _bearing_action(prompt: str) -> Decision:
    """Deterministic step toward the TARGET cue (turn to face it, then forward)."""
    if "BLOCKED" in prompt:
        return Decision(action=Action.TURN_LEFT, raw_text="bearing:escape")
    m = re.search(r"TARGET (?:(\d+)° (left|right), )?([\d.]+)m", prompt)
    if not m:
        return Decision(action=Action.FORWARD, raw_text="bearing:noinfo")
    ang = int(m.group(1)) if m.group(1) else 0
    side = m.group(2)
    if ang > 25:
        return Decision(action=Action.TURN_LEFT if side == "left" else Action.TURN_RIGHT,
                        raw_text="bearing:turn")
    return Decision(action=Action.FORWARD, raw_text="bearing:advance")


class OracleNav:
    """Scripted teacher: follows the privileged TARGET bearing in [state] deterministically."""

    def reset(self) -> None:
        pass

    def describe(self, observation: object) -> str:
        return "oracle"

    def act(self, observation: object) -> Decision:
        text = observation.prompt  # type: ignore[attr-defined]
        m = re.search(r"TARGET .*?([\d.]+)m", text)
        if m and float(m.group(1)) <= STOP_REACH:
            return Decision(action=Action.STOP, raw_text="oracle:arrived")
        return _bearing_action(text)


class StopWhenArrived:
    """Wrap the VLM with ground-truth termination gating, for clean demos (goto/find).

    We know the target's true position, so: force STOP exactly when within reach (no overshoot),
    and OVERRIDE a premature VLM ``stop`` (it often declares "arrived" early) with a bearing step
    so the episode runs until the goal is really reached. The VLM still drives every other step.
    """

    def __init__(self, inner: object, is_final, *, reach: float, enabled: bool) -> None:
        self.inner = inner
        self.is_final = is_final
        self.reach = reach
        self.enabled = enabled

    def reset(self) -> None:
        self.inner.reset()  # type: ignore[attr-defined]

    def describe(self, o: object) -> str:
        return self.inner.describe(o)  # type: ignore[attr-defined]

    def act(self, observation: object) -> Decision:
        prompt = observation.prompt  # type: ignore[attr-defined]
        if not self.enabled:
            return self.inner.act(observation)  # type: ignore[attr-defined]
        m = re.search(r"TARGET .*?([\d.]+)m", prompt)
        arrived = self.is_final() and m is not None and float(m.group(1)) <= self.reach
        if arrived:
            return Decision(action=Action.STOP, raw_text="arrived")
        dec = self.inner.act(observation)  # type: ignore[attr-defined]
        if dec.action is Action.STOP:  # VLM quit early — not arrived yet, keep going
            return _bearing_action(prompt)
        return dec


def run_one(pick: int, *, oracle: bool = False, direct: bool = False,
            max_steps: int = MAX_STEPS) -> dict:
    """Run task ``pick``, save run_record.json, return a result summary dict."""
    tasks = json.loads((HERE / "tasks.json").read_text())["tasks"]
    task = tasks[pick]
    graph = json.loads((HERE / "scene_graph.json").read_text())
    objs = {o["id"]: o for o in graph["objects"]}
    target_c = np.asarray(objs[task["target"]]["center"])

    cfg = load_config().overrides(
        {"mode": "task", "task_prompt": task["instruction"], "use_real_vlm": True,
         "max_steps": max_steps, "aerial_survey": False, "coverage_3d": False}
    )
    explorer, seeds, coverage = build_session(cfg)
    builder = explorer.observation_builder
    builder.task_target = target_c
    ua = builder.up_axis

    waypoints: list[np.ndarray] = []
    if not direct:
        from floor_planner import plan  # noqa: PLC0415

        means = np.asarray(explorer.renderer.cloud.means.detach().cpu())  # type: ignore[attr-defined]
        tgt_floor = floor_xy(target_c, ua)[0]
        waypoints = plan(means, ua, floor_xy(seeds[0].pose.position, ua)[0], tgt_floor)
        if waypoints and float(np.linalg.norm(waypoints[-1] - tgt_floor)) > 0.3:
            waypoints.append(tgt_floor)  # final approach aims at the object, not the A* cell

    wp_state = {"i": 0}

    def is_final() -> bool:
        return not waypoints or wp_state["i"] >= len(waypoints)

    if oracle:
        explorer.vlm = OracleNav()
    else:
        explorer.vlm = StopWhenArrived(  # type: ignore[assignment]
            explorer.vlm, is_final, reach=STOP_REACH, enabled=task["type"] in ("goto", "find")
        )
    explorer.actions_per_query = 1  # re-decide each step so stop-on-arrival fires promptly

    run: dict = {"task_id": task["task_id"], "trajectory": [], "forwards": [],
                 "grabs": [], "drops": [], "waypoints": [w.tolist() for w in waypoints]}

    def aim() -> None:
        if is_final():
            builder.task_target = target_c
            return
        wp = waypoints[wp_state["i"]]
        builder.task_target = np.array([wp[0], target_c[1], wp[1]])

    aim()

    def sink(ev: object) -> None:
        if isinstance(ev, StepEvent):
            run["trajectory"].append([float(x) for x in ev.pose.position])
            run["forwards"].append([float(x) for x in viewing_direction(ev.pose.rotation)])
            if waypoints and wp_state["i"] < len(waypoints):
                cur = floor_xy(ev.pose.position, ua)[0]
                if float(np.linalg.norm(cur - waypoints[wp_state["i"]])) < 0.9:
                    wp_state["i"] += 1
                aim()
        elif isinstance(ev, CarryEvent):
            (run["grabs"] if ev.kind == "grab" else run["drops"]).append(
                [float(x) for x in ev.floor]
            )

    explorer.event_sink = sink
    result = explorer.run_walk(seeds[0].pose, coverage, walk_id="task0")
    (HERE / "run_record.json").write_text(json.dumps(run, indent=1))

    from evaluate import evaluate_task  # noqa: PLC0415

    res = evaluate_task(task, objs, run, graph["up_axis"])
    return {"task_id": task["task_id"], "type": task["type"], "instruction": task["instruction"],
            "success": bool(res["success"]), "gcs": round(res["goal_condition_success"], 2),
            "stop_reason": result.stop_reason, "steps": len(result.steps),
            "errors_m": [r["error_m"] for r in res["predicates"]]}


def main() -> None:
    tasks = json.loads((HERE / "tasks.json").read_text())["tasks"]
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    pick = int(nums[0]) if nums else next(i for i, t in enumerate(tasks) if t["type"] == "goto")
    print(f"running task {pick}: {tasks[pick]['instruction']}", flush=True)
    out = run_one(pick, oracle="--oracle" in sys.argv, direct="--direct" in sys.argv)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
