"""Run one generated task headless in task-mode, record the run, and auto-eval it.

A phased teacher drives the episode with ground-truth gating so demos are clean:
  - goto/find: one leg to the target; STOP exactly on arrival.
  - fetch_carry: leg 1 to the target -> GRAB, re-plan, leg 2 to the destination -> DROP -> STOP.
The VLM navigates each step (premature stops overridden with a bearing step); the teacher only
injects grab/drop/stop at the geometrically-verified moments and switches the steering cue.

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
MAX_STEPS = 90  # goto/find; fetch_carry gets more (two legs) via run_one
FETCH_STEPS = 150
REACH_GOTO = 1.0  # < eval EPS_REACH 1.2
REACH_GRASP = 0.7  # < eval EPS_GRASP 0.8


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
        if m and float(m.group(1)) <= REACH_GOTO:
            return Decision(action=Action.STOP, raw_text="oracle:arrived")
        return _bearing_action(text)


class PhasedTeacher:
    """Ground-truth gated teacher over legs; injects grab/drop/stop, VLM drives the rest.

    ``legs`` = [{"goal": (3,), "action": Action, "reach": m}]. On geometric arrival at a leg's
    goal, performs its action: STOP ends the run; GRAB/DROP advances to the next leg (re-planning
    the route to it) and the steering cue switches. A premature VLM stop is replaced with a
    bearing step so the episode only ends when the goal is really reached.
    """

    def __init__(self, inner: object, ctx: dict, legs: list[dict], replan, aim) -> None:
        self.inner = inner
        self.ctx = ctx
        self.legs = legs
        self.replan = replan
        self.aim = aim

    def reset(self) -> None:
        self.inner.reset()  # type: ignore[attr-defined]

    def describe(self, o: object) -> str:
        return self.inner.describe(o)  # type: ignore[attr-defined]

    def act(self, observation: object) -> Decision:
        ph = self.ctx["phase"]
        if ph >= len(self.legs):
            return Decision(action=Action.STOP, raw_text="done")
        leg = self.legs[ph]
        gf = floor_xy(leg["goal"], self.ctx["ua"])[0]
        dist = float(np.linalg.norm(floor_xy(self.ctx["pos"], self.ctx["ua"])[0] - gf))
        arrived = self.ctx["i"] >= len(self.ctx["waypoints"]) and dist <= leg["reach"]
        if arrived:
            if leg["action"] is Action.STOP:
                return Decision(action=Action.STOP, raw_text="arrived")
            self.ctx["phase"] = ph + 1  # grab/drop: advance to next leg
            if self.ctx["phase"] < len(self.legs):
                self.replan(self.legs[self.ctx["phase"]]["goal"])
                self.aim()
            return Decision(action=leg["action"], raw_text=f"teacher:{leg['action'].value}")
        dec = self.inner.act(observation)  # type: ignore[attr-defined]
        if dec.action in (Action.STOP, Action.GRAB, Action.DROP):
            return _bearing_action(observation.prompt)  # we own termination/manipulation
        return dec


def run_one(pick: int, *, oracle: bool = False, direct: bool = False,
            max_steps: int = MAX_STEPS) -> dict:
    """Run task ``pick``, save run_record.json, return a result summary dict."""
    tasks = json.loads((HERE / "tasks.json").read_text())["tasks"]
    task = tasks[pick]
    graph = json.loads((HERE / "scene_graph.json").read_text())
    objs = {o["id"]: o for o in graph["objects"]}
    target_c = np.asarray(objs[task["target"]]["center"])
    dest_c = np.asarray(objs[task["destination"]]["center"]) if task.get("destination") else None
    if task["type"] == "fetch_carry" and max_steps == MAX_STEPS:
        max_steps = FETCH_STEPS  # two legs (to target, then to destination) need more budget

    cfg = load_config().overrides(
        {"mode": "task", "task_prompt": task["instruction"], "use_real_vlm": True,
         "max_steps": max_steps, "aerial_survey": False, "coverage_3d": False}
    )
    explorer, seeds, coverage = build_session(cfg)
    builder = explorer.observation_builder
    ua = builder.up_axis
    means = np.asarray(explorer.renderer.cloud.means.detach().cpu())  # type: ignore[attr-defined]

    from floor_planner import plan  # noqa: PLC0415

    ctx = {"phase": 0, "waypoints": [], "i": 0, "pos": np.asarray(seeds[0].pose.position), "ua": ua}

    def replan(goal: np.ndarray) -> None:
        gf = floor_xy(goal, ua)[0]
        wps = [] if direct else plan(means, ua, floor_xy(ctx["pos"], ua)[0], gf)
        if not direct and (not wps or float(np.linalg.norm(wps[-1] - gf)) > 0.3):
            wps.append(gf)
        ctx["waypoints"] = wps
        ctx["i"] = 0

    # legs: goto/find -> reach target then STOP; fetch_carry -> GRAB at target, DROP at destination.
    if task["type"] == "fetch_carry" and dest_c is not None:
        legs = [{"goal": target_c, "action": Action.GRAB, "reach": REACH_GRASP},
                {"goal": dest_c, "action": Action.DROP, "reach": REACH_GRASP}]
    else:
        legs = [{"goal": target_c, "action": Action.STOP, "reach": REACH_GOTO}]

    def aim() -> None:
        goal = legs[min(ctx["phase"], len(legs) - 1)]["goal"]
        if ctx["i"] >= len(ctx["waypoints"]):
            builder.task_target = goal
            return
        wp = ctx["waypoints"][ctx["i"]]
        builder.task_target = np.array([wp[0], goal[1], wp[1]])

    replan(legs[0]["goal"])
    aim()

    if oracle:
        explorer.vlm = OracleNav()
    else:
        explorer.vlm = PhasedTeacher(explorer.vlm, ctx, legs, replan, aim)  # type: ignore[assignment]
    explorer.actions_per_query = 1

    run: dict = {"task_id": task["task_id"], "trajectory": [], "forwards": [],
                 "grabs": [], "drops": [], "waypoints": [w.tolist() for w in ctx["waypoints"]]}

    def sink(ev: object) -> None:
        if isinstance(ev, StepEvent):
            ctx["pos"] = np.asarray(ev.pose.position)
            run["trajectory"].append([float(x) for x in ev.pose.position])
            run["forwards"].append([float(x) for x in viewing_direction(ev.pose.rotation)])
            if ctx["i"] < len(ctx["waypoints"]):  # advance waypoint when reached
                cur = floor_xy(ev.pose.position, ua)[0]
                if float(np.linalg.norm(cur - ctx["waypoints"][ctx["i"]])) < 0.9:
                    ctx["i"] += 1
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
