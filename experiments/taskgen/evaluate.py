"""Auto-eval (ADR-0010): score a task-mode run against the scene ground truth.

Checks each task's goal_predicates geometrically against the extracted object centres:

  at(agent, X)      -> the trajectory passed within EPS_REACH (floor dist) of X
  seen(agent, X)    -> X fell within the camera frustum (in front, within EPS_SEEN) at some step
  holding(agent, X) -> a `grab` event fired within EPS_GRASP (floor dist) of X
  at(X, DEST)       -> a `drop` event fired within EPS_GRASP of DEST, after the grab

Outcome per task: binary success (all predicates) + partial credit (fraction satisfied),
mirroring ALFRED's Task-Success / Goal-Condition-Success. Tolerances are coarse (navigation /
symbolic pick-place at scene scale), tunable, and reported so near-misses are visible.

Run the built-in self-test:  uv run python experiments/taskgen/evaluate.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from gaussian_robot.metrics.coverage import floor_xy

HERE = Path(__file__).parent
EPS_REACH = 1.2   # m, floor distance for at(agent, X) / reaching a landmark
EPS_GRASP = 0.8   # m, floor distance for grab / drop placement
EPS_SEEN = 5.0    # m, max range to count an object as "seen"
SEEN_HALF_FOV = math.radians(40)  # in front of the camera


def _floor(center: list[float], up_axis: str) -> np.ndarray:
    return floor_xy(np.asarray(center, dtype=np.float64), up_axis)[0]


def _check(pred: dict, objs: dict, run: dict, up_axis: str) -> tuple[bool, float]:
    """Return (satisfied, best_error_meters) for one goal predicate."""
    name, args = pred["pred"], pred["args"]
    traj = np.asarray(run.get("trajectory", []), dtype=np.float64)  # (T,3) eye positions
    grabs = np.asarray(run.get("grabs", []), dtype=np.float64).reshape(-1, 2)
    drops = np.asarray(run.get("drops", []), dtype=np.float64).reshape(-1, 2)

    if name == "at" and args[0] == "agent":  # reached object
        tf = _floor(objs[args[1]]["center"], up_axis)
        if traj.size == 0:
            return False, float("inf")
        d = np.min(np.linalg.norm(floor_xy(traj, up_axis) - tf, axis=1))
        return d <= EPS_REACH, float(d)

    if name == "seen":
        tc = np.asarray(objs[args[1]]["center"], dtype=np.float64)
        best = float("inf")
        for p, f in zip(run.get("trajectory", []), run.get("forwards", []), strict=False):
            v = tc - np.asarray(p)
            dist = float(np.linalg.norm(v))
            if dist < 1e-6:
                return True, 0.0
            ang = math.acos(float(np.clip(np.dot(v / dist, np.asarray(f)), -1, 1)))
            if ang <= SEEN_HALF_FOV and dist <= EPS_SEEN:
                return True, dist
            best = min(best, dist)
        return False, best

    if name == "holding":  # grab near the target
        tf = _floor(objs[args[1]]["center"], up_axis)
        if grabs.size == 0:
            return False, float("inf")
        d = float(np.min(np.linalg.norm(grabs - tf, axis=1)))
        return d <= EPS_GRASP, d

    if name == "at":  # at(X, DEST): drop near the destination object
        df = _floor(objs[args[1]]["center"], up_axis)
        if drops.size == 0:
            return False, float("inf")
        d = float(np.min(np.linalg.norm(drops - df, axis=1)))
        return d <= EPS_GRASP, d

    return False, float("inf")


def evaluate_task(task: dict, objs: dict, run: dict, up_axis: str) -> dict:
    results = []
    for pred in task["goal_predicates"]:
        ok, err = _check(pred, objs, run, up_axis)
        results.append({"pred": pred, "satisfied": ok, "error_m": round(err, 2)})
    n_ok = sum(r["satisfied"] for r in results)
    return {"task_id": task["task_id"], "type": task["type"],
            "success": n_ok == len(results), "goal_condition_success": n_ok / len(results),
            "predicates": results}


def _self_test() -> None:
    objs = {"obj_A": {"center": [2.0, -1.0, 2.0]}, "obj_B": {"center": [-3.0, -1.0, 4.0]}}
    fetch = {"task_id": "t/fetch", "type": "fetch_carry",
             "goal_predicates": [{"pred": "holding", "args": ["agent", "obj_A"]},
                                 {"pred": "at", "args": ["obj_A", "obj_B"]}]}
    good = {"trajectory": [[2.0, -1.0, 2.0], [-3.0, -1.0, 4.0]],
            "grabs": [[2.1, 2.0]], "drops": [[-2.9, 4.1]]}  # grab@A, drop@B
    bad = {"trajectory": [[0, -1, 0]], "grabs": [[0.0, 0.0]], "drops": []}  # grabbed nothing, no drop
    g = evaluate_task(fetch, objs, good, "-y")
    b = evaluate_task(fetch, objs, bad, "-y")
    assert g["success"] and g["goal_condition_success"] == 1.0, g
    assert not b["success"] and b["goal_condition_success"] == 0.0, b
    goto = {"task_id": "t/goto", "type": "goto",
            "goal_predicates": [{"pred": "at", "args": ["agent", "obj_A"]}]}
    near = evaluate_task(goto, objs, {"trajectory": [[2.5, -1.0, 2.3]]}, "-y")  # within EPS_REACH
    far = evaluate_task(goto, objs, {"trajectory": [[8.0, -1.0, 8.0]]}, "-y")
    assert near["success"] and not far["success"], (near, far)
    print("self-test OK: fetch good/bad and goto near/far scored correctly")
    print("  good fetch:", g)
    print("  far goto:  ", far)


def main() -> None:
    tasks_p, graph_p = HERE / "tasks.json", HERE / "scene_graph.json"
    run_p = HERE / "run_record.json"
    if not run_p.exists():
        print("no run_record.json — running self-test instead.\n")
        _self_test()
        return
    graph = json.loads(graph_p.read_text())
    objs = {o["id"]: o for o in graph["objects"]}
    tasks = {t["task_id"]: t for t in json.loads(tasks_p.read_text())["tasks"]}
    run = json.loads(run_p.read_text())
    task = tasks[run["task_id"]]
    res = evaluate_task(task, objs, run, graph["up_axis"])
    print(f"task {task['task_id']} ({task['type']}): {task.get('instruction','')}")
    print(f"  SUCCESS={res['success']}  goal-condition-success={res['goal_condition_success']:.2f}")
    for r in res["predicates"]:
        print(f"   [{'OK ' if r['satisfied'] else 'XX '}] {r['pred']['pred']}"
              f"{r['pred']['args']}  err={r['error_m']}m")


if __name__ == "__main__":
    main()
