"""Build the demo dataset: run every task, save the FULL trajectory + task + outcome per task.

Each demos/<task_id>.json is one labelled episode (instruction, goal predicates, target/dest
centres, the trajectory, grab/drop events, planned waypoints, and the auto-eval result) — the
unit a VLM would train on. A manifest.json lists them with success flags.

Run:  uv run python experiments/taskgen/make_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

from run_task import run_one

HERE = Path(__file__).parent
DEMOS = HERE / "demos"


def main() -> None:
    DEMOS.mkdir(exist_ok=True)
    tasks = json.loads((HERE / "tasks.json").read_text())["tasks"]
    objs = {o["id"]: o for o in json.loads((HERE / "scene_graph.json").read_text())["objects"]}
    manifest = []
    for i, t in enumerate(tasks):
        print(f"[{i + 1}/{len(tasks)}] {t['task_id']} ({t['type']})", flush=True)
        res = run_one(i)
        run = json.loads((HERE / "run_record.json").read_text())  # full trajectory just written
        demo = {
            "task_id": t["task_id"], "type": t["type"], "instruction": t["instruction"],
            "goal_predicates": t["goal_predicates"],
            "target_center": objs[t["target"]]["center"],
            "destination_center": objs[t["destination"]]["center"] if t.get("destination") else None,
            "result": res, "run": run,
        }
        (DEMOS / f"{t['task_id'].replace('/', '_')}.json").write_text(json.dumps(demo, indent=1))
        manifest.append({"task_id": t["task_id"], "type": t["type"], "success": res["success"],
                         "instruction": t["instruction"], "steps": res["steps"]})
        print(f"   success={res['success']} steps={res['steps']}", flush=True)
        (DEMOS / "manifest.json").write_text(json.dumps({"demos": manifest}, indent=1))

    ok = sum(m["success"] for m in manifest)
    print(f"\ndataset: {ok}/{len(manifest)} successful demos in {DEMOS}", flush=True)


if __name__ == "__main__":
    main()
