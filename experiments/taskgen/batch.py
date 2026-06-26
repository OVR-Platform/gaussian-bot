"""Batch: run every generated task through the planner+VLM loop and report the success rate.

Produces the first labelled trajectory dataset on this scene: per-task success / goal-condition
score / stop reason, plus an aggregate by task type. Writes batch_results.json incrementally so
progress survives interruption.

Run:  uv run python experiments/taskgen/batch.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from run_task import run_one

HERE = Path(__file__).parent


def main() -> None:
    tasks = json.loads((HERE / "tasks.json").read_text())["tasks"]
    out = HERE / "batch_results.json"
    results: list[dict] = []
    for i, t in enumerate(tasks):
        print(f"\n===== [{i + 1}/{len(tasks)}] {t['task_id']} ({t['type']}): {t['instruction']}",
              flush=True)
        try:
            r = run_one(i)
        except Exception as exc:  # noqa: BLE001  keep the batch going
            print("  ERROR:", exc, flush=True)
            traceback.print_exc()
            r = {"task_id": t["task_id"], "type": t["type"], "success": False,
                 "gcs": 0.0, "stop_reason": f"error:{type(exc).__name__}", "steps": 0,
                 "errors_m": []}
        results.append(r)
        print(f"  -> success={r['success']} gcs={r['gcs']} stop={r['stop_reason']} "
              f"steps={r['steps']} err={r['errors_m']}", flush=True)
        out.write_text(json.dumps({"results": results}, indent=1))  # incremental save

    print("\n================ SUMMARY ================", flush=True)
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)
    for typ, rs in sorted(by_type.items()):
        ok = sum(r["success"] for r in rs)
        gcs = sum(r["gcs"] for r in rs) / len(rs)
        print(f"  {typ:11s}: {ok}/{len(rs)} success   mean goal-condition={gcs:.2f}", flush=True)
    total_ok = sum(r["success"] for r in results)
    print(f"  TOTAL      : {total_ok}/{len(results)} success", flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
