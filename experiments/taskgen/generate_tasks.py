"""Task generation prototype (ADR-0010): given an extracted scene object-graph, propose
verifiable robot tasks over the CLOSED SET of detected objects, then feasibility-gate them.

GRS-style anti-hallucination: the VLM may only reference object ids that exist in the graph;
any task naming an unknown id, or a fetch_carry whose target is fixed architecture, is dropped.
Output tasks carry goal_predicates so they are auto-checkable against the objects' 3D centres
(ADR-0010 auto-eval).

Run:  uv run python experiments/taskgen/generate_tasks.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

HERE = Path(__file__).parent
VLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-9B"
N_TASKS = 12

# Labels that are fixed architecture — valid as goto/find landmarks and as destinations,
# but NOT pickup targets for fetch_carry (you can't carry a wall).
FIXED = ("wall", "floor", "ceiling", "column", "door", "window", "partition", "staircase",
         "blinds", "panel", "grille", "tile", "frame", "desk surface", "desk leg", "monitor",
         "light fixture", "exit sign")


def movable(label: str) -> bool:
    return not any(f in label.lower() for f in FIXED)


def call_vlm(inventory: str) -> list[dict]:
    prompt = (
        "You are generating robot tasks for an indoor office, grounded in a FIXED inventory of "
        "detected objects (id, label, 3D center x,y,z). You may ONLY reference object ids that "
        "appear below — never invent objects.\n\n"
        f"INVENTORY:\n{inventory}\n\n"
        f"Propose {N_TASKS} diverse, plausible tasks. Types: 'goto' (reach an object), 'find' "
        "(locate an object), 'fetch_carry' (pick a MOVABLE object and bring it to another "
        "object's location). For fetch_carry the target must be something physically carryable "
        "(a sign, mat, banner stand, small item) — never a wall/door/window/column/staircase/"
        "ceiling. Each task: a natural-language instruction and the referenced ids.\n"
        "Output ONLY JSON: {\"tasks\":[{\"type\":\"goto|find|fetch_carry\",\"instruction\":\"...\","
        "\"target\":\"obj_XX\",\"destination\":\"obj_XX or null\"}]}"
    )
    body = {"model": MODEL, "max_tokens": 1500, "temperature": 0.4,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": "Reply with only the JSON object."},
                {"role": "user", "content": prompt}]}
    with httpx.Client(timeout=180) as c:
        txt = c.post(VLM_URL, json=body).json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    return json.loads(m.group(0)).get("tasks", []) if m else []


def heuristic_proposals(objs: dict) -> list[dict]:
    """Offline stand-in for the VLM proposer (used when the endpoint is down).

    Deliberately includes a few INFEASIBLE proposals (a hallucinated id, a fixed-architecture
    fetch target, a bad destination) so the feasibility gate is exercised, not just bypassed.
    """
    have = {o["label"]: oid for oid, o in objs.items()}

    def first(*subs: str) -> str | None:
        for oid, o in objs.items():
            if any(s in o["label"].lower() for s in subs):
                return oid
        return None

    p: list[dict] = []
    for sub, typ in [("staircase", "goto"), ("emergency exit", "find"),
                     ("glass partition", "goto"), ("structural column", "find"),
                     ("flat screen monitor", "find")]:
        oid = first(sub)
        if oid:
            p.append({"type": typ, "instruction": f"{typ} the {objs[oid]['label']}",
                      "target": oid, "destination": None})
    bs, st = first("banner stand"), first("staircase")
    if bs and st:
        p.append({"type": "fetch_carry", "target": bs, "destination": st,
                  "instruction": "Take the banner stand to the staircase."})
    mat, cab = first("blue mat"), first("cabinet")
    if mat and cab:
        p.append({"type": "fetch_carry", "target": mat, "destination": cab,
                  "instruction": "Carry the blue mat to the office cabinet."})
    # infeasible, to exercise the gate:
    p.append({"type": "goto", "target": "obj_99", "destination": None,
              "instruction": "Go to the coffee machine."})  # hallucinated id
    wall = first("office wall")
    if wall and st:
        p.append({"type": "fetch_carry", "target": wall, "destination": st,
                  "instruction": "Bring the office wall to the staircase."})  # fixed architecture
    if bs:
        p.append({"type": "fetch_carry", "target": bs, "destination": "obj_77",
                  "instruction": "Take the banner stand to the kitchen."})  # bad destination
    return p


def goal_predicates(t: dict) -> list[dict]:
    tgt = t["target"]
    if t["type"] == "find":
        return [{"pred": "seen", "args": ["agent", tgt]}]
    if t["type"] == "goto":
        return [{"pred": "at", "args": ["agent", tgt]}]
    return [{"pred": "holding", "args": ["agent", tgt]},
            {"pred": "at", "args": [tgt, t["destination"]]}]


def main() -> None:
    graph = json.loads((HERE / "scene_graph.json").read_text())
    objs = {o["id"]: o for o in graph["objects"]}
    inv = "\n".join(f"  {o['id']}: {o['label']} at "
                    f"({o['center'][0]:.1f},{o['center'][1]:.1f},{o['center'][2]:.1f})"
                    for o in graph["objects"])
    try:
        raw = call_vlm(inv)
        src = "VLM (Qwen)"
    except (httpx.ConnectError, httpx.TimeoutException, KeyError, json.JSONDecodeError):
        raw = heuristic_proposals(objs)
        src = "OFFLINE heuristic (vLLM unreachable)"
    print(f"proposer = {src}: {len(raw)} candidate tasks; gating against the inventory...\n")

    kept, dropped = [], []
    for i, t in enumerate(raw):
        typ, tgt, dst = t.get("type"), t.get("target"), t.get("destination")
        if typ not in ("goto", "find", "fetch_carry"):
            dropped.append((t, "bad type")); continue
        if tgt not in objs:
            dropped.append((t, f"target {tgt} not in inventory (hallucinated)")); continue
        if typ == "fetch_carry":
            if dst not in objs:
                dropped.append((t, f"destination {dst} not in inventory")); continue
            if not movable(objs[tgt]["label"]):
                dropped.append((t, f"target '{objs[tgt]['label']}' is fixed architecture")); continue
        kept.append({
            "task_id": f"{graph['scene_id']}/t{len(kept):02d}",
            "scene_id": graph["scene_id"], "type": typ,
            "instruction": t.get("instruction", ""),
            "target": tgt, "destination": dst if typ == "fetch_carry" else None,
            "goal_predicates": goal_predicates(t),
            "feasibility": {"target_exists": True,
                            "destination_exists": typ != "fetch_carry" or dst in objs},
        })

    print(f"=== KEPT {len(kept)} feasible tasks ===")
    for t in kept:
        d = f" -> {objs[t['destination']]['label']}" if t["destination"] else ""
        print(f"  [{t['type']:11s}] {t['instruction']}")
        print(f"               target={objs[t['target']]['label']}{d}  preds={t['goal_predicates']}")
    print(f"\n=== DROPPED {len(dropped)} (feasibility gate) ===")
    for t, why in dropped:
        print(f"  - {t.get('type','?'):11s} target={t.get('target')}: {why}")

    out = HERE / "tasks.json"
    out.write_text(json.dumps({"scene_id": graph["scene_id"], "schema_version": 1,
                               "tasks": kept}, indent=1))
    print(f"\nwrote {out} ({len(kept)} tasks)")


if __name__ == "__main__":
    main()
