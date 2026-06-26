"""Perception prototype: extract a 3D object inventory from the splat + VLM, and
quantify how badly glass/reflective surfaces corrupt it.

Reuses our own stack (gsplat renderer + Qwen via vLLM), no GroundingDINO/SAM needed:
  render perspective views from the splat at COLMAP poses (RGB + aligned depth)
  -> Qwen lists objects with a rough 2D location + a surface flag (solid/glass/reflective)
  -> lift each detection to 3D via the splat depth + pose
  -> greedy cluster across views into object instances
  -> report a glass-vs-solid reliability diagnostic.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter

import httpx
import numpy as np

from gaussian_robot.backends.gsplat_renderer import GsplatRenderer
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose
from gaussian_robot.splat.capture_poses import discover_capture_poses, load_capture_poses
from gaussian_robot.vlm.qwen import jpeg_data_url

PLY = "/mnt/archive/datasets/ufficio360-35a39133-e1f2-4426-86d4-a3d7a00614ee-PIC/gaussian_pointcloud_30000_original.ply"
DEVICE = "cuda:1"
VLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-9B"
N_VIEWS = 16
W = H = 512
FOV_DEG = 75.0
FX = FY = (W / 2) / math.tan(math.radians(FOV_DEG) / 2)
INTR = CameraIntrinsics(fx=FX, fy=FY, cx=W / 2, cy=H / 2, width=W, height=H)

PROMPT = (
    "This is a render from a 3D Gaussian-Splat reconstruction of an indoor office. "
    "List the distinct physical objects/structures you can clearly see. For each, give a "
    "rough CENTER as normalized image coords cx,cy in [0,1] (origin top-left), and classify "
    "its surface as one of: solid, glass, reflective. Reply ONLY with JSON: "
    '{"objects":[{"label":"...","cx":0.0,"cy":0.0,"surface":"solid|glass|reflective"}]}'
)


def call_vlm(rgb: np.ndarray) -> list[dict]:
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": jpeg_data_url(rgb)}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "max_tokens": 700,
        "temperature": 0.2,
    }
    with httpx.Client(timeout=120) as c:
        r = c.post(VLM_URL, json=body)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group(0)).get("objects", [])
    except json.JSONDecodeError:
        return []


def patch_depth(depth: np.ndarray, u: int, v: int, k: int = 7) -> tuple[float, float]:
    """Median finite depth and its coeff-of-variation in a (2k+1)^2 patch -> (median, cov, frac_finite)."""
    sub = depth[max(0, v - k) : v + k + 1, max(0, u - k) : u + k + 1]
    finite = sub[np.isfinite(sub)]
    frac = finite.size / max(1, sub.size)
    if finite.size == 0:
        return float("nan"), float("nan"), 0.0
    med = float(np.median(finite))
    cov = float(np.std(finite) / (abs(med) + 1e-6))
    return med, cov, frac


def unproject(u: int, v: int, d: float, pose: Pose) -> np.ndarray:
    cam = np.array([(u - INTR.cx) / FX * d, (v - INTR.cy) / FY * d, d], dtype=np.float64)
    return pose.position + pose.rotation.T @ cam  # world->cam rows -> transpose to lift


def main() -> None:
    print(f"loading splat on {DEVICE} ...", flush=True)
    renderer = GsplatRenderer.from_path(PLY, device=DEVICE)
    poses = load_capture_poses(discover_capture_poses(PLY))
    idx = np.linspace(0, len(poses) - 1, N_VIEWS).astype(int)
    print(f"{len(poses)} poses, sampling {len(idx)}", flush=True)

    dets: list[dict] = []  # each: world point + surface + label + reliability
    by_surface: dict[str, list[dict]] = {}
    for n, i in enumerate(idx):
        pose = poses[int(i)]
        res = renderer.render(Camera(pose=pose, intrinsics=INTR))
        finite_frac_view = float(np.isfinite(res.depth).mean())
        objs = call_vlm(res.rgb)
        kept = 0
        for o in objs:
            try:
                cx, cy = float(o["cx"]), float(o["cy"])
                surf = str(o.get("surface", "solid")).lower()
                label = str(o.get("label", "?"))[:40]
            except (KeyError, TypeError, ValueError):
                continue
            u, v = int(np.clip(cx, 0, 1) * (W - 1)), int(np.clip(cy, 0, 1) * (H - 1))
            med, cov, frac = patch_depth(res.depth, u, v)
            rec = {"label": label, "surface": surf, "depth": med, "cov": cov,
                   "frac_finite": frac, "liftable": math.isfinite(med)}
            by_surface.setdefault(surf, []).append(rec)
            if math.isfinite(med):
                rec["xyz"] = unproject(u, v, med, pose).tolist()
                dets.append(rec)
                kept += 1
        print(f"  view {n:2d} (pose {i:3d}): depth_finite={finite_frac_view:.2f} "
              f"objs={len(objs)} lifted={kept}", flush=True)

    # greedy 3D clustering into instances
    R = 0.5
    inst: list[dict] = []
    for d in dets:
        p = np.array(d["xyz"])
        hit = None
        for c in inst:
            if np.linalg.norm(p - np.array(c["center"])) < R:
                hit = c
                break
        if hit:
            hit["pts"].append(p)
            hit["labels"].append(d["label"])
            hit["center"] = np.mean(hit["pts"], axis=0).tolist()
            hit["n_views"] += 1
        else:
            inst.append({"center": p.tolist(), "pts": [p], "labels": [d["label"]],
                         "surface": d["surface"], "n_views": 1})

    print("\n========== GLASS-vs-SOLID DIAGNOSTIC ==========")
    for surf in ("solid", "glass", "reflective"):
        recs = by_surface.get(surf, [])
        if not recs:
            print(f"{surf:11s}: (none detected)")
            continue
        liftable = sum(r["liftable"] for r in recs)
        covs = [r["cov"] for r in recs if math.isfinite(r["cov"])]
        fracs = [r["frac_finite"] for r in recs]
        print(f"{surf:11s}: detections={len(recs):3d}  liftable(finite depth)={liftable}/{len(recs)} "
              f"({100*liftable/len(recs):.0f}%)  mean depth-CoV={np.mean(covs) if covs else float('nan'):.3f}  "
              f"mean finite-frac in patch={np.mean(fracs):.2f}")

    multi = [c for c in inst if c["n_views"] >= 2]
    print(f"\nobject instances: {len(inst)} total, {len(multi)} seen in >=2 views (stable)")

    # The REAL glass test: do per-view lifts RECONCILE across views? Glass bakes reflections
    # into surface gaussians -> depth is finite (passes the CoV test) but lands at different 3D
    # places per view, so it fails to merge (R=0.5m) and stays a singleton. Cross-view
    # reconciliation rate is the honest consistency signal; single-view depth-CoV cannot see it.
    print("\n--- cross-view reconciliation (stable = lifted detection merged across >=2 views) ---")
    det_in_stable = {c_id: 0 for c_id in ("solid", "glass", "reflective")}
    det_total = {c_id: 0 for c_id in ("solid", "glass", "reflective")}
    for c in inst:
        det_total[c["surface"]] = det_total.get(c["surface"], 0) + c["n_views"]
        if c["n_views"] >= 2:
            det_in_stable[c["surface"]] = det_in_stable.get(c["surface"], 0) + c["n_views"]
    for surf in ("solid", "glass", "reflective"):
        tot = det_total.get(surf, 0)
        if tot:
            rate = 100 * det_in_stable.get(surf, 0) / tot
            print(f"  {surf:11s}: {det_in_stable.get(surf,0)}/{tot} lifted dets reconciled across views "
                  f"({rate:.0f}%)")
        else:
            print(f"  {surf:11s}: (none)")
    print("sample instances (label / surface / n_views / center):")
    for c in sorted(inst, key=lambda x: -x["n_views"])[:12]:
        lab = Counter(c["labels"]).most_common(1)[0][0]
        ctr = ", ".join(f"{v:.2f}" for v in c["center"])
        print(f"  {lab:28s} {c['surface']:10s} x{c['n_views']:2d}  ({ctr})")

    out = "/tmp/claude-1001/-mnt-wd-gaussian-bot/4348b233-110b-4581-9dba-45ed28a58b14/scratchpad/scene_graph.json"
    graph = {"scene_id": "ufficio360", "n_views": int(N_VIEWS),
             "objects": [{"id": f"obj_{k:03d}", "label": Counter(c["labels"]).most_common(1)[0][0],
                          "surface": c["surface"], "center": c["center"], "n_views": c["n_views"]}
                         for k, c in enumerate(inst)]}
    with open(out, "w") as f:
        json.dump(graph, f, indent=1)
    print(f"\nwrote {out} ({len(inst)} objects)")


if __name__ == "__main__":
    sys.exit(main())
