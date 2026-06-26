"""Perception prototype v3 = the synthesis (ConceptGraphs-style):
SAM class-agnostic masks (good localization) + open-vocab VLM labels (office-wide breadth),
fused multi-view. Goal: keep v2's high reconciliation AND v1's open-vocab coverage.

Run with:  uv run --with scipy python perception_probe_v3.py
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
import torch
from PIL import Image

from gaussian_robot.backends.gsplat_renderer import GsplatRenderer
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose
from gaussian_robot.splat.capture_poses import discover_capture_poses, load_capture_poses
from gaussian_robot.vlm.qwen import jpeg_data_url

PLY = "/mnt/archive/datasets/ufficio360-35a39133-e1f2-4426-86d4-a3d7a00614ee-PIC/gaussian_pointcloud_30000_original.ply"
DEVICE = "cuda:1"
VLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-9B"
N_VIEWS = 14
W = H = 768
FX = FY = W / 2  # 90 deg fov
INTR = CameraIntrinsics(fx=FX, fy=FY, cx=W / 2, cy=H / 2, width=W, height=H)
MIN_AREA = W * H * 0.004   # ignore tiny SAM fragments
MAX_AREA = W * H * 0.45    # ignore whole-image background blobs
MERGE_R = 0.6


def unproject(u: float, v: float, d: float, pose: Pose) -> np.ndarray:
    cam = np.array([(u - INTR.cx) / FX * d, (v - INTR.cy) / FY * d, d], dtype=np.float64)
    return pose.position + pose.rotation.T @ cam


def vlm_label(crop: np.ndarray) -> dict:
    """Label + classify one instance with the VLM. The VLM (not a label heuristic) decides whether
    it is a pervasive surface / fixed landmark / discrete object and whether it is manipulable —
    so the downstream feasibility gate never has to guess from hand-coded label lists. The visual
    is decisive: a floor-covering mask reads as a 'surface', not an object."""
    prompt = ("This is a cropped instance from an indoor office (3D-reconstruction render). "
              "Name the single main thing in 1-4 words; classify its material as "
              "solid|glass|reflective; classify it as one of: 'surface' (pervasive floor/wall/"
              "ceiling/carpet — no single location), 'landmark' (fixed structure you can walk to: "
              "column, door, staircase), or 'object' (discrete, possibly movable); and say if it is "
              "manipulable (carryable) true/false. Output ONLY JSON, no reasoning: "
              '{"label":"..","surface":"..","object_class":"surface|landmark|object","manipulable":bool}')
    body = {"model": MODEL, "max_tokens": 256, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": "Reply with only the JSON object. Do not explain."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": jpeg_data_url(crop)}},
                    {"type": "text", "text": prompt}]}]}
    try:
        with httpx.Client(timeout=60) as c:
            txt = c.post(VLM_URL, json=body).json()["choices"][0]["message"]["content"]
        o = json.loads(re.findall(r'\{[^{}]*"label"[^{}]*\}', txt, re.DOTALL)[-1])
        return {"label": str(o.get("label", "?"))[:40], "surface": str(o.get("surface", "solid")).lower(),
                "object_class": str(o.get("object_class", "object")).lower(),
                "manipulable": bool(o.get("manipulable", False))}
    except Exception as e:
        return {"label": f"?({type(e).__name__})", "surface": "solid",
                "object_class": "object", "manipulable": False}


def main() -> None:
    from transformers import pipeline

    print("loading SAM mask-generation ...", flush=True)
    sam = pipeline("mask-generation", model="facebook/sam-vit-base", device=DEVICE)
    print(f"loading splat on {DEVICE} ...", flush=True)
    renderer = GsplatRenderer.from_path(PLY, device=DEVICE)
    poses = load_capture_poses(discover_capture_poses(PLY))
    idx = np.linspace(0, len(poses) - 1, N_VIEWS).astype(int)
    print(f"{len(poses)} poses, sampling {len(idx)}", flush=True)

    renders: list[np.ndarray] = []
    dets: list[dict] = []
    for n, i in enumerate(idx):
        pose = poses[int(i)]
        res = renderer.render(Camera(pose=pose, intrinsics=INTR))
        renders.append(res.rgb)
        out = sam(Image.fromarray(res.rgb), points_per_side=16, pred_iou_thresh=0.88)
        kept = 0
        for mask in out["masks"]:
            mask = np.asarray(mask, dtype=bool)
            area = int(mask.sum())
            if area < MIN_AREA or area > MAX_AREA:
                continue
            ys, xs = np.nonzero(mask)
            md = res.depth[ys, xs]
            md = md[np.isfinite(md)]
            if md.size < 0.3 * area:  # mostly into the void -> skip
                continue
            d = float(np.median(md))
            u, v = float(xs.mean()), float(ys.mean())
            dets.append({"xyz": unproject(u, v, d, pose).tolist(), "view": n,
                         "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))})
            kept += 1
        print(f"  view {n:2d} (pose {i:3d}): sam_masks={len(out['masks'])} kept={kept}", flush=True)

    # proximity fusion (SAM is class-agnostic, so no label gate yet)
    inst: list[dict] = []
    for d in dets:
        p = np.array(d["xyz"])
        hit = next((c for c in inst if np.linalg.norm(p - np.array(c["center"])) < MERGE_R), None)
        if hit:
            hit["pts"].append(p)
            hit["center"] = np.mean(hit["pts"], axis=0).tolist()
            hit["views"].add(d["view"])
            hit["src"].append(d)
        else:
            inst.append({"center": p.tolist(), "pts": [p], "views": {d["view"]}, "src": [d]})

    multi = [c for c in inst if len(c["views"]) >= 2]
    recon = sum(len(c["views"]) for c in multi)
    print("\n========== v3 (SAM masks + open-vocab VLM labels) ==========")
    print(f"detections: {len(dets)}   instances: {len(inst)}   stable(>=2 views): {len(multi)}")
    print(f"cross-view reconciliation: {recon}/{len(dets)} ({100*recon/max(1,len(dets)):.0f}%)  "
          f"[v1 points 24% | v2 COCO-masks 47%]")

    # label + classify each stable instance with the VLM (open-vocab; class & manipulability too)
    print("\nlabelling stable instances with the VLM (open-vocab)...", flush=True)
    for c in sorted(multi, key=lambda x: -len(x["views"])):
        s = c["src"][0]
        x0, y0, x1, y1 = s["bbox"]
        crop = renders[s["view"]][max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4]
        c.update(vlm_label(crop))
    print(f"\nstable office inventory ({len(multi)} objects):")
    cls_count: Counter = Counter()
    for c in sorted(multi, key=lambda x: -len(x["views"])):
        ctr = ", ".join(f"{v:.2f}" for v in c["center"])
        cls_count[c["object_class"]] += 1
        print(f"  {c['label']:26s} {c['object_class']:8s} manip={str(c['manipulable']):5s} "
              f"x{len(c['views']):2d}  ({ctr})")
    print("\nclass mix of stable instances:", dict(cls_count))

    out = Path(__file__).parent / "scene_graph.json"
    graph = {"scene_id": "ufficio360", "up_axis": "-y", "schema_version": 1,
             "provenance": {"perception": "probe_v3_sam_openvocab"},
             "objects": [{"id": f"obj_{i:02d}", "label": c["label"], "surface": c["surface"],
                          "object_class": c["object_class"], "manipulable": c["manipulable"],
                          "center": [round(float(v), 2) for v in c["center"]]}
                         for i, c in enumerate(sorted(multi, key=lambda x: -len(x["views"])))]}
    out.write_text(json.dumps(graph, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
