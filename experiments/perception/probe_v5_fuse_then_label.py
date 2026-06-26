"""Perception v5 (ConceptGraphs-style): fuse masks across views FIRST, then label each instance ONCE.

Improves v4 on three axes the data needs: more photos, real multi-view fusion (a physical object
seen in many frames becomes ONE 3D instance), and dedup (pervasive surfaces like the ceiling
collapse to one, not dozens of tiles). Labelling once per fused instance (not per mask per view)
is far cheaper AND less noisy. All thresholds are fractions of scene extent — no absolute units.

Run from repo root:  uv run --with scipy python experiments/perception/probe_v5_fuse_then_label.py
"""

from __future__ import annotations

import json
import re
import struct
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
import torch
from PIL import Image

from gaussian_robot.backends.gsplat_renderer import GsplatRenderer
from gaussian_robot.render.camera import Camera, CameraIntrinsics
from gaussian_robot.splat.capture_poses import parse_colmap_images_bin
from gaussian_robot.vlm.qwen import jpeg_data_url

SCENE = Path("/mnt/archive/datasets/ufficio360-35a39133-e1f2-4426-86d4-a3d7a00614ee-PIC")
DEVICE = "cuda:1"
VLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-9B"
N_PHOTOS = 30
S = 1024
FX = 1024 * (S / 2048)
INTR = CameraIntrinsics(fx=FX, fy=FX, cx=S / 2, cy=S / 2, width=S, height=S)
OUT = Path(__file__).parent / "scene_graph_v5.json"


def image_names() -> list[str]:
    names = []
    with open(SCENE / "sparse/0/images.bin", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4); f.read(32); f.read(24); f.read(4); nm = b""
            while (ch := f.read(1)) not in (b"\x00", b""):
                nm += ch
            npt = struct.unpack("<Q", f.read(8))[0]; f.read(npt * 24); names.append(nm.decode())
    return names


def unproject_many(us, vs, ds, pose) -> np.ndarray:
    cam = np.stack([(us - INTR.cx) / FX * ds, (vs - INTR.cy) / FX * ds, ds], axis=1)
    return pose.position + cam @ pose.rotation  # (N,3); rotation is world->cam, so @R = R^T applied


def vlm_label(crop: np.ndarray) -> dict:
    prompt = ("Cropped instance from a REAL indoor-office photo. Name the main thing (1-4 words); "
              "object_class = surface (pervasive floor/wall/ceiling/carpet) | landmark (fixed: "
              "column/door/staircase) | object (discrete); manipulable (carryable) true/false. "
              'ONLY JSON: {"label":"..","object_class":"..","manipulable":bool}')
    body = {"model": MODEL, "max_tokens": 150, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": "Reply with only the JSON object."},
                         {"role": "user", "content": [
                             {"type": "image_url", "image_url": {"url": jpeg_data_url(crop)}},
                             {"type": "text", "text": prompt}]}]}
    try:
        with httpx.Client(timeout=60) as c:
            txt = c.post(VLM_URL, json=body).json()["choices"][0]["message"]["content"]
        o = json.loads(re.findall(r'\{[^{}]*"label"[^{}]*\}', txt, re.DOTALL)[-1])
        return {"label": str(o.get("label", "?"))[:40],
                "object_class": str(o.get("object_class", "object")).lower(),
                "manipulable": bool(o.get("manipulable", False))}
    except Exception:
        return {"label": "?", "object_class": "object", "manipulable": False}


def main() -> None:
    from transformers import pipeline

    poses = parse_colmap_images_bin(SCENE / "sparse/0/images.bin")
    names = image_names()
    idx = np.linspace(0, len(poses) - 1, N_PHOTOS).astype(int)
    sam = pipeline("mask-generation", model="facebook/sam-vit-base", device=DEVICE)
    renderer = GsplatRenderer.from_path(str(SCENE / "gaussian_pointcloud_30000_original.ply"),
                                        device=DEVICE)
    min_a, max_a = S * S * 0.004, S * S * 0.5

    # --- pass 1: SAM + lift to 3D (no VLM) ---
    dets = []
    for n, i in enumerate(idx):
        pose = poses[int(i)]
        ip = SCENE / "images" / names[int(i)].replace(".jpg", ".png")
        if not ip.exists():
            continue
        photo = np.array(Image.open(ip).convert("RGB").resize((S, S)))
        depth = renderer.render(Camera(pose=pose, intrinsics=INTR)).depth
        out = sam(Image.fromarray(photo), points_per_side=16, pred_iou_thresh=0.9)
        kept = 0
        for m in out["masks"]:
            m = np.asarray(m, bool)
            a = int(m.sum())
            if not (min_a <= a <= max_a):
                continue
            ys, xs = np.nonzero(m)
            dd = depth[ys, xs]
            fin = np.isfinite(dd)
            if int(fin.sum()) < 0.3 * a:
                continue
            pts = unproject_many(xs[fin].astype(float), ys[fin].astype(float), dd[fin], pose)
            sel = np.linspace(0, len(pts) - 1, min(60, len(pts))).astype(int)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            dets.append({"c": np.median(pts, axis=0), "pts": pts[sel], "view": n, "area": a,
                         "bbox": (x0, y0, x1, y1), "photo": int(i)})
            kept += 1
        print(f"  photo {n:2d} ({names[int(i)][-12:]}): masks={len(out['masks'])} lifted={kept}",
              flush=True)

    diag = float(np.linalg.norm(np.ptp(np.array([d["c"] for d in dets]), axis=0))) if dets else 1.0
    R, R_DEDUP = 0.03 * diag, 0.06 * diag
    print(f"\n{len(dets)} mask-detections; scene diag={diag:.1f} -> fuse R={R:.2f} dedup={R_DEDUP:.2f}",
          flush=True)

    # --- fuse masks into 3D instances by centroid proximity ---
    inst = []
    for d in sorted(dets, key=lambda x: -x["area"]):
        hit = next((c for c in inst if np.linalg.norm(d["c"] - c["c"]) < R), None)
        if hit:
            hit["members"].append(d); hit["views"].add(d["view"])
            hit["allpts"].append(d["pts"]); hit["c"] = np.median(np.vstack(hit["allpts"]), axis=0)
        else:
            inst.append({"c": d["c"], "members": [d], "views": {d["view"]}, "allpts": [d["pts"]]})
    inst = [c for c in inst if len(c["views"]) >= 2]  # keep multi-view instances
    print(f"fused -> {len(inst)} multi-view instances; labelling once each...", flush=True)

    # --- label ONCE per fused instance (best/largest member crop) ---
    for c in inst:
        best = max(c["members"], key=lambda m: m["area"])
        x0, y0, x1, y1 = best["bbox"]
        photo = np.array(Image.open(SCENE / "images" / names[best["photo"]].replace(".jpg", ".png"))
                         .convert("RGB").resize((S, S)))
        c.update(vlm_label(photo[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4]))

    # --- dedup: surfaces collapse by label (pervasive); objects/landmarks by label+proximity ---
    final = []
    for c in sorted(inst, key=lambda x: -len(x["views"])):
        key = c["label"].lower()
        if c["object_class"] == "surface":
            same = next((f for f in final if f["object_class"] == "surface" and f["label"].lower() == key), None)
        else:
            same = next((f for f in final if f["label"].lower() == key
                         and np.linalg.norm(np.array(f["center"]) - c["c"]) < R_DEDUP), None)
        if same:
            same["_n"] += len(c["views"])
            continue
        final.append({"label": c["label"], "object_class": c["object_class"],
                      "manipulable": c["manipulable"], "center": [round(float(v), 2) for v in c["c"]],
                      "_n": len(c["views"])})

    graph = {"scene_id": "ufficio360", "up_axis": "-y", "schema_version": 1,
             "provenance": {"perception": f"probe_v5_fuse_then_label ({N_PHOTOS} real photos)"},
             "objects": [{"id": f"obj_{k:02d}", **{x: o[x] for x in
                          ("label", "object_class", "manipulable", "center")}}
                         for k, o in enumerate(final)]}
    OUT.write_text(json.dumps(graph, indent=1))
    print(f"\nclass mix: {dict(Counter(o['object_class'] for o in graph['objects']))}")
    print(f"dataset: {len(graph['objects'])} objects (was {len(inst)} pre-dedup) -> {OUT}")
    for o in graph["objects"]:
        print(f"  {o['label']:26s} {o['object_class']:8s} manip={str(o['manipulable']):5s}")


if __name__ == "__main__":
    main()
