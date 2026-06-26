"""Perception v4 = annotate the REAL mapping photos (not splat renders), then lift to 3D.

Architectural correction: the semantic layer must come from the actual captured images (sharp,
real pixels) — splat renders carry artifacts (smearing on glass, the carpet misread). So we run
SAM + VLM (label + object_class + manipulable) on the real images/, and use the splat ONLY for
metric depth: render at each photo's COLMAP pose+intrinsics to get an aligned depth map, lift each
mask to 3D, fuse multi-view -> scene_graph.json. Task generation then consumes this annotated map.

Run from repo root:  uv run --with scipy python experiments/perception/probe_v4_annotate_photos.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
import torch
from PIL import Image

from gaussian_robot.backends.gsplat_renderer import GsplatRenderer
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose
from gaussian_robot.splat.capture_poses import parse_colmap_images_bin
from gaussian_robot.vlm.qwen import jpeg_data_url

SCENE = Path("/mnt/archive/datasets/ufficio360-35a39133-e1f2-4426-86d4-a3d7a00614ee-PIC")
PLY = SCENE / "gaussian_pointcloud_30000_original.ply"
IMAGES = SCENE / "images"
DEVICE = "cuda:1"
VLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-9B"
N_PHOTOS = 12
S = 1024  # annotate/lift at this resolution; capture is SIMPLE_PINHOLE f=1024 @2048 -> scale 0.5
FX = FY = 1024 * (S / 2048)
INTR = CameraIntrinsics(fx=FX, fy=FY, cx=S / 2, cy=S / 2, width=S, height=S)
MERGE_R = 0.6
OUT = Path(__file__).parent / "scene_graph_v4.json"


def unproject(u: float, v: float, d: float, pose: Pose) -> np.ndarray:
    cam = np.array([(u - INTR.cx) / FX * d, (v - INTR.cy) / FY * d, d], dtype=np.float64)
    return pose.position + pose.rotation.T @ cam


def vlm_annotate(crop: np.ndarray) -> dict:
    prompt = ("Cropped instance from a REAL photo of an indoor office. Name the main thing "
              "(1-4 words); classify object_class as 'surface' (pervasive floor/wall/ceiling/"
              "carpet), 'landmark' (fixed, walk-to: column/door/staircase) or 'object' (discrete); "
              "and manipulable (carryable) true/false. ONLY JSON, no reasoning: "
              '{"label":"..","object_class":"surface|landmark|object","manipulable":bool}')
    body = {"model": MODEL, "max_tokens": 200, "temperature": 0.0,
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
    except Exception as e:
        return {"label": f"?({type(e).__name__})", "object_class": "object", "manipulable": False}


def main() -> None:
    from transformers import pipeline

    poses = parse_colmap_images_bin(SCENE / "sparse/0/images.bin")
    import struct
    names = []
    with open(SCENE / "sparse/0/images.bin", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4); f.read(32); f.read(24); f.read(4)
            nm = b""
            while (ch := f.read(1)) not in (b"\x00", b""):
                nm += ch
            npt = struct.unpack("<Q", f.read(8))[0]; f.read(npt * 24)
            names.append(nm.decode())
    idx = np.linspace(0, len(poses) - 1, N_PHOTOS).astype(int)
    print(f"{len(poses)} mapping photos, annotating {len(idx)} real images", flush=True)

    sam = pipeline("mask-generation", model="facebook/sam-vit-base", device=DEVICE)
    renderer = GsplatRenderer.from_path(str(PLY), device=DEVICE)
    min_area, max_area = S * S * 0.004, S * S * 0.5
    dets = []
    for n, i in enumerate(idx):
        pose = poses[int(i)]
        ipath = IMAGES / names[int(i)].replace(".jpg", ".png")
        if not ipath.exists():
            continue
        photo = np.array(Image.open(ipath).convert("RGB").resize((S, S)))  # REAL pixels
        depth = renderer.render(Camera(pose=pose, intrinsics=INTR)).depth  # splat depth, aligned
        out = sam(Image.fromarray(photo), points_per_side=16, pred_iou_thresh=0.88)
        kept = 0
        for mask in out["masks"]:
            m = np.asarray(mask, bool)
            if not (min_area <= int(m.sum()) <= max_area):
                continue
            ys, xs = np.nonzero(m)
            md = depth[ys, xs]; md = md[np.isfinite(md)]
            if md.size < 0.3 * int(m.sum()):
                continue
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            crop = photo[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4]  # crop the REAL photo
            ann = vlm_annotate(crop)
            dets.append({"xyz": unproject(float(xs.mean()), float(ys.mean()), float(np.median(md)),
                                          pose).tolist(), "view": n, **ann})
            kept += 1
        print(f"  photo {n:2d} ({names[int(i)][-12:]}): masks={len(out['masks'])} kept={kept}",
              flush=True)

    inst = []  # fuse by 3D proximity + same object_class
    for d in dets:
        p = np.array(d["xyz"])
        hit = next((c for c in inst if c["object_class"] == d["object_class"]
                    and np.linalg.norm(p - np.array(c["center"])) < MERGE_R), None)
        if hit:
            hit["pts"].append(p); hit["center"] = np.mean(hit["pts"], axis=0).tolist()
            hit["views"].add(d["view"]); hit["labels"].append(d["label"])
        else:
            inst.append({"center": p.tolist(), "pts": [p], "views": {d["view"]},
                         "labels": [d["label"]], "object_class": d["object_class"],
                         "manipulable": d["manipulable"]})
    multi = [c for c in inst if len(c["views"]) >= 2]
    graph = {"scene_id": "ufficio360", "up_axis": "-y", "schema_version": 1,
             "provenance": {"perception": "probe_v4_annotate_photos (real mapping images)"},
             "objects": [{"id": f"obj_{k:02d}",
                          "label": Counter(c["labels"]).most_common(1)[0][0],
                          "object_class": c["object_class"], "manipulable": c["manipulable"],
                          "center": [round(float(v), 2) for v in c["center"]]}
                         for k, c in enumerate(sorted(multi, key=lambda x: -len(x["views"])))]}
    OUT.write_text(json.dumps(graph, indent=1))
    print(f"\nclass mix: {dict(Counter(o['object_class'] for o in graph['objects']))}")
    print(f"wrote {OUT}: {len(graph['objects'])} objects from REAL photos")
    for o in graph["objects"][:15]:
        print(f"  {o['label']:26s} {o['object_class']:8s} manip={str(o['manipulable']):5s}")


if __name__ == "__main__":
    main()
