"""Faithful single-photo annotation: SAM masks + VLM label/class drawn WHERE detected.

No 3D projection, no fusion — just what the perception actually sees on one real frame, so the
true label quality is visible (not confounded by lift/fusion localisation noise).

Run from repo root:  uv run --with scipy python experiments/perception/annotate_one_photo.py [photo_idx]
"""

from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageDraw

from gaussian_robot.vlm.qwen import jpeg_data_url

SCENE = Path("/mnt/archive/datasets/ufficio360-35a39133-e1f2-4426-86d4-a3d7a00614ee-PIC")
VLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-9B"
S = 1024
COL = {"surface": (90, 160, 230), "landmark": (240, 180, 40), "object": (80, 230, 120)}


def annotate(crop: np.ndarray) -> dict:
    prompt = ("Cropped instance from a real indoor-office photo. Name the main thing (1-4 words); "
              "object_class = surface (floor/wall/ceiling/carpet) | landmark (column/door/stairs) | "
              "object (discrete); manipulable true/false. ONLY JSON: "
              '{"label":"..","object_class":"..","manipulable":bool}')
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
        return {"label": str(o.get("label", "?"))[:30],
                "cls": str(o.get("object_class", "object")).lower()}
    except Exception:
        return {"label": "?", "cls": "object"}


def main() -> None:
    from transformers import pipeline

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    names = []
    with open(SCENE / "sparse/0/images.bin", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4); f.read(32); f.read(24); f.read(4); nm = b""
            while (ch := f.read(1)) not in (b"\x00", b""):
                nm += ch
            npt = struct.unpack("<Q", f.read(8))[0]; f.read(npt * 24); names.append(nm.decode())
    photo = np.array(Image.open(SCENE / "images" / names[idx].replace(".jpg", ".png"))
                     .convert("RGB").resize((S, S)))
    print(f"photo {idx}: {names[idx]}", flush=True)

    sam = pipeline("mask-generation", model="facebook/sam-vit-base", device="cuda:1")
    out = sam(Image.fromarray(photo), points_per_side=16, pred_iou_thresh=0.9)
    img = Image.fromarray(photo); d = ImageDraw.Draw(img)
    kept = []
    for m in out["masks"]:
        m = np.asarray(m, bool)
        a = int(m.sum())
        if not (S * S * 0.01 <= a <= S * S * 0.4):  # ignore tiny + whole-frame masks
            continue
        ys, xs = np.nonzero(m)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        ann = annotate(photo[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4])
        c = COL.get(ann["cls"], (200, 200, 200))
        d.rectangle([x0, y0, x1, y1], outline=c, width=2)
        d.rectangle([x0, y0, x0 + 9 * len(f"{ann['label']} [{ann['cls']}]"), y0 + 14], fill=(0, 0, 0))
        d.text((x0 + 2, y0 + 2), f"{ann['label']} [{ann['cls']}]", fill=c)
        kept.append(ann)
        print(f"  {ann['label']:26s} [{ann['cls']}]", flush=True)
    out_p = Path(__file__).parent / f"annotated_one_{idx}.png"
    img.save(out_p)
    print(f"\n{len(kept)} detections -> {out_p}")


if __name__ == "__main__":
    main()
