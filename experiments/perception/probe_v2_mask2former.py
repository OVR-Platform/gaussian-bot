"""Perception prototype v2: real instance masks (Mask2Former) instead of VLM center-points.

Tests the hypothesis from v1 that the bottleneck was localization/association, not glass:
swap the VLM point-guess for proper instance segmentation, lift each mask's centroid via the
splat depth, fuse across views label-aware, and compare cross-view reconciliation to v1's ~24%.

Run with:  uv run --with scipy python perception_probe_v2.py
(masks_sam3/ turned out to be BINARY foreground masks, not instances, so we segment here.)
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch

from gaussian_robot.backends.gsplat_renderer import GsplatRenderer
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose
from gaussian_robot.splat.capture_poses import discover_capture_poses, load_capture_poses

PLY = "/mnt/archive/datasets/ufficio360-35a39133-e1f2-4426-86d4-a3d7a00614ee-PIC/gaussian_pointcloud_30000_original.ply"
DEVICE = "cuda:1"
MODEL_ID = "facebook/mask2former-swin-small-coco-instance"
N_VIEWS = 20
W = H = 1024
FX = FY = W / 2  # 90 deg fov, matching the capture rig (SIMPLE_PINHOLE fx=1024 @ 2048)
INTR = CameraIntrinsics(fx=FX, fy=FY, cx=W / 2, cy=H / 2, width=W, height=H)
SCORE_THR = 0.85
MIN_AREA = 1500  # px
MERGE_R = 0.6  # meters, label-aware 3D merge radius


def unproject(u: float, v: float, d: float, pose: Pose) -> np.ndarray:
    cam = np.array([(u - INTR.cx) / FX * d, (v - INTR.cy) / FY * d, d], dtype=np.float64)
    return pose.position + pose.rotation.T @ cam


def main() -> None:
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    print(f"loading Mask2Former {MODEL_ID} ...", flush=True)
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    seg = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID).to(DEVICE).eval()
    id2label = seg.config.id2label

    print(f"loading splat on {DEVICE} ...", flush=True)
    renderer = GsplatRenderer.from_path(PLY, device=DEVICE)
    poses = load_capture_poses(discover_capture_poses(PLY))
    idx = np.linspace(0, len(poses) - 1, N_VIEWS).astype(int)
    print(f"{len(poses)} poses, sampling {len(idx)}", flush=True)

    dets: list[dict] = []
    for n, i in enumerate(idx):
        pose = poses[int(i)]
        res = renderer.render(Camera(pose=pose, intrinsics=INTR))
        rgb = res.rgb
        inputs = proc(images=rgb, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = seg(**inputs)
        result = proc.post_process_instance_segmentation(
            out, target_sizes=[(H, W)], threshold=SCORE_THR
        )[0]
        seg_map = result["segmentation"].cpu().numpy()
        kept = 0
        for info in result["segments_info"]:
            mask = seg_map == info["id"]
            area = int(mask.sum())
            if area < MIN_AREA:
                continue
            ys, xs = np.nonzero(mask)
            u, v = float(xs.mean()), float(ys.mean())
            md = res.depth[ys, xs]
            md = md[np.isfinite(md)]
            if md.size == 0:
                continue
            d = float(np.median(md))
            label = id2label.get(info["label_id"], str(info["label_id"]))
            dets.append({"label": label, "score": float(info["score"]), "area": area,
                         "xyz": unproject(u, v, d, pose).tolist(), "view": n})
            kept += 1
        print(f"  view {n:2d} (pose {i:3d}): instances={len(result['segments_info'])} kept={kept}",
              flush=True)

    # label-aware greedy 3D fusion
    inst: list[dict] = []
    for d in dets:
        p = np.array(d["xyz"])
        hit = None
        for c in inst:
            if c["label"] == d["label"] and np.linalg.norm(p - np.array(c["center"])) < MERGE_R:
                hit = c
                break
        if hit:
            hit["pts"].append(p)
            hit["center"] = np.mean(hit["pts"], axis=0).tolist()
            hit["views"].add(d["view"])
        else:
            inst.append({"label": d["label"], "center": p.tolist(), "pts": [p],
                         "views": {d["view"]}})

    multi = [c for c in inst if len(c["views"]) >= 2]
    recon = sum(len(c["views"]) for c in multi)
    total = len(dets)
    print("\n========== v2 (Mask2Former instance masks) ==========")
    print(f"detections: {total}   instances: {len(inst)}   stable(>=2 views): {len(multi)}")
    print(f"cross-view reconciliation: {recon}/{total} dets in multi-view instances "
          f"({100*recon/max(1,total):.0f}%)   [v1 VLM-points was ~24%]")
    print("\ntop instances (label / n_views / center):")
    for c in sorted(inst, key=lambda x: -len(x["views"]))[:15]:
        ctr = ", ".join(f"{v:.2f}" for v in c["center"])
        print(f"  {c['label']:18s} x{len(c['views']):2d}  ({ctr})")
    print("\nlabel histogram:", Counter(d["label"] for d in dets).most_common(12))


if __name__ == "__main__":
    main()
