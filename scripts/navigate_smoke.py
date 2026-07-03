"""Navigate-mode eval smoke: two scripted instructions on the office scene (ADR-0012).

Runs `gaussian-robot navigate` for a couple of goto instructions whose target
coordinates come from the extracted scene graph (experiments/taskgen/scene_graph.json),
so success is MEASURED (geometric), and prints a compact scoreboard.

    uv run python scripts/navigate_smoke.py \
        --ply /mnt/archive/datasets/ufficio360-.../gaussian_pointcloud_30000_original.ply \
        --vlm-url http://127.0.0.1:8000/v1     # or: --start-vllm

Exit code = number of failed episodes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

EPISODES = [
    {"instruction": "Go to the water cooler.", "target": "0.78,-1.93,-0.35"},
    {"instruction": "Go to the photocopier.", "target": "1.21,-0.43,-0.92"},
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply", required=True, help="Office-scene splat .ply")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vlm-url", default=None, help="Existing OpenAI-compatible endpoint")
    ap.add_argument("--start-vllm", action="store_true")
    ap.add_argument("--demo-vlm", action="store_true", help="Scripted VLM (plumbing check only)")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--out-root", default="data/episodes/smoke")
    args = ap.parse_args()

    failures = 0
    board: list[dict[str, object]] = []
    for i, ep in enumerate(EPISODES):
        out_dir = Path(args.out_root) / f"ep{i:02d}"
        cmd = [
            "uv",
            "run",
            "gaussian-robot",
            "navigate",
            args.ply,
            "--instruction",
            ep["instruction"],
            "--target-xyz",
            ep["target"],
            "--device",
            args.device,
            "--max-steps",
            str(args.max_steps),
            "--out-dir",
            str(out_dir),
        ]
        if args.vlm_url:
            cmd += ["--vlm-url", args.vlm_url]
        if args.start_vllm and i == 0:  # the first episode boots the server; later ones reuse it
            cmd += ["--start-vllm"]
        if args.demo_vlm:
            cmd += ["--demo-vlm"]
        print(f"\n=== episode {i}: {ep['instruction']}", flush=True)
        proc = subprocess.run(cmd, check=False)
        record = json.loads((out_dir / "episode.json").read_text())
        board.append(
            {
                "instruction": ep["instruction"],
                "success": record["success"],
                "source": record["success_source"],
                "stop_reason": record["stop_reason"],
                "steps": record["steps"],
                "gif": str(out_dir / "episode.gif"),
            }
        )
        failures += 0 if proc.returncode == 0 else 1

    print("\n=== smoke scoreboard ===")
    print(json.dumps(board, indent=1))
    print(f"{len(EPISODES) - failures}/{len(EPISODES)} succeeded")
    return failures


if __name__ == "__main__":
    sys.exit(main())
