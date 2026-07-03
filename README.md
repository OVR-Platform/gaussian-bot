# gaussian-robot

> R&D project: navigating a robot **inside** a 3D Gaussian Splat scene by
> extracting rendered views and feeding them to a Vision-Language Model (VLM).

## Status

🧪 **Pre-Alpha / Research.** The architecture is intentionally pluggable:
nothing is pinned to a specific splat renderer or VLM yet. We are exploring the
design space.

## Pipeline (working hypothesis)

```
        ┌─────────────┐    pose    ┌──────────────┐   RGB(+depth)  ┌─────────┐
 Robot  │  Navigator  │ ─────────▶ │   Renderer   │ ─────────────▶ │   VLM   │
 ◀──────┤  (planner)  │ ◀───────── │ (from splat) │ ◀───────────── │ (vLLM)  │
 action └─────────────┘   decision └──────────────┘    perception  └─────────┘
```

1. A robot holds a **pose** inside a reconstructed scene.
2. A **renderer** turns that pose into a rendered view (RGB, optionally depth).
3. A **VLM** consumes the view and returns a decision / description.
4. The **navigator** translates decisions back into pose changes.

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11–3.12.

```bash
# install base + dev tooling
uv sync --extra dev

# opt into a renderer backend later (not required yet)
# uv sync --extra gsplat
# opt into a VLM client later
# uv sync --extra vlm
# opt into a local vLLM server later
# uv sync --extra vlm --extra vllm

# run the CLI
uv run gaussian-robot --help

# run the dashboard on the LAN
uv run gaussian-robot ui

# optionally launch vLLM with the configured Hugging Face model id
uv run gaussian-robot ui --start-vllm

# lint / type-check / test
uv run ruff check .
uv run mypy
uv run pytest
```

## Enhance a splat (fix broken / under-observed gaussians)

The supported path (ADR-0011) is one command — the robot explores the scene,
marks under-observed viewpoints, the reference-conditioned Difix3D+ filler
cleans them, and the result is distilled into a **new** `.ply` (the source is
never touched):

```bash
uv sync --extra gsplat   # renderer + distiller
uv pip install diffusers accelerate peft safetensors  # Difix filler deps

uv run gaussian-robot enhance /path/scene/points.ply \
    --out data/enhanced/scene_enhanced.ply \
    --colmap /path/scene/sparse/0 \
    --images /path/scene/images \
    --report data/enhanced/report.json
```

Outputs: the new `.ply`, a `[BEFORE | AFTER | map]` fly-through GIF along the
robot's path, the held-out **Δ-PSNR** regression guard, and the fill-phase
**peak VRAM** (the whole loop fits a single 24 GB card; the Difix forward is
fp16 by default). Opt-in quality levers: `--denoise-steps N` (multi-step
τ-ladder) and `--sdedit` (ArtiFixer-style opacity-mix latent init, requires
`N ≥ 2`). See `uv run gaussian-robot enhance --help` for the full knob set.

**Regime honesty:** this pays off on *sparse / under-observed* captures. On a
dense, well-covered scene the gate measures ~0 gain and the fill is a no-op by
design (see [`docs/research/README.md`](docs/research/README.md)).

## Project layout

```
src/gaussian_robot/
├── splat/    # Scene loading & representation (.ply / .splat)
├── render/   # Pluggable renderer protocol + camera/pose types
├── nav/      # Robot state, planner
├── vlm/      # VLM client interface (vLLM etc.)
└── cli.py    # Entry point
docs/
├── adr/      # Architecture Decision Records
└── CONTEXT.md# Domain language & project context
scripts/      # One-off research scripts
data/         # Gitignored: splats, renders, weights
```

## Docs

- [`docs/CONTEXT.md`](docs/CONTEXT.md) — domain language and project context.
- [`docs/adr/`](docs/adr/) — architecture decision records.
- [`AGENTS.md`](AGENTS.md) — guidance for AI coding agents working in this repo.

## License

Proprietary.
