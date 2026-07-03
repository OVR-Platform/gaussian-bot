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

## Navigate with an instruction (VLN)

One command runs a goal-conditioned episode against the real stack — gsplat
renderer + Qwen served via vLLM (ADR-0012):

```bash
uv sync --extra gsplat --extra vlm   # renderer + OpenAI-compatible client

# spawn vLLM and block until ready, then run the episode
uv run gaussian-robot navigate /path/scene/points.ply \
    --instruction "go to the blue mat" \
    --target-xyz "1.9,0.4,-3.4" \
    --start-vllm

# or against an already-running server / with the scripted demo VLM
uv run gaussian-robot navigate /path/scene/points.ply \
    --instruction "find the desk" --vlm-url http://127.0.0.1:8000/v1
```

Outputs: `data/episodes/episode.json` (trajectory + outcome),
`episode.gif` (`[robot view | top-down trail+goal]`), and a printed
`success / stop_reason / steps` line. **Success provenance is explicit**:
with `--target-xyz` (goal coordinates, e.g. from the scene graph) arrival is
*measured* (`geometric`, radius `--goal-eps`); without it, the VLM declaring
completion is reported as `vlm_declared` — unverified. The office-scene smoke
is `scripts/navigate_smoke.py`.

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
