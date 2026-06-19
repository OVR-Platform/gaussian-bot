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

# run the CLI
uv run gaussian-robot --help

# lint / type-check / test
uv run ruff check .
uv run mypy
uv run pytest
```

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
