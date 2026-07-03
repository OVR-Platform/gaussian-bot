# AGENTS.md

Guidance for AI coding agents (and humans pairing with them) working in this
repo.

## What this project is

R&D: navigating a robot inside a 3D Gaussian Splat by rendering views and
feeding them to a VLM. See `docs/CONTEXT.md` for the domain language and
`README.md` for the pipeline.

**We are early.** The renderer and VLM backends are *not chosen* (see
`docs/adr/0001-...`). Treat them as pluggable.

## Before you change code

1. **Read `docs/CONTEXT.md`** and use its terminology in code, comments, and
   commit messages.
2. **Respect the seams.** `nav/` must not import a concrete renderer or VLM —
   only the protocols in `render/base.py` and `vlm/client.py`. If you find
   yourself wanting to, the right move is a new backend module passed in via DI.
3. **Check the ADRs** (`docs/adr/`). If your change contradicts an accepted ADR,
   write a superseding ADR instead of silently diverging. If your change is a
   significant decision that isn't recorded, add an ADR.

## Commands

This project uses `uv`. Always run tools through `uv run`.

```bash
uv sync --extra dev        # install everything you need
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy                # type-check (strict)
uv run pytest              # tests
uv run pytest -m "not slow"   # skip slow/gpu tests
```

Run lint + mypy + pytest before declaring a task done.

## Conventions

- **Python 3.11–3.12.** ML libs lag; do not bump past 3.12 without an ADR.
- **Types everywhere.** `mypy --strict` is on. No `Any` without a comment
  explaining why.
- **Math types are numpy-backed dataclasses** (see `render/camera.py`), not
  pydantic. Pydantic is for config/IO, not linear algebra.
- **No comments unless asked.** Self-documenting names > comments. Docstrings on
  public modules/classes/functions are encouraged.
- **Heavy deps (torch, gsplat, openai) are optional extras**, never in the base
  `dependencies`. Import them lazily inside backend modules.
- **Tests** go in `tests/`, mirroring `src/gaussian_robot/`. Mark GPU/slow tests
  with `@pytest.mark.gpu` / `@pytest.mark.slow`.

## Data & weights

`data/` is gitignored. Never commit a `.ply`, `.splat`, model weights, or
rendered images. Put one-off research scripts in `scripts/`.

In `experiments/`, only code and tiny JSON fixtures are tracked; every heavy
output (GIF/PNG/video/arrays/logs) is gitignored — see `experiments/README.md`.
Agent scratch worktrees live under `.claude/worktrees/` and are gitignored.

## Committing

- Don't commit unless explicitly asked.
- Keep commits focused; match the existing commit message style.
- Never commit secrets — `.env` is gitignored for a reason.
