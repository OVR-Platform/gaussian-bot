# scripts/

One-off research scripts. Anything exploratory (a render sweep, a prompt
experiment, a calibration plot) goes here, not in `src/`.

Run them with the project's environment:

```bash
uv run python scripts/my_experiment.py
```

These scripts are **not** part of the installed package. Don't import from
`scripts/` into `src/gaussian_robot/`.

## Status of the enhancement scripts (ADR-0011)

The supported splat-enhancement path is the CLI:

```bash
uv run gaussian-robot enhance <scene.ply> --out <new.ply> --colmap <sparse/0> --images <dir>
```

| Script | Status |
|---|---|
| `enhance_scene.py` | Superseded by the CLI (filler-less anchored polish; research reference). |
| `fill_gaps.py` | Superseded by the CLI (`--rounds-mode` covers its regime). |
| `explore_and_fill.py` | Superseded by the CLI (which wraps exactly this pipeline). |
| `milestone0_identity_distill.py` | **Active** — the Milestone-0 plumbing gate (PLY round-trip + identity distill under the VRAM ceiling); not a user path. |
