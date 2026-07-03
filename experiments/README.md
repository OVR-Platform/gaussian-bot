# experiments/

One-off R&D probes and batch runners. Each subdirectory is a self-contained
experiment; see `perception/README.md` for the largest one.

## What gets committed

- **Tracked:** Python scripts and *tiny* JSON fixtures (scene graphs, task
  lists) that make an experiment reproducible.
- **Not tracked (gitignored):** everything an experiment *produces* — GIFs,
  PNGs/JPGs, videos, `.npy`/`.npz` arrays, `logs/`, `out/` directories, and
  anything covered by the repo-wide data rules (`.ply`, `.splat`, weights).

The point: a fresh clone gets the code to regenerate any artifact, never the
artifacts themselves. If an experiment's output is worth keeping, link it from
a doc and store it outside the repo (or regenerate it on demand). A genuinely
tiny fixture image needed by a test can be force-added (`git add -f`) with a
justification in the commit message.

## Layout

| Directory | What it is |
|---|---|
| `perception/` | Scene-graph extraction probes v1–v5 (VLM points → SAM → fuse-then-label). |
| `taskgen/` | Task generation over the extracted scene graph + batch task runner (ADR-0010). |
| `filler_probe/` | Standalone Difix3D+ smoke probe (loads the HF pipeline, one fill). |
| `gapfill_run/` | End-to-end gap-fill run + verification / novel-view inspection scripts. |
