# docs/research/ — index & reading order

Research notes behind the two active workstreams. These docs deliberately
overlap: they were written as successive deep-research passes, each building
on (and sometimes measuring away) the previous one. This index tells you what
each one is for, what supersedes what, and where to start.

## Splat enhancement (fix broken / under-observed gaussians)

Four docs, written in this order — read them in this order:

| # | Doc | What it is | Freshness |
|---|-----|------------|-----------|
| 1 | [splat-enhancement-study.md](splat-enhancement-study.md) | The foundational design study: architecture and vocabulary (`ViewFiller`, `mask.py`/`O_z`, distiller, orchestrator), tiered filler design, VRAM discipline, Milestone-0 gates. *(Lands with the `feat/splat-enhance-milestone0` branch.)* | Predates implementation — its VRAM/PSNR numbers are explicitly **projected** (§9). The measured figures (8.4 GB Difix peak, Milestone-0 results) supersede parts of §6–§7. |
| 2 | [difix3d-to-artifixer-evolution.md](difix3d-to-artifixer-evolution.md) | Conceptual background: what NVIDIA ArtiFixer actually adds over Difix3D+, with per-claim verification (several public claims are refuted in §5). Canonical opacity-mix formula + the latent-pooling caveat. | Current. |
| 3 | [difix-gap-artifixer-24gb.md](difix-gap-artifixer-24gb.md) | Plan A (five cheap supervision fixes) + Plan B (SEVA backbone) survey. **Read the in-doc measured Update first:** all five Plan A fixes were implemented and *none beat the +0.318 dB / +0.019 coverage baseline* on the office scene. | Plan A body **superseded by its own measurement** — the knobs ship default-neutral; do not re-implement them as wins. Plan B (SEVA) remains smoke-test-gated. |
| 4 | [artifixer-closeness-24gb.md](artifixer-closeness-24gb.md) | **The current actionable plan — implement from here.** Ranked closeness levers on the vendored `difix_ref`: (1) fp16, (2) multi-step denoise, (3) opacity-mix/SDEdit init via the unused `mask.downscale_to_latent`, (4) the one-line reference-pose fix, wired to exact `file:line` seams. | Current. |

Shared regime conclusion (all four docs, different words): the Difix-style fill
pays off on **small near-camera holes with frozen geometry** — i.e. sparse /
under-observed captures. On dense well-covered scenes (ufficio360: 244 views,
~30 dB) it buys nothing; **large/far empty-volume gaps with geometry allowed to
move** are ArtiFixer's 14B regime and out of 24 GB reach. Empty-region quality
has no ground truth: argue it with FID/LPIPS and eyes, never PSNR.

## VLN / embodied navigation

| Doc | What it is |
|-----|------------|
| [vln-embodied-exploration.md](vln-embodied-exploration.md) | Technique menu for instruction-following on top of the coverage explorer (topological memory + MapGPT-style NL prompting, uncertainty-driven NBV via FisherRF/GauSS-MI, frontier exploration, curiosity value maps), ranked P0–P4, training-free picks flagged. Part 3 sketches the instruction-following recipe the `navigate` mode builds on. |
| [task-pipeline-survey.md](task-pipeline-survey.md) | Survey behind ADR-0010: task formalisms, open-vocab 3D perception, LLM task generation, auto-eval (success tolerances, SPL, partial credit) for pick/place tasks on 3DGS scenes at 200k scale. |

## Misc

| Doc | What it is |
|-----|------------|
| [gaussian-robot-deck.html](gaussian-robot-deck.html) | Presentation deck for the whole project (23 slides, self-contained HTML). |

## Conventions

New research passes get a new dated doc rather than editing an old one; when a
measurement kills a recommendation, the doc records it in-place as an Update
section and this index marks the doc superseded. Decisions that graduate from
research become ADRs in [`docs/adr/`](../adr/).
