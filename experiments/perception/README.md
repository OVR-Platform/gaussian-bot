# Perception probes — object extraction from a real 3DGS scene

R&D prototypes (not production; excluded from lint/type gates) that test whether we can
extract a labelled 3D object inventory from a real Gaussian-Splat office scene using only the
stack we already have, and which approach to adopt. Background: ADR-0010 (task schema &
auto-eval) and `docs/research/task-pipeline-survey.md` (cited survey).

All three render perspective views from the splat at the COLMAP capture poses (RGB + aligned
depth), detect objects, lift each detection to 3D via the splat depth + pose, and fuse across
views into instances. They differ only in the detector/labeller. The scene PLY path is
hardcoded near the top of each script (the `ufficio360` test scene); edit `PLY` to re-point.

Run (v2/v3 need scipy, added ephemerally so the project deps are untouched):

```bash
uv run python experiments/perception/probe_v1_vlm_points.py
uv run --with scipy python experiments/perception/probe_v2_mask2former.py
uv run --with scipy python experiments/perception/probe_v3_sam_openvocab.py
```

## Findings (ufficio360, ~14–20 sampled views)

| probe | detector | localization (cross-view reconciliation) | vocabulary | result |
|---|---|---|---|---|
| **v1** | VLM (Qwen) center-points | poor — **24%** | open | ~50 objects, badly localized |
| **v2** | Mask2Former (COCO instance) | good — **47%** | closed/wrong | 17 objects, hallucinates `oven`/`fridge`/`person` on office furniture |
| **v3** | **SAM masks + open-vocab VLM labels** | **good** | **open** | **~47 stable instances, correct labels** |

**v3 is the synthesis** (ConceptGraphs/OpenGaussian pattern): class-agnostic masks give the
localization, open-vocab labelling gives office-wide breadth. Stable (≥2-view) inventory
included `Glass door`, `structural column`, `Staircase with railing`, `vertical blinds`,
`Emergency Exit Sign`, `flat screen monitor`, … with surface mix **34 solid / 7 glass / 6
reflective**.

Two empirical conclusions that re-shaped the plan:

1. **Glass is NOT the bottleneck on this scene.** It is frosted/privacy-film, so 3DGS
   reconstructs it as a quasi-opaque surface — depth is finite and stable (CoV 0.075 vs 0.044
   solid, 100% liftable). The survey's "glass = no depth" warning applies to *clear* glass.
2. **The real bottleneck is localization + open vocabulary**, jointly. Neither a VLM point
   (open but imprecise) nor closed-vocab masks (precise but blind to offices) suffices; the
   SAM-masks + open-vocab-labels combination does.

### Known limits (for productionizing, not for the demo)
- SAM over-segments → duplicate/fragment instances (`ceiling panel` ×N, staircase fragments)
  and the occasional hallucinated label; needs instance dedup/merge.
- v3 labels each stable instance with a VLM call — too costly at 200k scenes; swap for CLIP
  embeddings (the ConceptGraphs approach) to scale.
- A native-3DGS extractor (OpenGaussian / GaussianGraph) would avoid the render→detect→lift
  round-trip entirely; worth evaluating next.
