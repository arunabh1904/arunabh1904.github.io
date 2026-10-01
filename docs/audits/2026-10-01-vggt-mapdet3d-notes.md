# VGGT and Map-Det3D release audit

Scope: add the missing VGGT note and expand the existing Map-Det3D note in place. Both are Markdown Arxiv Notes (`section: paper-shorts`). No duplicate Map-Det3D route was created.

## Sources and evidence boundaries

- VGGT: arXiv 2503.11651v1, full paper and technical appendix, https://arxiv.org/html/2503.11651v1. Architecture, coordinate normalization, training, all main benchmarks, ablations, IMC caveat, downstream adaptation, and runtime tables reviewed.
- Map-Det3D: arXiv 2608.12179v1, full paper, https://arxiv.org/html/2608.12179v1. Architecture, scale parameterization, training, per-frame/per-scene protocols, ablations, and limitations reviewed. The downloaded 18-page paper includes references but no separate technical appendix.
- Official Map-Det3D implementation pinned to commit `665e054ff4fb74e3363f72fb3f534c8addf60338`; model freezing, feature projections, query/decoder defaults, optimizer, data, and losses reviewed. The note distinguishes the default 50k-step config from the paper's 100k-step final run, including different sample-count settings.
- CA-1M original CVPR 2025 paper, Sections 3.1–3.2 and annotation figures: https://openaccess.thecvf.com/content/CVPR2025/papers/Lazarow_Cubify_Anything_Scaling_Indoor_3D_Object_Detection_CVPR_2025_paper.pdf. Explains laser-scan annotation, human/model assistance, registration, and per-frame rendering.
- VGGT: exact mixture probabilities are not reported; normalized scale is not metric scale; IMC overlaps some training scenes; optional bundle adjustment matters; large-frame runtime is backbone-only, not a subsecond end-to-end guarantee.
- Map-Det3D: indoor/class-agnostic scope, camera-conditioning defaults, missing timing GPU, variable aspect-ratio resolution, code/paper schedule mismatch, and the distinction between per-frame ScanNet200 and per-scene ScanNetV2 are explicit.

## Routes, metadata, and figures

### VGGT

- [Live note](https://arunabh1904.github.io/paper%20shorts/2025/03/14/vggt-visual-geometry-grounded-transformer.html)
- New slug: `vggt-visual-geometry-grounded-transformer`.
- Date: `2025-03-14T09:00:00.000Z`, canonical initial publication date.
- Field: `Vision Foundations`; tags: 3D Vision, Reconstruction, Vision Foundations.
- Two source figures: Figure 2 explains shared alternating attention and output heads; Figure 3 shows weak-overlap/texture examples while distinguishing plausible geometry from independently measured truth.

### Map-Det3D

- [Live note](https://arunabh1904.github.io/paper%20shorts/2026/08/12/map-det3d-metric-feed-forward-3d-reconstruction-prior-for-multi-view-3d-object-detection-from-streaming-inputs.html)
- Existing title, slug, legacy route, date `2026-08-12T00:00:00.000Z`, and `BEV Perception` field preserved.
- Tags: 3D Detection, Reconstruction Priors, Indoor Perception. Replaced misleading Autonomous Driving tag and serialized tags inline for Paper Radar compatibility.
- Two source figures: retained Figure 2 showing the metric-scale reconstruction backbone; added Figure 3 showing channel projections, detector queries, auxiliary 2D supervision, and scale-aware 3D output.
- Existing results, counterexamples, and runtime comparisons retained. Added training/implementation and benchmark-construction sections plus direct technical cross-links with VGGT.

## Validation

- Canonical `npm run ci`: type checks, 780 tests, build, and route checks passed. Final pre-push validation reruns after the camera-metadata clarification.
- Visually inspected both rendered notes at 1280px and 390px browser widths. Four images loaded, no page-level horizontal overflow or KaTeX errors. Source figures have full-resolution links for dense labels on mobile.
- Map-Det3D prose audit reports the intended expansion (+809 words before the final metadata clarification); sentence mean decreased from 17.3 to 16.7 words. Added detail is scoped to the requested architecture, recipe, and benchmark coverage.
- Source figures preserved and converted to local WebP; no generated scientific diagrams.
