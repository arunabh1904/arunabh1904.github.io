# November 3 queued-paper release audit

Scope: eight explicitly selected Paper Radar notes and Ben Kuhn's “Impact, agency, and taste” in the Reading List's Writing & craft group. Queue filenames, dates, slugs, and public routes are preserved. HENet++ retains its 2025 original-paper year while the note covers the October 2026 revision. Other local drafts, the separate seminal-paper list, and VQ-Map are outside this publication.

Canonical full texts, method details, evaluation tables, figure captions, and relevant technical appendices were reviewed. Seven notes cover v1; HENet++ covers v2, including its added NAVSIM evaluation, revised perception scores, RayIoU, and repeated initialization ablation. The notes distinguish paper claims from proposed follow-up experiments and retain material negative results.

Each note contains one attributed source architecture figure downloaded from the corresponding arXiv HTML and converted to WebP without changing its content. Full-resolution links preserve access to dense source labels on small screens. Source licenses reported by arXiv are CC BY 4.0 for AUGUstus, CC BY-NC-SA 4.0 for HENet++, and the arXiv perpetual non-exclusive distribution license for the other six papers. Source authors retain their figure rights; conversion does not grant a new license to the images.

## Review ledger

| Note / canonical source | Substantive coverage and boundary | Source figure |
| --- | --- | --- |
| [CLAP-AD, v1](https://arxiv.org/html/2610.36079v1) | Path imitation versus rule-derived speed labels; detached criterion-specific heads; backbone-specific training; separate Bench2Drive and Speed suites; TFv6 regression; NAVSIM progress/comfort trade-off. | Figure 2: path and scoring branches. |
| [FastBEVFusion, v1](https://arxiv.org/html/2610.37153v1) | Uniform-depth projection; LiDAR-query fusion; three training stages; backbone-dependent throughput; head and association ablations; hardware comparability and reversed calibration-table metrics. | Figure 2: sensor branches and fusion. |
| [Scene-Adaptive VLA, v1](https://arxiv.org/html/2610.24590v1) | Historical complexity tokens; prefix and routed layers; supervised routing then loss-based GRPO budgets; latency versus memory; route, language, and emergency-braking regressions. | Figure 2: complexity and routing. |
| [Lane-Structured Conditioning Fields, v1](https://arxiv.org/html/2610.25284v1) | Unoriented lane tensor and confidence; field losses; feature modulation and attention guidance; OpenLane-V2 subset and optional priors; topology and structural limits. | Figure 2: field-conditioned mapping. |
| [SafeDriveVLA, v1](https://arxiv.org/html/2610.20609v1) | World-model training and freezing; maneuver anchors and future-token projector; VLA action interface and rule-derived modes; construction of four benchmark suites; compliance/conflict separation and speed regressions. | Figure 6: dreaming, mode, and actions. |
| [ConDenseDrive, v1](https://arxiv.org/html/2610.16908v1) | Three-teacher distillation; register-based planning; LoRA adaptation and staged token compression; training schedules; timing boundary; HUGSIM zero-shot evidence and extreme-case regression. | Figure 2: three-stage visual encoder. |
| [AUGUstus, v1](https://arxiv.org/html/2610.16985v1) | Action-first attention; continuous action interface and language supervision; SFT and two-stage joint GRPO; target-content ablations; CUDA-graph latency; public versus private-data evidence. | Figure 2: action-first shared model. |
| [HENet++, v2](https://arxiv.org/html/2511.07106v2) | Hybrid temporal encoding; sparse/dense tasks and model merging; planning losses and radar; protocol-specific collision/L2 trade-offs; revised NAVSIM and perception results; heterogeneous compute settings. | Figure 2: revised hybrid architecture. |

## Validation

- `git diff --check` passed.
- `npm run ci` passed: Astro check reported zero errors and warnings; all 780 tests in 16 files passed; the build emitted 458 pages; verification checked critical pages, legacy routes, and 370 paper reading paths.
- All eight rendered routes were inspected at 390-pixel and 1440-pixel viewport widths. Every source image loaded at its intrinsic resolution, no page-level horizontal overflow appeared, and no KaTeX error nodes were present. Captions and full-resolution image links rendered correctly.
- The Reading List expanded correctly and showed Ben Kuhn directly after Shreya Shankar in Writing & craft.
- Release state at commit: prepared and locally verified. PR checks, merge, deployment, live route verification, and authoritative queue completion are recorded in the release report after shipping.
