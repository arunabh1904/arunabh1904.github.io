---
title: 'Cross-View Sequential Visual Localization with Spatio-Temporal Context Modeling for Autonomous Driving'
date: '2026-08-11T08:44:42.000Z'
section: paper-shorts
postSlug: cross-view-sequential-visual-localization-with-spatio-temporal-context-modeling-for-autonomous-driving
legacyPath: /paper shorts/2026/08/11/cross-view-sequential-visual-localization-with-spatio-temporal-context-modeling-for-autonomous-driving.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2026 – recurrent temporal context sharpens satellite candidates before cross-view localization refinement'
---

**arXiv:** [2608.10660](https://arxiv.org/abs/2608.10660)

## Summary

> This paper puts temporal context before cross-view refinement. A recurrent state enriches the current ground-view feature, hierarchical features then score a 19×19 satellite grid, and a mask-guided localizer regresses the final offset only inside retained cells. On CVIS, the reported mean error falls from 3.80 m for the strongest listed baseline to 1.57 m, but the gain is tied to six-frame sequences, a fixed map-crop protocol, and difficult road layouts that remain visible in field tests.

## Core Insights

### Recurrent history changes what gets refined

The input is one satellite crop and six ordered ground images. DINOv2 supplies a coarse and a fine feature pyramid. For the current ground frame, the spatio-temporal enhancer uses the previous recurrent state as Key and Value and the current feature as Query; spatial position encodings keep the temporal exchange from becoming an unstructured average. The enhanced coarse feature enters Stage 1, which classifies cells on a 19×19 satellite grid. Stage 2 receives the top-K mask and uses fine ground/satellite features to predict a local offset. A bad coarse cell cannot be repaired by a better offset regressor, so the useful place for memory is before candidate pruning.

![Cross-view sequential localization framework with recurrent context, coarse grid matching, and mask-guided refinement](/assets/images/cross-view-sequential-visual-localization-paper-figure.webp)
*Fig 1: The recurrent state improves coarse candidate selection before fine offset regression; the displayed framework is the paper’s Figure 2. | source: [Cross-View Sequential Visual Localization, Figure 2](https://arxiv.org/abs/2608.10660)*

The candidate-recall view makes the design easier to evaluate. With the full model, a top-64 mask covers 99.98% of ground-truth cells while retaining 17.73% of the search grid. The refinement stage therefore sees almost every plausible answer without evaluating all 361 cells. That is a retrieval claim, not evidence that the recurrent state has learned a vehicle motion model.

### The ablations separate representation, position, and time

The CVIS stepwise ablation starts with a 12.25 m mean error. Multi-level features reduce it to 5.92 m, a position-aware update to 4.96 m, and temporal context to 1.57 m. The full test split reports 1.57 m mean, 1.21 m median, 40.22% R@1 m, 77.51% R@2 m, and 98.99% R@5 m. The corresponding no-temporal model is 5.92 m mean and 61.95% R@5 m, while its latency is only 28.94 ms versus 97.32 ms for the full six-frame sequence on the authors’ RTX 3090 setup. The comparison makes the cost legible: history is doing more than smoothing an already-correct answer, but the recurrent branch is also the main latency increase.

| Decision | What the paper measures | Why it matters |
| --- | --- | --- |
| Candidate size | Top-64 retains 17.73% of the grid and covers 99.98% of targets | Recall must be high before local refinement can help |
| Temporal context | Full model versus no-temporal ablation | Tests where history enters, not only whether it exists |
| Target transfer | KITTI-CVL zero-shot and fine-tuned variants | Separates learned matching from domain adaptation |

### Transfer exposes the map and scene assumptions

On KITTI-CVL, direct transfer from CVIS gives 2.61 m mean error, improving to 2.27 m after target-domain fine-tuning. The real-vehicle experiment is more useful as a boundary test because the CVIS-trained model receives no additional training or update: across 1,031 sequences it reports 2.84 m mean error, 2.92 m median error, and 96.86% R@5 m. The difficult subsets are elevated roads at 3.67 m and uphill segments at 4.21 m. The field protocol also uses a low-precision GPS-centered satellite crop and RTK ground truth, so these numbers test a realistic prior-plus-refinement setup rather than localization from an unconstrained global map.

## High-Level Takeaways

- The paper’s main decision is to spend temporal compute on candidate recall, where a wrong map cell would otherwise make fine localization impossible.
- The 12.25→5.92→4.96→1.57 m ablation supports a cumulative design, but it does not isolate sequence length, map-crop error, backbone quality, and temporal recurrence under one equal-cost budget.
- The top-64 result is a practical operating point: it preserves nearly all target cells while shrinking the fine-search workload to under one-fifth of the grid.
- KITTI-CVL fine-tuning and the field subsets show that cross-view appearance transfer is still sensitive to geography, road elevation, and map ambiguity.
- A deployment study should vary GNSS-prior quality, sequence interruptions, seasonal imagery, and long-horizon drift while measuring latency against a no-history baseline.
