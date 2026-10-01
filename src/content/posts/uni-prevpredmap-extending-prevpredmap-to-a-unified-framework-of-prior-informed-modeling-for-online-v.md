---
title: "Uni-PrevPredMap: A Unified Framework of Prior-Informed Online Vectorized HD Mapping"
date: '2026-09-18T00:00:00.000Z'
section: paper-shorts
postSlug: uni-prevpredmap-extending-prevpredmap-to-a-unified-framework-of-prior-informed-modeling-for-online-v
legacyPath: /paper shorts/2026/09/18/uni-prevpredmap-extending-prevpredmap-to-a-unified-framework-of-prior-informed-modeling-for-online-v.html
tags: [Autonomous Driving, Mapping]
topics: [autonomy]
field: Mapping
summary: "2026 \u2013 Uni-PrevPredMap: A Unified Framework of Prior-Informed Online Vectorized HD Mapping"
---

# 2026 – Uni-PrevPredMap: A Unified Framework of Prior-Informed Online Vectorized HD Mapping

**Paper:** [2504.06647v4](https://arxiv.org/abs/2504.06647v4)

**Code:** [official implementation](https://github.com/pnnnnnnn/Uni-PrevPredMap)

This note covers the September 18, 2026 v4 revision, which extends the original April 2025 PrevPredMap work. The later title and experiments should not be attributed unchanged to v1.

## Summary

> Uni-PrevPredMap trains one mapper under absent, temporal, and temporal-plus-HD-map prior conditions. It stores predictions and external map vectors in a tile-indexed representation and conditions both BEV features and query generation on retrieved priors. On nuScenes, the same model reaches 64.9 mAP with neither prior, 74.0 with history, 71.3 with the external map alone, and 80.9 with both. Robustness is tested through synthetic corruption; the paper does not establish reliable correction of genuine historical map changes.

## Core Insights

### Two imperfect priors share an interface

Historical predictions preserve recently observed geometry. A prebuilt map supplies structure the vehicle has not yet seen. Uni-PrevPredMap represents both as 3D vectors in spatial tiles, retrieves nearby tiles using vehicle position, filters to the perception range, and rasterizes them into separate prior heatmaps. High-confidence current predictions refresh the temporal representation.

The figure shows the same retrieved information entering two places. Convolutional fusion augments BEV features, while deformable attention initializes map queries from prior features. The model learns their influence through task supervision rather than a hand-chosen reliability score.

![Uni-PrevPredMap source Figure 2 shows tri-mode training, tile retrieval, BEV conditioning, and query generation](/assets/images/uni-prevpredmap-source-figure.png)
*Fig 1: Temporal and external map vectors share retrieval and rasterization, then condition the BEV encoder and query generator. Training also supplies empty priors so the mapper learns an independent perception path. | source: [Uni-PrevPredMap v4, Figure 2; figure crop](https://arxiv.org/abs/2504.06647v4)*

This differs from [MapTracker](/paper%20shorts/2024/03/23/maptracker-tracking-with-strided-memory-fusion-for-consistent-vector-hd-mapping.html)'s persistent vector latents and explicit track supervision. Uni-PrevPredMap carries predicted geometry through a rasterized prior. Its authors identify that representation as a possible reason for weaker crossing AP than MapTracker when external maps are absent. A common interface simplifies reuse but can discard instance detail.

### Prior availability becomes a training condition

Training alternates among no-prior, temporal-only, and temporal-plus-map modes. The main nuScenes ratio is 0.50:0.30:0.20. External map vectors receive instance-level displacement sampled from −6 to 6 meters on selected elements. Empty heatmaps represent unavailable priors. At inference, the model chooses the available mode instead of requiring continuous map coverage.

The implementation uses ResNet-50, LSS/BEVPoolv2, 0.3-meter BEV cells, 100 instance queries, 20 point queries, and six decoder layers. AdamW training uses batch size 16 on four A100 GPUs. The principal nuScenes comparison uses 24 epochs; a longer schedule uses 72. Argoverse 2 adds elevation-aware map evaluation under its own sampling and training setup.

| Same model, nuScenes prior condition | mAP | FPS |
| --- | ---: | ---: |
| Neither prior | 64.9 | 14.2 |
| Temporal only | 74.0 | 12.2 |
| External map only | 71.3 | 13.1 |
| Both | 80.9 | 11.5 |

“Map-absent” in the headline therefore means temporal-only, not stateless perception. The training-mode ablation makes dependence visible: a single-mode model reaches 82.9 with the map but only 16.6 without it. Tri-mode training gives up some map-present performance to preserve useful operation when the map disappears.

### Unseen perturbations are still synthetic

The robustness table adds corruption types absent from training: element insertion/deletion and frame-level translation, rotation, and scaling. Perfect-map mAP is 85.6; ±9-meter global displacement yields 74.6; an empty external map yields 74.0. Global displacement damages the result more than the tested instance-level displacement because the whole prior is misregistered.

The qualitative outdated-map example uses a mirrored ground-truth map, not a historical map collected before a real road modification. It is a useful structured contradiction but cannot close the synthetic-to-real gap exposed by [the real-change study](/paper%20shorts/2024/06/04/real-world-map-change-generalization.html). Likewise, the paper's vector AP does not directly score a directed lane graph or permanent change decisions.

The expensive choice is training one model to retain independent perception while exploiting several prior conditions. My next test would keep that architecture and compare mode schedules on real stale maps, geographically held-out roads, and unavailable tiles. Gains that require synthetic priors derived from current labels would not establish the same benefit with a coarse SD road skeleton.

## High-Level Takeaways

- Distinguish no external map from no prior of any kind; the reported temporal baseline retains history.
- Training under varied prior availability prevents a severe missing-map collapse in this experiment.
- Tile-indexed vectors and raster conditioning simplify reuse but differ from instance-memory tracking.
- Synthetic corruption and mirrored maps support a bounded robustness claim, not verified real-world maintenance.
