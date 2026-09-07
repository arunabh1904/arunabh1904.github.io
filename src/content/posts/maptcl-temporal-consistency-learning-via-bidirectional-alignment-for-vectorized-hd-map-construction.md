---
title: "MapTCL: Temporal Consistency Learning via Bidirectional Alignment for Vectorized HD Map Construction"
date: '2026-08-05T00:00:00.000Z'
section: paper-shorts
postSlug: maptcl-temporal-consistency-learning-via-bidirectional-alignment-for-vectorized-hd-map-construction
legacyPath: /paper shorts/2026/08/05/maptcl-temporal-consistency-learning-via-bidirectional-alignment-for-vectorized-hd-map-construction.html
tags:
  - Autonomous Driving
  - HD Maps
  - Temporal Consistency
field: 'BEV Perception & Mapping'
summary: "2026 – MapTCL: Temporal Consistency Learning via Bidirectional Alignment for Vectorized HD Map Construction"
---

**arXiv:** [2608.05209](https://arxiv.org/abs/2608.05209)

## Summary

> MapTCL teaches an online vector mapper to disagree less with its own recent predictions. During training, it aligns historical and current map elements in both directions, penalizing coordinate and class-distribution changes; a raster consistency loss supplies a dense counterpart. These auxiliary branches disappear at inference, preserving the baseline's existing temporal architecture. On geographically non-overlapping splits, StreamMapNet gains 3.7 mAP and 2.8 consistency-aware mAP on nuScenes, with a clear limit: unreliable associations and overly long training histories can make consistency supervision worse.

## Core Insights

### Temporal features do not explicitly penalize temporal disagreement

A mapper can ingest previous BEV features yet optimize every output only against the ground truth at that timestamp. This encourages per-frame accuracy without directly asking whether the same road boundary has changed shape between frames. Occlusion can then produce a plausible map at each instant while dividers flicker or shift over time.

MapTCL retains ordinary supervised mapping losses and adds prediction-to-prediction supervision. Its main baseline is StreamMapNet, whose BEV features already pass through temporal GRU fusion. A separate training buffer stores vector predictions and raster maps from five historical frames. The new loss compares them with the current prediction after ego-motion alignment.

![MapTCL adds vector and raster consistency branches around an existing temporal mapper](/assets/images/maptcl-temporal-consistency-learning-via-bidirectional-alignment-for-vectorized-hd-map-construction-source-figure-2.webp)
*Fig 1: The orange and blue branches compare current predictions with warped history during training. Removing them at inference leaves the central mapper, including its original GRU-based temporal fusion, intact. | source: [MapTCL, Figure 2](https://arxiv.org/abs/2608.05209)*

“No inference overhead” therefore means no additional serving machinery relative to the chosen baseline. It does not mean that the deployed system becomes stateless or that consistency matching is free during training.

### Matching in both directions avoids making history the sole authority

Bidirectional Temporal Matching first selects confident historical instances and warps them into the current ego frame. Hungarian matching pairs them with current predictions using class-distribution and position costs, followed by point-order matching within each instance. The process is repeated in the opposite direction using confident current elements and the inverse pose transformation.

The two directions need not select identical pairs. A boundary confidently visible now might have been uncertain behind a vehicle earlier. Treating the old prediction as an unquestioned teacher would preserve its omission. Bidirectional losses allow both sets of confident observations to contribute, although a confidently wrong association can still propagate an error.

The vector objective uses SmoothL1 coordinate differences and KL divergence between matched class distributions. Raster Consistency Learning adds ground-truth-supervised binary segmentation and a focal-loss comparison with warped historical raster maps. Recent frames receive higher weights. The confidence threshold is 0.3 in the default setting, and consistency learning begins only in a later training stage, after predictions become suitable for matching.

### The gains persist when train and validation locations are separated

The main comparison uses non-overlapping geographic splits, a 60×30 m range, and one historical fused frame for the reproduced temporal baselines. Five frames in the *training-loss buffer* are a separate setting from that inference fusion length.

| StreamMapNet comparison | mAP before / after | C-mAP before / after | FPS before / after |
| --- | ---: | ---: | ---: |
| nuScenes | 35.2 / 38.9 | 25.7 / 28.5 | 14.5 / 14.5 |
| Argoverse2 | 53.6 / 56.7 | 35.4 / 37.9 | 14.9 / 14.9 |

Ordinary mAP scores per-frame vector detection; C-mAP also accounts for consistent detection and tracking of the same elements. The two measures improve together here. Applied to MapTracker, the non-overlapping nuScenes result rises from 39.4/30.8 to 41.0/32.4 mAP/C-mAP, supporting use beyond one base architecture.

The original geographically overlapping split gives much higher absolute nuScenes scores: StreamMapNet reaches 59.3 mAP before MapTCL and 62.1 afterward. Those numbers should not be mixed with the 35.2→38.9 comparison. Geography changes the difficulty more than the auxiliary loss changes either score.

![Successive maps under vehicle occlusion compared across temporal mappers](/assets/images/maptcl-temporal-consistency-learning-via-bidirectional-alignment-for-vectorized-hd-map-construction-source-figure-4.webp)
*Fig 2: Read downward through time, then compare the MapTCL column with ground truth. The occluding truck changes visibility while the underlying map should remain stable; this is the continuity that single-frame accuracy can miss. | source: [MapTCL, Figure 4](https://arxiv.org/abs/2608.05209)*

### Consistency helps only while the paired evidence remains trustworthy

The nuScenes component ablation improves mAP from 35.2 to 35.9 with forward-only vector consistency, to 37.2 with bidirectional vector consistency, and to 38.9 with the full design. This supports the reverse association and dense raster objectives, without proving that any stable prediction is correct.

The memory ablation is sharper. On Argoverse2, five training-history frames give 56.7 mAP and 37.9 C-mAP; seven fall to 51.6 and 32.5. Lowering the confidence threshold from 0.3 to 0.1 drops C-mAP to 23.8. Additional history and more matches can amplify stale or unreliable predictions.

Ego pose is another dependency. StreamMapNet and SQD-MapNet experiments assume accurate alignment poses, while MapTracker adds pose noise during training. The occlusion subset selects scenes with dynamic objects within five meters of the ego vehicle; it is a useful stress slice, not a comprehensive occlusion or map-change benchmark.

## High-Level Takeaways

- Training a mapper on history and explicitly supervising temporal consistency are distinct choices.
- Bidirectional matching reduces dependence on one historical teacher, while confidence and pose quality still govern the supervision.
- Separate geographic split, training-buffer length, and inference fusion length when comparing results.
- The best history is the longest reliable one, not necessarily the largest buffer; the seven-frame and low-confidence ablations demonstrate the cost of bad temporal evidence.
