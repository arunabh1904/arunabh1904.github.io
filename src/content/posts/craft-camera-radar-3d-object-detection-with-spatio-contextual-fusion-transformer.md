---
title: 'CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer'
date: '2022-09-14T04:00:00.000Z'
section: paper-shorts
postSlug: craft-camera-radar-3d-object-detection-with-spatio-contextual-fusion-transformer
legacyPath: /paper shorts/2022/09/14/craft-camera-radar-3d-object-detection-with-spatio-contextual-fusion-transformer.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2022 – CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer'
---
## 2022 – CRAFT

**arXiv:** [2209.06535](https://arxiv.org/abs/2209.06535)

### Method and reported result

CRAFT starts from camera-generated 3D proposals, associates nearby radar returns in polar coordinates, and lets radar-to-image and image-to-radar cross-attention refine each proposal. The association is soft enough to tolerate image-depth and radar-angle uncertainty, but sparse enough to avoid letting every proposal attend to every return.

## Summary

> The key modeling choice is coordinate-aware correspondence. Camera localization error is anisotropic: depth is much less certain than azimuth. Polar association expresses that structure more naturally than a Cartesian ball.

## Core Insights

Soft Polar Association builds a proposal-specific radar set. A spatio-contextual fusion transformer then exchanges position and feature evidence between image tokens and radar returns. On nuScenes test, the paper reports 41.1 mAP and 52.3 NDS. Relative to its camera-only CRAFT-I baseline on validation, radar adds 7.9 mAP overall, with larger gains on reflective metal classes and distant objects.

![CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer source figure: Overall architecture of CRAFT.](/assets/images/craft-camera-radar-3d-object-detection-with-spatio-contextual-fusion-transformer-paper-figure.webp)
*Overall architecture of CRAFT. source: [CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer](https://arxiv.org/abs/2209.06535)*

![Figure 5 from CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer](/assets/images/craft-camera-radar-3d-object-detection-with-spatio-contextual-fusion-transformer-source-figure-5.webp)
*Figure 5 Qualitative results of CRAFT. Blue circles indicate samples that are refined by fusing radar points and have more accurate localization, and red circles indicate samples that are predicted by camera-only since there are no valid radar returns among associated points. Best viewed in color with zoom in. source: [CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer](https://arxiv.org/abs/2209.06535)*

![Figure 4 from CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer](/assets/images/craft-camera-radar-3d-object-detection-with-spatio-contextual-fusion-transformer-source-figure-4.webp)
*Figure 4 Analysis of different object distances and the number of radar points. source: [CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer](https://arxiv.org/abs/2209.06535)*


| Ablation | Car AP | Association recall | What it isolates |
| --- | ---: | ---: | --- |
| RoI pooling | 29.8 | 50.7 | Hard image regions miss valid returns under depth error. |
| Cartesian ball query | 39.5 | 78.2 | Broad neighborhoods recover returns but include clutter. |
| Soft Polar Association | 41.3 | 77.1 | Polar uncertainty improves the precision-recall tradeoff. |

Replacing naive concatenation with the fusion transformer adds 2.4 car AP. Polar coordinates add 13.6 AP over the reported Cartesian setup and reduce both radial and azimuth error. For objects beyond 35 m, fusion yields a 32.2% relative improvement in the paper's analysis; with few or no valid radar points, the gain is much smaller.

## High-Level Takeaways

- CRAFT informs whether camera-radar fusion should occur around object proposals instead of a dense BEV. Its atomic unit is an image proposal paired with a variable radar set. The camera supplies semantics and an initial spatial hypothesis; radar is asked to correct range and motion where returns exist.
- The rejection test compares proposal fusion with radar-guided BEV lifting under the same camera detector, sweep count, range buckets, and corruption suite. CRAFT loses if proposal errors prevent the relevant radar return from entering the association set or if a dense radar representation recovers weak actors without camera proposals. At crowded range, proposal-return pairing and repeated cross-attention are the likely scaling bottlenecks.
- PointPainting hard-projects semantics onto LiDAR points. CRAFT instead uses camera proposals as the sparse fusion unit. CRN and RCBEVDet later move camera-radar interaction into BEV and strengthen radar-specific encoding.
- Radar fusion improves when the association rule reflects how each sensor is uncertain, not merely where their nominal coordinates coincide.
