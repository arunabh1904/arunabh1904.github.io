---
title: 'GAFusion: Adaptive LiDAR-Camera Fusion'
date: '2024-06-17T04:00:00.000Z'
section: paper-shorts
postSlug: gafusion-adaptive-lidar-camera-fusion
legacyPath: /paper shorts/2024/06/17/gafusion-adaptive-lidar-camera-fusion.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2024 – GAFusion: guide camera BEV with LiDAR depth, occupancy, scale, and time'
---
## 2024 – GAFusion

**Paper:** [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Li_GAFusion_Adaptive_Fusing_LiDAR_and_Camera_with_Multiple_Guidance_for_CVPR_2024_paper.html)

### Method and reported result

GAFusion uses LiDAR to guide camera geometry through Sparse Depth Guidance and LiDAR Occupancy Guidance, then applies local-to-global adaptive fusion, multi-scale processing, and a two-frame temporal path. It combines several known bottleneck fixes in one LiDAR-camera BEV detector.

## Summary

The architecture's unifying idea is guidance: LiDAR should shape where and at what scale camera evidence enters BEV before the modalities are blended.

## Core Insights

On nuScenes, the paper reports 72.1 mAP / 73.5 NDS on validation and 73.6 / 74.9 on test. In its ablations, combining depth and occupancy guidance adds about 1.4 mAP and 0.8 NDS; adaptive local-global fusion improves over addition or concatenation; the temporal module contributes about 0.3 mAP and 0.1 NDS.

| Component | Intended repair | Reported signal |
| --- | --- | --- |
| Sparse Depth Guidance | Camera range ambiguity | Positive isolated mAP/NDS gain. |
| LiDAR Occupancy Guidance | Empty/occupied localization | Complementary to depth. |
| Local-global fusion | Misalignment and context | Beats simple merge operators. |
| Multi-scale + temporal | Small actors and short history | Incremental cost and gain. |

## High-Level Takeaways

- GAFusion is useful as a recipe study, but its many interacting blocks make attribution the central question. Reproduce each component at matched parameters and latency, then test whether gains survive calibration error, LiDAR sparsity, and sensor dropout.
- The published test score is not evidence that every production system should adopt the full stack. Guidance modules also strengthen dependence on healthy LiDAR unless a fallback path is trained.
- BEVFusion establishes the dense shared grid; GAFusion adds explicit geometric guidance, adaptive interaction, scale, and short temporal context.
- A high-performing fusion stack often combines several small, targeted corrections; deployment should retain only the corrections that survive matched and degraded-mode tests.
