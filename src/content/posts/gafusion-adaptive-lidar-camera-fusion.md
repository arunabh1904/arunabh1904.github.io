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

**Paper:** [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Li_GAFusion_Adaptive_Fusing_LiDAR_and_Camera_with_Multiple_Guidance_for_CVPR_2024_paper.html)

## Summary

> GAFusion treats LiDAR as a geometric guide for the camera branch before the two BEV streams are fused. Sparse Depth Guidance injects projected LiDAR depth into image features, LiDAR Occupancy Guidance shapes the camera 3D volume, MSDPT enlarges camera receptive fields, and LGAFT learns local-to-global fusion weights. The full stack reaches 73.6 mAP / 74.9 NDS on nuScenes test without test-time augmentation or ensemble, while the ablations show that guidance drives most of the gain and the two-frame temporal path is nearly neutral in this setup.

## Core Insights

### Guide the camera before asking it to fuse

Camera BEV features have semantic richness but uncertain depth. GAFusion addresses that ambiguity twice. Sparse Depth Guidance projects LiDAR points into the image branch, giving 2D features a sparse metric anchor before view transformation. LiDAR Occupancy Guidance uses a LiDAR-derived 3D occupancy volume to guide the camera feature volume after view transformation. The two operations act at different places: one tells image features where observed depth lies; the other gives the lifted 3D representation a spatial occupancy prior.

The LiDAR stream uses VoxelNet with additional downsampling and sparse height compression. The camera stream uses Swin-T/FPN at 448×800, with 0.075×0.075×0.2 m voxels in the LiDAR branch. The paper’s architecture figure makes the information flow explicit: LiDAR features guide the camera before LGAFT sees the two BEV streams.

![GAFusion architecture with sparse depth, occupancy guidance, multi-scale processing, adaptive fusion, and temporal memory](/assets/images/gafusion-adaptive-lidar-camera-fusion-paper-figure.webp)
*Fig 1: GAFusion guides the camera stream with sparse depth and LiDAR occupancy, enlarges its receptive field with MSDPT, and fuses the resulting BEV features with LGAFT; the displayed architecture is the paper’s Figure 2. | source: [GAFusion: Adaptive Fusing LiDAR and Camera with Multiple Guidance for 3D Object Detection, Figure 2](https://openaccess.thecvf.com/content/CVPR2024/html/Li_GAFusion_Adaptive_Fusing_LiDAR_and_Camera_with_Multiple_Guidance_for_CVPR_2024_paper.html)*

### Receptive field and fusion are separate choices

MSDPT uses dual paths over three feature scales: local windowed processing supplies nearby structure while a broader path aggregates context. The point is not just “more transformer.” The ablation reports that removing the multi-scale dual-path module costs about 0.5 mAP and 0.4 NDS, while redundant scales add computation without a clear payoff. LGAFT then expands both LiDAR and camera BEV features and predicts a sigmoid weight map, allowing different locations to favor different modalities instead of using one global addition or concatenation rule.

The guidance ablation on nuScenes validation is more revealing than the final score. The BEVFusion base is 68.52 mAP / 71.38 NDS; SDG alone reaches 71.92 / 73.39, LOG alone 72.01 / 73.45, and both 72.08 / 73.53. Their effects are complementary, but the table does not support attributing the full gain to temporal fusion. In the fusion ablation, LGAFT without temporal fusion is 72.08 / 73.53 and with the two-frame temporal path is 72.07 / 73.54. The authors describe this as partial alignment of adjacent frames and leave stronger multi-frame aggregation for future work.

### Read the headline with its training protocol

nuScenes contains 1,000 scenes with six cameras, five radars, and one LiDAR, split 700/150/150 for train/validation/test; keyframes are annotated at 2 Hz. GAFusion first trains a LiDAR detector for 20 epochs, freezes the pretrained LiDAR components, and jointly trains the proposed framework for another six epochs with AdamW, learning rate 5×10⁻⁵, weight decay 10⁻², and batch size 4 on two RTX 3090 GPUs. The reported results use no test-time augmentation or model ensemble.

On validation, GAFusion reports 72.1 mAP / 73.5 NDS. On test, it reports 73.6 / 74.9, above the listed BEVFusion 71.3 / 73.3 comparison. The result supports a carefully engineered geometry-guidance recipe; it does not show that every module remains useful under LiDAR dropout, calibration error, or a matched compute budget.

## High-Level Takeaways

- GAFusion’s central decision is to use LiDAR first as a camera-geometry teacher and only later as a fused BEV feature.
- SDG and LOG repair different depth ambiguities, and their paired ablation is stronger evidence than the full-stack leaderboard score.
- MSDPT improves context while LGAFT makes fusion spatially adaptive; these are separate design choices and should be benchmarked separately.
- The reported temporal gain is marginal in the shown two-frame ablation, so the paper does not establish a large benefit from temporal memory.
- A production evaluation should retain the guidance gains only if they survive sensor sparsity, calibration drift, dropped LiDAR frames, and matched latency comparisons.
