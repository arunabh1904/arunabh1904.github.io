---
title: 'IS-Fusion: Instance-Scene Multimodal Fusion'
date: '2024-06-17T04:00:00.000Z'
section: paper-shorts
postSlug: is-fusion-instance-scene-multimodal-fusion
legacyPath: /paper shorts/2024/06/17/is-fusion-instance-scene-multimodal-fusion.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2024 – IS-Fusion: combine dense scene fusion with proposal-level interaction'
---

**Paper:** [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_IS-Fusion_Instance-Scene_Collaborative_Fusion_for_Multimodal_3D_Object_Detection_CVPR_2024_paper.html)

**Code:** [yinjunbo/IS-Fusion](https://github.com/yinjunbo/IS-Fusion)

## Summary

> IS-Fusion argues that multimodal detection needs two fusion granularities. Hierarchical Scene Fusion builds a globally coherent BEV from point and image evidence, while Instance-Guided Fusion selects salient proposals, gathers local multimodal context, and sends the refined instance features back into the scene representation. On nuScenes validation it reaches 72.8 mAP / 74.0 NDS at 3.2 FPS; the test result is 73.0 / 75.2 without the ensemble and test-time augmentation used by its 76.5 / 77.4 variant.

## Core Insights

### Scene context and object detail fail differently

Dense BEV fusion is good at placing an object in the broader road scene, but it spends compute on empty cells and can blur the evidence that distinguishes a small or occluded object. Proposal-only fusion has the opposite problem: it can preserve object detail while missing the lane, neighboring vehicles, or contextual cues that make a proposal plausible. IS-Fusion keeps both paths and makes them collaborate.

The Hierarchical Scene Fusion (HSF) module starts with a VoxelNet LiDAR BEV and Swin-T image features. Point-to-Grid projects the points inside each pillar into the image views, bilinearly samples image features, and uses self-attention before max pooling them into a grid cell. This compares several points within a pillar, which helps absorb calibration noise instead of trusting one projection. Grid-to-Region then uses local and shifted region attention to exchange information across BEV cells without global attention over every cell.

![IS-Fusion scene and instance collaboration pipeline](/assets/images/is-fusion-paper-figure.webp)
*Fig 1: IS-Fusion builds hierarchical scene context, selects top-K instance candidates, aggregates local multimodal evidence, and propagates the refined instances back into BEV; the displayed framework is the paper’s Figure 2. | source: [IS-Fusion: Instance-Scene Collaborative Fusion for Multimodal 3D Object Detection, Figure 2](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_IS-Fusion_Instance-Scene_Collaborative_Fusion_for_Multimodal_3D_Object_Detection_CVPR_2024_paper.html)*

### Instance fusion is a feedback path, not a second detector

Instance-Guided Fusion (IGF) reads a BEV heatmap and keeps the top K=200 candidates. Each instance query aggregates information from neighboring instances and samples D=16 locations from the multimodal feature. Cross-attention lets the proposal retrieve image and point evidence around its own geometry; an Instance-to-Scene transformer then writes the refined instance context back into the scene BEV before the final decoder.

This ordering explains the complementary ablation. On nuScenes validation, a LiDAR-only baseline is 65.4 mAP / 70.1 NDS. Simple LiDAR-camera fusion reaches 69.4 / 71.6. HSF alone reaches 71.6 / 73.2, while IGF alone reaches 70.9 / 72.8; the full HSF+IGF model reaches 72.8 / 74.0. The scene path supplies coherent context, and the instance path recovers detail around the candidates that the scene representation may have diluted.

The hyperparameter study also bounds the proposal strategy. Increasing K from 64 to 200 helps, while K=300 does not; D=16 outperforms D=32. The paper’s interpretation is that instance-to-instance attention has already found a sufficient receptive field, so more sampled regions add work without adding useful evidence.

### Leaderboard numbers need their inference recipe

The nuScenes split has 700 training, 150 validation, and 150 test scenes with six calibrated cameras. IS-Fusion uses 384×1056 images, 0.075×0.075×0.2 m voxels, a 180×180 BEV feature map, and trains end-to-end for 10 epochs with AdamW and a maximum learning rate of 10⁻³. The validation comparison reports 72.8 mAP / 74.0 NDS and 3.2 FPS, comparable to the listed multimodal systems.

On the test set, the base IS-Fusion model reports 73.0 / 75.2, while IS-Fusion† reports 76.5 / 77.4 with model ensemble and test-time augmentation. The base number is the cleaner architecture comparison. The † number is useful as a leaderboard operating point, but it combines multiple models, point-cloud rotations, and flips, so it should not be read as a single-model latency result.

## High-Level Takeaways

- IS-Fusion pays for both dense scene context and proposal-centered detail because each resolves a different failure mode of multimodal BEV fusion.
- Point-to-Grid compares several projected points before aggregation, giving the scene path a way to tolerate calibration noise and preserve local evidence.
- Instance-to-Scene feedback makes IGF part of the scene representation rather than an isolated late-fusion head.
- The validation ablation supports both paths, while the test-time ensemble result should remain separate from the single-model comparison.
- A production decision should measure proposal count, active-region latency, calibration drift, and gains by object size, range, occlusion, and candidate recall.
