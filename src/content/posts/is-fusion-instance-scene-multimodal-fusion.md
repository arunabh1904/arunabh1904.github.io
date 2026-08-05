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
## 2024 – IS-Fusion

**Paper:** [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_IS-Fusion_Instance-Scene_Collaborative_Fusion_for_Multimodal_3D_Object_Detection_CVPR_2024_paper.html)

**Code:** [yinjunbo/IS-Fusion](https://github.com/yinjunbo/IS-Fusion)

### Method and reported result

IS-Fusion argues that one fusion granularity cannot serve both scene context and object detail. Hierarchical Scene Fusion builds a dense multimodal scene representation, while Instance-Guided Fusion forms proposal-level regions and lets instance features retrieve relevant image and point-cloud evidence.

## Summary

The two paths answer different questions: where is the scene structure, and what evidence belongs to this candidate object?

## Core Insights

Scene fusion uses hierarchical interaction to strengthen BEV context; instance fusion combines point-to-grid and grid-to-region transformations around proposals. On nuScenes validation, the full model reports 72.8 mAP and 74.0 NDS. The paper's ablation reports simple image-LiDAR fusion at 69.4/71.6, scene fusion adding about 2.2 mAP and 1.6 NDS, and the instance path contributing further gains.

| Fusion scale | Shared unit | Strength | Risk |
| --- | --- | --- | --- |
| Scene | Dense BEV cells | Maps context across the frame | Pays for empty/background area. |
| Instance | Proposals and regions | Preserves object-specific detail | Depends on proposal recall. |
| Hierarchy | Multi-scale features | Handles small and large actors | More alignment points. |
| Collaboration | Both paths | Complementary evidence | Harder attribution and deployment. |

## High-Level Takeaways

IS-Fusion is relevant when dense BEV fusion localizes the scene but loses fine object evidence. The decisive control matches total parameters against a stronger scene-only encoder and measures gains by range, size, occlusion, and proposal confidence.

Its latency should be profiled by number of proposals and active regions, not only average FPS, because crowded scenes expand instance work.

BEVFusion chooses dense scene fusion; TransFusion chooses object queries; IS-Fusion deliberately pays for both.

Scene cells and object proposals carry complementary information, but a dual path earns its complexity only when matched ablations show each granularity fixes a distinct failure.
