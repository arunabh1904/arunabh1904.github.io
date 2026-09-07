---
title: 'CRKD: Camera-Radar Distillation from LiDAR-Camera'
date: '2024-06-17T04:00:00.000Z'
section: paper-shorts
postSlug: crkd-camera-radar-distillation-from-lidar-camera
legacyPath: /paper shorts/2024/06/17/crkd-camera-radar-distillation-from-lidar-camera.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2024 – CRKD: train a camera-radar student with a stronger LiDAR-camera teacher'
---
## 2024 – CRKD

**Paper:** [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_CRKD_Enhanced_Camera-Radar_Object_Detection_with_Cross-modality_Knowledge_Distillation_CVPR_2024_paper.html)

**Code:** [robotics0105/CRKD](https://github.com/robotics0105/CRKD)

### Method and reported result

CRKD uses a BEVFusion-style LiDAR-camera teacher to supervise a cheaper camera-radar student. The student has gated BEV fusion, while cross-stage radar, feature, relation, and response distillation losses transfer geometry and detection knowledge. LiDAR appears during teacher training and distillation, not student inference.

## Summary

> This is a concrete version of privileged sensing: development-fleet hardware raises the supervision ceiling without entering the deployed bill of materials.

## Core Insights

Cross-stage distillation aligns radar features with the teacher's LiDAR representation; mask scaling concentrates feature loss near foreground; relation loss preserves spatial structure; response loss transfers predictions with dynamic class weighting. The reported camera-radar baseline is 43.2 mAP / 54.1 NDS, gated fusion reaches 44.9 / 55.9, and full CRKD reaches 46.7 / 57.3. The teacher reports 66.1 mAP.

The ablation shows why the losses are not interchangeable. Adding response distillation after the gated student moves the score to 45.7/56.7, the largest single step; adding CSRD and MSFD reaches 46.0/57.0 and 46.2/57.2, while relation distillation closes at 46.7/57.3. The per-class table gives the mechanism a useful shape: relative to the 43.2-mAP baseline, truck improves by 6.3 AP, bus by 4.7, motor by 4.8, and bicycle by 4.4. Those are radar-friendly dynamic categories, consistent with CRKD's choice to weight dynamic response targets by 2 and static targets by 1. The gain is therefore a transfer of radar's velocity-sensitive advantage, not uniform recovery of the teacher's LiDAR geometry.

![CRKD teacher-student framework transferring LiDAR-camera knowledge into a camera-radar detector](/assets/images/crkd-camera-radar-distillation-paper-figure.webp)
*Fig 1: The runtime student keeps only camera and radar; LiDAR appears in the teacher and in feature-, relation-, and response-level training losses. | source: [CRKD](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_CRKD_Enhanced_Camera-Radar_Object_Detection_with_Cross-modality_Knowledge_Distillation_CVPR_2024_paper.html)*

| Knowledge path | Purpose | Risk |
| --- | --- | --- |
| LiDAR to radar | Stronger metric geometry | Student cannot reproduce all teacher cues. |
| Feature masking | Focus on objects | Background/map knowledge may be lost. |
| Relation loss | Preserve BEV structure | Adds pairwise optimization complexity. |
| Response loss | Transfer class/box output | Inherits teacher bias and confidence. |

## High-Level Takeaways

- CRKD is useful when the production sensor set is fixed but richer offline sensors are available. The proper control is the identical student trained directly, with teacher confidence, adverse-weather slices, and distillation weight ablations reported separately.
- Knowledge distillation adds no student inference block, but it adds a teacher lifecycle: versioning, target regeneration, calibration, and bias monitoring.
- BEVDepth uses LiDAR as a depth label for camera inference; CRKD transfers a broader camera-LiDAR representation into camera-radar perception.
- Train-time LiDAR can improve a cheaper runtime model, provided the teacher remains a supervised data dependency rather than a hidden inference dependency.
