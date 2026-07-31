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

**Summary:** CRKD uses a BEVFusion-style LiDAR-camera teacher to supervise a cheaper camera-radar student. The student has gated BEV fusion, while cross-stage radar, feature, relation, and response distillation losses transfer geometry and detection knowledge. LiDAR appears during teacher training and distillation, not student inference.

This is a concrete version of privileged sensing: development-fleet hardware raises the supervision ceiling without entering the deployed bill of materials.

## Paper Insights

Cross-stage distillation aligns radar features with the teacher's LiDAR representation; mask scaling concentrates feature loss near foreground; relation loss preserves spatial structure; response loss transfers predictions with dynamic class weighting. The reported camera-radar baseline is 43.2 mAP / 54.1 NDS, gated fusion reaches 44.9 / 55.9, and full CRKD reaches 46.7 / 57.3. The teacher reports 66.1 mAP.

| Knowledge path | Purpose | Risk |
| --- | --- | --- |
| LiDAR to radar | Stronger metric geometry | Student cannot reproduce all teacher cues. |
| Feature masking | Focus on objects | Background/map knowledge may be lost. |
| Relation loss | Preserve BEV structure | Adds pairwise optimization complexity. |
| Response loss | Transfer class/box output | Inherits teacher bias and confidence. |

## Decision Lens

CRKD is useful when the production sensor set is fixed but richer offline sensors are available. The proper control is the identical student trained directly, with teacher confidence, adverse-weather slices, and distillation weight ablations reported separately.

Knowledge distillation adds no student inference block, but it adds a teacher lifecycle: versioning, target regeneration, calibration, and bias monitoring.

**Context:** BEVDepth uses LiDAR as a depth label for camera inference; CRKD transfers a broader camera-LiDAR representation into camera-radar perception.

**Takeaway:** Train-time LiDAR can improve a cheaper runtime model, provided the teacher remains a supervised data dependency rather than a hidden inference dependency.
