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

## Summary

> CRKD uses a LiDAR–camera teacher to improve a camera–radar detector, but does not ask sparse radar features to imitate dense LiDAR geometry directly. Its radar branch learns from the teacher's objectness heatmap; other losses transfer masked features, spatial relations, and detection responses. On nuScenes validation, gated fusion raises the camera–radar baseline from 43.2 to 44.9 mAP, and distillation raises it to 46.7. The teacher and distillation branches disappear at inference, while the gated student remains. The useful result is better supervision for the available sensors, not replacement of their missing measurements.

## Core Insights

### Distill the teacher's conclusion when its internal representation is a poor target

LiDAR and radar both return points, but those points carry different information. Dense LiDAR samples object surfaces; the radar input is sparse, noisy, and includes velocity measurements. Matching their intermediate features merely because both can be placed in BEV space can force radar to reconstruct geometry its measurements do not support.

CRKD's Cross-Stage Radar Distillation instead compares a learned scalar radar map with the teacher's objectness map. The teacher predicts one heatmap channel per class. After sigmoid, CRKD averages those channels into a class-agnostic spatial target. Three convolution–normalization–ReLU blocks and a final projection transform the radar features, and an L1 loss asks where object evidence should appear. This is learned feature calibration, rather than correction of the physical sensor calibration matrix.

The source-choice ablation tests that distinction directly. With response distillation included, using LiDAR features as the radar target reaches 44.9 mAP / 56.3 NDS; using the teacher heatmap reaches 46.0 / 57.0. The appendix finds smaller gains from the calibration module itself, 45.9 to 46.0 mAP, and from teacher heatmaps over ground-truth heatmaps. The stronger result is the choice of representation to transfer, rather than the size of the added calibration network.

![Complete CRKD teacher–student architecture, with radar heatmap supervision, masked features, relation loss, and response distillation](/assets/images/crkd-source-figure-2-full-architecture.png)
*Fig 1: The purple path connects an early radar representation to the teacher's later objectness heatmap. The remaining paths transfer gated camera features, fused features, spatial relations, and detection responses while ground-truth detection supervision remains active. | source: [CRKD, Figure 2](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_CRKD_Enhanced_Camera-Radar_Object_Detection_with_Cross-modality_Knowledge_Distillation_CVPR_2024_paper.pdf)*

### Shared cameras do not imply identical camera features

Teacher and student use the same camera modality and compatible BEV grids, but their camera features are gated using different companion sensors. Each gate reads concatenated modality features and applies sigmoid weights before convolutional fusion. The teacher's gated camera map can therefore carry LiDAR-informed geometry that is absent from the student's radar-informed map.

The supplementary comparison makes that distinction visible. The ungated camera feature map has strong radial structure, while the gated map exposes more scene geometry. Transferring the gated map is a way to supervise how the camera representation should use complementary sensing, even though the input camera images are shared. The numerical difference between gated and ungated feature distillation is small, so the image illustrates the mechanism rather than establishing a large independent gain.

![Ungated and gated camera BEV feature maps from the CRKD LiDAR–camera teacher](/assets/images/crkd-supplement-figure-1-gated-features.png)
*Fig 2: The teacher's camera features change after gating with LiDAR context. The right-hand map exposes clearer scene structure, explaining why an apparently camera-to-camera loss can still transfer information from another sensor. | source: [CRKD supplement, Figure 1](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Zhao_CRKD_Enhanced_Camera-Radar_CVPR_2024_supplemental.pdf)*

Mask-scaling distillation applies an L2 feature loss to foreground regions in both gated camera and fused BEV maps. It expands the masks for farther or faster objects, where view transformation and temporal alignment can displace features. This gives uncertain object boundaries some tolerance instead of insisting that teacher and student activate at exactly the same cells. A separate relation loss compares multi-scale cosine-affinity matrices, transferring which scene locations have related features rather than only their absolute feature values.

Response distillation supplies class and box targets through quality focal and Smooth L1 losses. Dynamic classes receive weight 2 and static classes weight 1. The appendix compares the opposite emphasis: weighting static classes more heavily reaches 45.4 mAP, while dynamic emphasis reaches 45.7, versus 45.3 with uniform weights. In this experiment, distillation works slightly better when it strengthens a capability radar already provides than when it emphasizes the student's weaker classes.

### Separate the architecture gain from the distillation gain

| nuScenes validation configuration | mAP | NDS |
| --- | ---: | ---: |
| Original camera–radar BEVFusion baseline | 43.2 | 54.1 |
| Add gated fusion | 44.9 | 55.9 |
| Add response distillation | 45.7 | 56.7 |
| Add radar heatmap distillation | 46.0 | 57.0 |
| Add mask-scaling feature distillation | 46.2 | 57.2 |
| Add relation distillation: full CRKD | 46.7 | 57.3 |

The headline 3.5-point mAP improvement includes the gated architecture change. Distillation adds 1.8 points over that stronger student, and the cumulative ablation does not isolate every loss independently. The paper explicitly warns that a module's best standalone design may not be its best design alongside response distillation. Loss interactions are part of the recipe.

Improvement is also metric-specific. Full CRKD improves orientation and localization over the original baseline, yet mean velocity error is 0.331 m/s versus 0.304 for the gated student. Prioritizing dynamic classes improves detection AP without guaranteeing better velocity estimates. The main comparison uses single-frame camera input, but the supplement includes the current radar sweep plus six previous sweeps; “single frame” is not a claim that every sensor contributes only one instant.

The qualitative examples show two different benefits. In the upper row, CRKD removes false student detections. In the lower row, it detects a car the teacher misses and rejects some of the teacher's false positives. That is possible because the student still receives radar evidence and ground-truth supervision: it is not restricted to reproducing the teacher's output exactly. These selected cases do not overturn the aggregate gap—teacher mAP is 66.1, versus 46.7 for CRKD.

![CRKD qualitative examples comparing student errors and teacher errors with corrected camera–radar predictions](/assets/images/crkd-source-figure-3-complementary-detections.png)
*Fig 3: Red denotes ground truth and blue CRKD. The upper row compares the yellow student predictions; the lower row compares the green teacher predictions. Radar points are magenta, and matched borders connect BEV regions to close-ups. | source: [CRKD, Figure 3](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_CRKD_Enhanced_Camera-Radar_Object_Detection_with_Cross-modality_Knowledge_Distillation_CVPR_2024_paper.pdf)*

The supplementary slices further bound the robustness claim. Relative to the gated student, rainy-scene mAP rises from 47.27 to 49.59, but night mAP changes only from 23.94 to 24.14. The largest range-group gain is within 20 meters, 58.54 to 61.59 mAP. Distillation helps across these slices, while their unequal gains prevent treating radar availability as a general solution to darkness, range, or sensor noise.

## High-Level Takeaways

- A shared BEV coordinate system does not make LiDAR and radar features interchangeable. Objectness is a more effective radar supervision target in the source-choice ablation.
- Gated camera features can contain information from the companion sensor, so transferring them teaches more than camera appearance alone.
- The 3.5-point headline mAP gain combines a 1.7-point architecture gain with 1.8 points from distillation over the gated student.
- Dynamic-class emphasis improves detection AP, but velocity error worsens relative to the gated student. The metric attached to a claimed radar benefit matters.
- Teacher errors are not an absolute ceiling on every student prediction, yet the large aggregate teacher gap and minimal night gain show the limits of transferring privileged sensing.
