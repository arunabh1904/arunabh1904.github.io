---
title: 'DeepLiDAR: Surface-Normal-Guided Depth Completion'
date: '2018-12-02T05:00:00.000Z'
section: paper-shorts
postSlug: deeplidar-surface-normal-guided-depth-completion
legacyPath: /paper shorts/2018/12/02/deeplidar-surface-normal-guided-depth-completion.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2018 – DeepLiDAR: combine direct completion with an intermediate surface-normal path'
---
## 2018 – DeepLiDAR

**arXiv:** [1812.00488](https://arxiv.org/abs/1812.00488)

### Method and reported result

DeepLiDAR predicts depth through two paths. One estimates surface normals from RGB and sparse depth, then recovers depth under geometric constraints; the other predicts depth directly. An attention mechanism combines the paths, while a learned confidence mask suppresses projected LiDAR points likely to be wrong because of occlusion or alignment.

## Summary

> The paper's useful intuition is that dense depth needs both measurement anchors and a representation of local surface structure.

## Core Insights

Surface normals transfer more consistently across range than raw depth and provide a boundary-sensitive intermediate target. The direct branch remains necessary where the geometric recovery assumptions fail. On KITTI, DeepLiDAR reports 758.38 mm RMSE versus 814.73 mm for the cited Sparse-to-Dense system. Removing the normal path adds about 87 mm; replacing learned confidence with a binary validity mask adds 69 mm.

| Component | Function | Evidence from ablation |
| --- | --- | --- |
| Normal branch | Encodes local geometry | Removing it materially raises RMSE. |
| Direct branch | Handles unconstrained regions | Complements geometric recovery. |
| Learned confidence | Filters warped sparse points | Beats a binary observed/missing mask. |
| Attention fusion | Chooses per-pixel pathway | Adds model and calibration complexity. |

## High-Level Takeaways

- DeepLiDAR is relevant when projected sparse depth has structured errors rather than simple missingness. The learned confidence should be audited under moving-object boundaries, timing offsets, reflective surfaces, and a changed LiDAR pattern.
- Like Sparse-to-Dense, it expects sparse depth at runtime. Its architecture should not be cited as camera-only privileged training unless that input is explicitly removed and the model retrained.
- Sparse-to-Dense establishes learned RGB-depth completion; GuideFormer later uses cross-modal transformer guidance for the same runtime contract.
- Geometry-aware completion improves when the model learns which measurements to distrust, not only how to interpolate the ones marked valid.
