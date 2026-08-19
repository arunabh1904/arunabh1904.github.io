---
title: 'GuideFormer: Transformers for Image-Guided Depth Completion'
date: '2022-06-19T04:00:00.000Z'
section: paper-shorts
postSlug: guideformer-transformers-for-image-guided-depth-completion
legacyPath: /paper shorts/2022/06/19/guideformer-transformers-for-image-guided-depth-completion.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2022 – GuideFormer: transfer RGB structure into sparse-depth features with guided attention'
---
## 2022 – GuideFormer

**Paper:** [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Rho_GuideFormer_Transformers_for_Image_Guided_Depth_Completion_CVPR_2022_paper.html)

### Method and reported result

GuideFormer uses separate color and sparse-depth branches, then transfers image context through guided attention rather than concatenating the modalities at the input. A depth-fusion stage combines predictions with learned confidence. The separation lets the depth branch retain measured geometry while the image branch supplies boundaries and long-range context.

## Summary

> The method illustrates a broader fusion rule: interaction should be directional when one modality supplies the metric anchor and another supplies structural guidance.

## Core Insights

Transformer blocks enlarge the receptive field available to sparse measurements, while guided attention limits exchange to features useful for depth. The paper reports 721.48 mm RMSE on KITTI depth completion. It also identifies computation speed as a limitation, so the accuracy result should be considered with attention cost and deployment support.

| Module | Information retained | Cost |
| --- | --- | --- |
| RGB branch | Edges, texture, semantics | Appearance-domain sensitivity. |
| Sparse-depth branch | Direct metric samples | Coverage follows runtime sensor. |
| Guided attention | Cross-modal long-range context | Quadratic or windowed attention work. |
| Confidence fusion | Per-pixel arbitration | Confidence needs calibration. |

## High-Level Takeaways

- GuideFormer is appropriate when local convolution cannot connect sparse anchors across large image regions. Compare it against dilated or multi-scale convolution at matched latency and memory, and test different scan patterns rather than one fixed density.
- The model still depends on runtime sparse depth. For a camera-only product, its useful lesson is the directional guidance architecture, not the published input contract.
- DeepLiDAR uses normals and confidence to structure completion; GuideFormer uses attention to increase the interaction range between RGB and depth.
- Cross-modal attention can make sparse range evidence influence distant pixels, but it must justify that reach against its runtime and calibration cost.
