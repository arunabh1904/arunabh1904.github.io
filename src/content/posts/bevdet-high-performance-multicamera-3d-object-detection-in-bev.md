---
title: 'BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye View'
date: '2021-12-22T05:00:00.000Z'
section: paper-shorts
postSlug: bevdet-high-performance-multicamera-3d-object-detection-in-bev
legacyPath: /paper shorts/2021/12/22/bevdet-high-performance-multicamera-3d-object-detection-in-bev.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2021 – BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye View'
---
## 2021 – BEVDet

**arXiv:** [2112.11790](https://arxiv.org/abs/2112.11790)

**Code:** [HuangJunJie2017/BEVDet](https://github.com/HuangJunJie2017/BEVDet)

### Method and reported result

BEVDet assembles an image backbone, Lift-Splat-style view transformer, BEV encoder, and CenterPoint-like head into a scalable camera-only detector. Its novelty is less a new mathematical operator than a careful argument that BEV needs its own augmentation, resolution, pooling, and suppression policies.

## Summary

> The paper is valuable because it treats the view transform as a systems component. Camera resolution, BEV resolution, and pooling implementation determine different parts of the cost.

## Core Insights

Image-space augmentation alone is insufficient because the detector's output lives in ego coordinates. BEVDet applies rotation, scaling, and flipping after view transformation, then uses a BEV encoder to build metric context. In the reported ablation, combining image and BEV augmentation raises peak mAP from 23.0 to 31.6 and largely removes late-training overfit.

![BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye View source figure: The framework of the proposed BEVDet paradigm.](/assets/images/bevdet-high-performance-multicamera-3d-object-detection-in-bev-paper-figure.webp)
*The framework of the proposed BEVDet paradigm. source: [BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye View](https://arxiv.org/abs/2112.11790)*

![Figure 3 from BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye View](/assets/images/bevdet-high-performance-multicamera-3d-object-detection-in-bev-source-figure-3.webp)
*Figure 3 Combining the features with the auxiliary indexes. source: [BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye View](https://arxiv.org/abs/2112.11790)*

![Figure 1 from BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye View](/assets/images/bevdet-high-performance-multicamera-3d-object-detection-in-bev-source-figure-1.webp)
*Figure 1 The framework of the proposed BEVDet paradigm. BEVDet with a modular design consists of four modules: Image-view encoder, including a backbone and a neck, is applied at first for image feature extraction. View transformer transforms the feature from the image view to BEV. BEV encoder further encodes the BEV features. Finally, a task-specific head is built upon the BVE features and predicts the target values of the 3D objects. We take BEVDet-Tiny as an example for illustrating the channels of different modules. source: [BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye View](https://arxiv.org/abs/2112.11790)*


| Configuration | mAP | NDS | FPS | What changed |
| --- | ---: | ---: | ---: | --- |
| 704×256 images, 0.8 m BEV | 31.2 | 39.2 | 15.6 | Efficient baseline. |
| 1056×384 images, 0.8 m BEV | 33.3 | 41.0 | 8.9 | More long-range image detail. |
| 704×256 images, 0.4 m BEV | 31.5 | 41.0 | 10.0 | Finer metric grid. |
| 1408×512 images, 0.4 m BEV | 36.0 | 43.8 | 5.0 | Accuracy bought on both axes. |

The optimized pooling path precomputes fixed-rig indices and replaces accumulation with an indexed matrix reduction, reducing the paper's BEVDet-Tiny latency from 137 to 64 ms. That result makes a broader point: a view transform can dominate wall time even when its conceptual role is one arrow in a diagram.

BEV improves translation, orientation, and velocity reasoning, but the paper reports weaker attribute prediction than image-view methods. Compressing appearance into BEV helps geometry and can weaken fine visual cues.

## High-Level Takeaways

- BEVDet informs when explicit BEV is worth its fixed spatial cost. Its atomic units change from pixels and depth bins to BEV cells. The image encoder is shared across cameras; geometry is shared through the vehicle-centered grid; the detector pays uniformly for the chosen range and resolution.
- The falsification compares dense BEV with query-based retrieval under equal input pixels, depth supervision, and optimized kernels. BEVDet loses if empty-cell processing dominates P99 latency or if attribute and small-object evidence is damaged by early pooling. At larger range, image resolution and BEV area grow on different axes and must be budgeted separately.
- Lift, Splat, Shoot supplies the geometric primitive. BEVDet turns it into a tuned detector; BEVDet4D adds recurrent BEV, BEVDepth adds LiDAR-supervised depth, and SOLOFusion stretches the temporal stereo horizon.
- BEV performance comes from the whole representation contract—augmentation, resolution, pooling, and heads—not from the view-transform equation alone.
