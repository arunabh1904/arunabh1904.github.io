---
title: 'PointPainting: Sequential Fusion for 3D Object Detection'
date: '2019-11-22T05:00:00.000Z'
section: paper-shorts
postSlug: pointpainting-sequential-fusion-for-3d-object-detection
legacyPath: /paper shorts/2019/11/22/pointpainting-sequential-fusion-for-3d-object-detection.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2019 – PointPainting: Sequential Fusion for 3D Object Detection'
---
## 2019 – PointPainting

**arXiv:** [1911.10150](https://arxiv.org/abs/1911.10150)

### Method and reported result

PointPainting runs image semantic segmentation first, projects every LiDAR point into the camera output, appends the class-probability vector to the point, and then feeds the enriched cloud to an otherwise standard LiDAR detector. Fusion happens before voxelization, but after the camera has compressed RGB into semantics.

## Summary

> Its appeal is modularity. The LiDAR model receives explicit semantic evidence without learning a large joint feature space. The same choice also defines its ceiling: image information can survive only where LiDAR produced a point and only in the form the segmentation model emitted.

## Core Insights

The paper paints PointPillars, VoxelNet, and PointRCNN and reports gains across 24 of 27 KITTI validation comparisons. On nuScenes test, its strengthened PointPillars+ baseline rises from 40.1 to 46.4 mAP and from 55.0 to 58.1 NDS after painting. Every nuScenes class improves, with especially large gains for traffic cones, whose sparse LiDAR returns benefit from image semantics.

![PointPainting: Sequential Fusion for 3D Object Detection source figure: PointPainting overview.](/assets/images/pointpainting-sequential-fusion-for-3d-object-detection-paper-figure.webp)
_PointPainting overview. Source: [PointPainting: Sequential Fusion for 3D Object Detection](https://arxiv.org/abs/1911.10150), Figure 2, via arXiv HTML._


| What PointPainting preserves | What it discards |
| --- | --- |
| Per-point geometry and LiDAR intensity | Image features between projected LiDAR samples |
| Camera class probabilities | Texture and geometry not expressible as semantic classes |
| Existing LiDAR detector interface | Joint adaptation of the camera network to 3D errors |
| Sensor provenance at each point | Soft correspondence when calibration is uncertain |

The segmentation-quality ablation shows detection improving with segmentation mIoU, while qualitative failures show a sharper boundary: incorrect semantic painting can create false classes. Pipelining reduces latency because segmentation and LiDAR processing can overlap, but that systems result assumes synchronization and a known dependency graph.

## High-Level Takeaways

- PointPainting informs whether sensor fusion needs an end-to-end joint model. Its atomic unit is a LiDAR point augmented with a semantic vector. The camera and LiDAR networks remain independently interpretable; the projection is the only shared interface. This makes the design easy to retrofit and test, but gives LiDAR sampling control over camera recall.
- The matched alternative is a learned point- or query-level fusion model using the same image backbone, LiDAR detector, and latency budget. Painting should be rejected if learned retrieval recovers small and distant actors from image regions without points, or if calibration perturbations cause the hard projection to fail abruptly. At scale, the separate segmentation model and duplicated camera computation can dominate the efficiency argument.
- PointPainting is a clean early-fusion reference. TransFusion relaxes hard point-pixel correspondence with object-query attention; BEVFusion preserves dense camera context by meeting LiDAR only after both reach BEV.
- Early fusion can be simple and effective, but the sensor chosen as the carrier decides which evidence the other sensor is allowed to contribute.
