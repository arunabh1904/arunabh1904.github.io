---
title: 'TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers'
date: '2022-03-22T00:00:00.000Z'
section: paper-shorts
postSlug: transfusion-robust-lidar-camera-fusion-with-transformers
legacyPath: /paper shorts/2022/03/22/transfusion-robust-lidar-camera-fusion-with-transformers.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2022 – TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers'
---
## 2022 – TransFusion

**arXiv:** [2203.11496](https://arxiv.org/abs/2203.11496)

**Code:** [XuyangBai/TransFusion](https://github.com/XuyangBai/TransFusion)

## Summary

> TransFusion lets LiDAR propose an object and lets a query search a camera feature map around that proposal. This soft association avoids forcing every image feature through a sparse LiDAR point or a single calibrated pixel. The first decoder layer predicts boxes from LiDAR; the second uses spatially modulated cross-attention to retrieve image evidence, with an image-guided heatmap available for LiDAR-sparse objects. On nuScenes test, fusion improves TransFusion-L from 65.5 to 68.9 mAP and from 70.2 to 71.7 NDS, while a one-meter calibration offset costs only 0.49 mAP in the paper's stress test.

## Core Insights

### LiDAR proposes before the camera refines

TransFusion uses convolutional backbones to produce a LiDAR BEV feature map and multiview image features. A class-specific LiDAR heatmap supplies input-dependent object queries: local maxima provide the query positions, the corresponding BEV features provide instance content, and a category embedding tells each query which class generated it. The first transformer decoder layer predicts a coarse box from LiDAR alone. Each query decodes center offsets, height, logarithmic dimensions, sine/cosine yaw, velocity, and class probabilities.

This initialization matters because a learned or random query must spend decoder layers moving toward an object. A heatmap query begins near a likely center, so one decoder layer can already make a useful proposal. The category embedding is also more than a label: BEV objects have relatively stable scale within a class, so the decoder can model within-class variation instead of rediscovering the category from scratch.

![TransFusion source Figure 2: LiDAR queries followed by soft image fusion](/assets/images/transfusion-paper-figure-2.png)
*Fig 1: The first decoder layer predicts initial 3D boxes from LiDAR object queries; the second layer uses those predictions to guide spatially modulated cross-attention over camera features. | source: [TransFusion, Figure 2](https://arxiv.org/abs/2203.11496)*

### SMCA turns calibration into a locality hint

Point-level fusion projects each LiDAR return to one image pixel. That correspondence is brittle when an object has few returns, when useful texture lies between them, or when calibration is slightly wrong. TransFusion keeps the complete image feature maps as a memory bank. The predicted query box is projected into its relevant camera, and spatially modulated cross-attention multiplies the attention map by a two-dimensional Gaussian around the projected box center. The radius comes from the projected 3D box, so larger or farther geometry changes the area the query can inspect.

The query is therefore encouraged to look near the object but can choose several pixels and use contextual features. It does not have to trust one point-to-pixel match. The calibration still identifies the camera and approximate region, but the attention distribution can move within that region when the projection is imperfect.

![TransFusion source Figure 1: hard point-to-pixel fusion under bad illumination and sparse returns](/assets/images/transfusion-robust-lidar-camera-fusion-with-transformers-source-figure-1.webp)
*Fig 2: The source failure cases show bad illumination and the sparse projected points that waste dense image evidence; a small calibration offset can move those points outside the object. | source: [TransFusion, Figure 1](https://arxiv.org/abs/2203.11496)*

The optional image-guided query initialization addresses a different failure. A LiDAR heatmap can miss a small or distant object entirely. TransFusion collapses each image feature map along height, uses cross-attention with the LiDAR BEV features to form an image-guided BEV hint, averages its heatmap with the LiDAR-only heatmap, and selects queries from the result. The image branch suggests where a candidate may exist; the LiDAR branch still supplies the geometric anchor.

### The attention map shows what soft association buys

The paper's qualitative attention maps are more informative than a claim that the model “uses images.” Object queries often attend to foreground pixels around the projected object rather than only the locations of LiDAR returns. That gives a small object with two or three returns access to the surrounding texture and color that the camera already observed.

![TransFusion source Figure 3: object-query projections and cross-attention maps](/assets/images/transfusion-robust-lidar-camera-fusion-with-transformers-source-figure-3.webp)
*Fig 3: The first row projects query predictions onto nuScenes and Waymo images; the second shows the cross-attention maps. The queries select relevant image regions beyond the sparse LiDAR points. | source: [TransFusion, Figure 3](https://arxiv.org/abs/2203.11496)*

The same design keeps a LiDAR-only path. TransFusion-L is the first decoder stage without camera fusion, so missing or poor images do not make the detector lose its geometric baseline. On nuScenes test, TransFusion-L reaches 65.5 mAP/70.2 NDS, and the fused model reaches 68.9/71.7 without test-time augmentation or a model ensemble. The fused model also reports 71.8 AMOTA on the nuScenes tracking test, compared with 68.6 for TransFusion-L.

### Robustness is measured with controlled sensor failures

The image-quality experiments use the nuScenes validation split with shortened 12-epoch training and compare TransFusion against two hard-fusion baselines built on the same LiDAR model: point-wise concatenation (CC) and PointAugmenting (PA). At night, TransFusion reaches 55.2 mAP, compared with 51.0 for PA and 49.4 for CC; in daytime it reaches 65.7, compared with 64.3 and 63.4. When images are dropped, TransFusion falls from 65.6 mAP with all views to 65.1 with one missing image, 63.9 with three, and 61.7 with six. CC falls to 39.5 and PA to 47.0 when six images are missing.

The calibration test randomly translates the camera-to-LiDAR transform. At a one-meter offset, TransFusion loses 0.49 mAP; PA loses 2.33 and CC 2.85. The paper's explanation is specific: the calibration projects the query into an approximate image region, but the cross-attention can still choose useful pixels around it. This is a robustness mechanism supported by the perturbation, rather than a claim that calibration no longer matters.

## High-Level Takeaways

- Input-dependent, category-aware LiDAR queries give the transformer a geometric proposal before image fusion begins.
- SMCA uses the projected box as a soft locality prior, allowing a query to search image context instead of trusting one LiDAR point or pixel.
- On nuScenes test, fusion improves TransFusion-L from 65.5/70.2 to 68.9/71.7 mAP/NDS and the tracking result from 68.6 to 71.8 AMOTA.
- Under controlled failures, one dropped image costs 0.5 mAP and a one-meter calibration offset costs 0.49 mAP; the LiDAR-only decoder remains the fallback when camera evidence degrades.
