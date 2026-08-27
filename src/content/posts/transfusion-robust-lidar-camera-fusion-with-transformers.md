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

### Method and reported result

TransFusion replaces hard point-to-pixel fusion with soft, object-centric association. A LiDAR BEV heatmap initializes category-aware object queries and a first decoder predicts coarse 3D boxes. A second decoder lets those queries attend to useful multi-camera image regions, so calibration gives a spatial prior without dictating a single brittle correspondence.

## Summary

> The architectural lesson is precise: a sensor can anchor geometry without forcing every other sensor through its samples. Image evidence joins only after a plausible object hypothesis exists.

## Core Insights

Hard association projects each LiDAR point onto one image pixel. It wastes most image features where LiDAR is sparse and becomes brittle when calibration or illumination degrades. TransFusion instead uses the initial 3D box as a soft spatial prior. Spatially modulated cross-attention searches the relevant image area, while an image-guided initialization path can introduce objects that the LiDAR heatmap misses.

On nuScenes, the paper reports that enabling camera fusion improves its LiDAR-only model by 3.4 mAP and 1.5 NDS. Under a simulated 1-meter misalignment, TransFusion loses 0.49 mAP, compared with 2.33 and 2.85 for the two hard-association baselines reported in the same experiment. When images are missing or poor, the first decoder still retains a LiDAR-only path.

![Figure 2 from TransFusion, showing LiDAR-initialized object queries followed by soft image fusion](/assets/images/transfusion-paper-figure-2.png)
*Fig 1: The two decoder stages separate geometric proposal formation from image refinement. | source: [TransFusion](https://arxiv.org/abs/2203.11496)*

![Figure 1 from TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers](/assets/images/transfusion-robust-lidar-camera-fusion-with-transformers-source-figure-1.webp)
*Fig 2: Left: An example of bad illumination conditions. Right: Due to the sparsity of point clouds, the hard-association based fusion methods waste many image features and are sensitive to sensor calibration, since the projected points may fall outside objects due to a small calibration error. | source: [TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers](https://arxiv.org/abs/2203.11496)*

![Figure 3 from TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers](/assets/images/transfusion-robust-lidar-camera-fusion-with-transformers-source-figure-3.webp)
*Fig 3: The first row shows the input images and the predictions of object queries projected on the images, and the second row shows the cross-attention maps. Our fusion strategy is able to dynamically choose relevant image pixels and is not limited by the number of LiDAR points. | source: [TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers](https://arxiv.org/abs/2203.11496)*


| Fusion decision | TransFusion choice | Practical effect |
| --- | --- | --- |
| Geometric anchor | Input-dependent LiDAR BEV queries | Starts near likely object centers and preserves a fallback path. |
| Camera association | Query-to-image attention | Uses a region of evidence instead of one calibrated pixel. |
| Duplicate removal | Set prediction without NMS | Makes the prediction interface cleaner, though matching remains part of training. |
| Degraded images | LiDAR proposal precedes camera fusion | Reduces dependence on every camera being usable. |

## High-Level Takeaways

- TransFusion informs whether camera fusion should happen densely before detection or selectively around object hypotheses. Its atomic unit is an object query initialized from a LiDAR BEV heatmap. Sensor backbones remain separate; sharing begins in the decoder after LiDAR has established a candidate set.
- The critical missing control matches BEV fusion and query fusion for image tokens inspected, latency, and LiDAR backbone. At 10× objects, queries and image cross-attention scale with scene density, while the camera encoder still dominates runtime. The design would be rejected for tasks such as free-space segmentation or lane topology where evidence cannot be reduced to object proposals, or if a dense BEV model matches calibration robustness while serving multiple heads more cheaply.
- TransFusion made query-level soft association a strong alternative to both point painting and dense BEV fusion.
- Calibration should narrow where a model looks; it need not become a hard feature correspondence that turns a small pose error into a missed object.
