---
title: 'ViDAR: Visual Point Cloud Forecasting for Autonomous Driving'
date: '2023-12-29T05:00:00.000Z'
section: paper-shorts
postSlug: vidar-visual-point-cloud-forecasting-for-autonomous-driving
legacyPath: /paper shorts/2023/12/29/vidar-visual-point-cloud-forecasting-for-autonomous-driving.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2023 – ViDAR: pretrain visual driving encoders by forecasting future point clouds'
---
## 2023 – ViDAR

**arXiv:** [2312.17655](https://arxiv.org/abs/2312.17655)

**Code:** [OpenDriveLab/ViDAR](https://github.com/OpenDriveLab/ViDAR)

### Method and reported result

ViDAR pretrains a multi-camera temporal encoder by forecasting future LiDAR point clouds from historical images. A future decoder evolves BEV queries, while latent rendering predicts ray-wise density so the latent can be supervised by future returns. The pretext task couples semantics, 3D geometry, ego motion, and scene dynamics.

## Summary

> LiDAR is the automatically collected future target; the transferred encoder remains visual. This separates expensive geometric supervision from the deployed sensor graph.

## Core Insights

The paper reports that ViDAR improves downstream 3D detection by 3.1 NDS, reduces motion-prediction error by about 10%, and transfers to occupancy, mapping, tracking, and planning. In its UniAD summary, ViDAR reaches 52.57 NDS, 42.33 mAP, 42.0 AMOTA, 0.67 minADE, and 0.91 m average planning L2, with gains across the baseline row.

![ViDAR: Visual Point Cloud Forecasting for Autonomous Driving source figure: Comparisons among visual autonomous driving pre-training paradigms and our ViDAR architecture.](/assets/images/vidar-visual-point-cloud-forecasting-for-autonomous-driving-paper-figure.webp)
*Fig 1: Comparisons among visual autonomous driving pre-training paradigms and our ViDAR architecture. | source: [ViDAR: Visual Point Cloud Forecasting for Autonomous Driving](https://arxiv.org/abs/2312.17655)*

![Figure 3 from ViDAR: Visual Point Cloud Forecasting for Autonomous Driving](/assets/images/vidar-visual-point-cloud-forecasting-for-autonomous-driving-source-figure-3.webp)
*Fig 2: Ray-shaped features radiate from the ego sensor and obscure scene geometry, whereas geometric features recover a spatial pattern closer to the ground-truth point cloud. | source: [ViDAR: Visual Point Cloud Forecasting for Autonomous Driving](https://arxiv.org/abs/2312.17655)*

![Figure 5 from ViDAR: Visual Point Cloud Forecasting for Autonomous Driving](/assets/images/vidar-visual-point-cloud-forecasting-for-autonomous-driving-source-figure-5.webp)
*Fig 3: The future decoder iteratively predicts the next BEV feature map from ego-motion and the previous BEV state, enabling futures conditioned on arbitrary ego controls. | source: [ViDAR: Visual Point Cloud Forecasting for Autonomous Driving](https://arxiv.org/abs/2312.17655)*


| Objective element | What it teaches | Ambiguity |
| --- | --- | --- |
| Historical images | Appearance and temporal context | Camera visibility limits. |
| Future point clouds | Metric change and occupancy | One sampled future among many. |
| Ego-motion conditioning | Separates vehicle from world motion | Pose error leaks into targets. |
| Latent rendering | Connects BEV state to rays | Adds specialized pretraining machinery. |

## High-Level Takeaways

- ViDAR is relevant when downstream tasks need one temporal representation and fleet logs include synchronized LiDAR during collection. Compare deterministic forecasting with occupancy distributions or multi-hypothesis targets; future geometry is inherently uncertain around occlusion and agent intent.
- Transfer should be measured at fixed encoder size, pretraining tokens, and fine-tuning labels. A world-model objective is valuable only if gains survive several consumers, not one detector head.
- UniWorld predicts 4D occupancy; DriveWorld maintains a reusable 4D latent; ViDAR makes future point-cloud forecasting the visual pretraining task.
- Forecasting geometric evidence forces a visual encoder to model persistence and motion, but the target is privileged supervision—not proof that one future is predictable.
