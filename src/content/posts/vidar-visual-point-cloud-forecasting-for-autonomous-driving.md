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

**arXiv:** [2312.17655](https://arxiv.org/abs/2312.17655)

**Code:** [OpenDriveLab/ViDAR](https://github.com/OpenDriveLab/ViDAR)

## Summary
>
> ViDAR turns multi-view video into a geometric forecasting task: a history encoder predicts future LiDAR point clouds through a latent BEV representation, and a future decoder rolls that representation forward under specified ego-motion. Its Latent Rendering operator prevents the pretraining signal from collapsing into identical responses along each camera ray.
>
> The paper is persuasive because it connects one failure mode to one operator. Differentiable ray casting can tell the model that something lies somewhere on a ray, but not which depth is geometrically meaningful. Latent Rendering first makes depth responses conditional along a ray and then weights the feature expectation by that conditional occupancy.
>
## Core Insights

### A ray is a line, not a surface

The history encoder turns multi-view images into BEV features. A simple differentiable ray-casting target gives every BEV cell along the same ray a similar response, producing the “ray-shaped” activations shown in the paper. Latent Rendering projects each BEV feature into an independent probability map, then computes a conditional probability for a cell by multiplying its response by the probability that all earlier waypoints on that ray are empty. It takes a feature expectation along the ray and multiplies that shared ray feature by the cell's conditional probability.

That sequence is the key intuition: the model first asks “where is the first plausible occupied point?” and only then writes a feature at that depth. The multi-group version splits the channels into parallel groups so different groups can retain different geometric evidence instead of averaging every ray into one response.

![ViDAR's visual point-cloud forecasting pretraining framework](/assets/images/vidar-visual-point-cloud-forecasting-for-autonomous-driving-paper-figure.webp)
*Fig 1: ViDAR encodes historical multi-view images, applies Latent Rendering, and autoregressively forecasts future point clouds under ego-motion conditions. | source: [ViDAR, Figure 2](https://arxiv.org/abs/2312.17655)*

![Ray-shaped versus geometric latent features in ViDAR](/assets/images/vidar-visual-point-cloud-forecasting-for-autonomous-driving-source-figure-3.webp)
*Fig 2: Differentiable ray casting produces similar responses along a ray, while Latent Rendering preserves a localized geometric response beside the ground-truth point cloud. | source: [ViDAR, Figure 3](https://arxiv.org/abs/2312.17655)*

### Forecast the representation, then condition the future

The Future Decoder is a six-layer transformer that iteratively predicts the next BEV feature from the previous feature and an ego-motion condition. Deformable self-attention, temporal cross-attention, and future BEV queries let it align the previous and target ego coordinate systems. A projection head turns each predicted feature into an occupancy volume, and a ray-wise cross-entropy loss scores the ground-truth future point along its ray against sampled alternatives.

The default pretraining uses five historical frames, six rollout steps at 0.5-second intervals, a 200 x 200 BEV grid, and a 200 x 200 x 16 occupancy volume. The model is trained on image-LiDAR sequences, but downstream inputs are visual. Because the decoder receives future ego-motion, the same history can be rolled under different turns or straight-line controls; that is a world-model capability, while the paper's downstream evidence remains open-loop.

### The operator is doing the work

The downstream detection ablation isolates the rendering choice:

| Forecasting structure | NDS (%) |
| --- | ---: |
| No forecasting pretraining | 44.11 |
| Differentiable ray casting | 40.20 |
| Latent Rendering, 16 groups | 47.58 |

The naive forecasting target is worse than the baseline because it teaches the encoder the wrong geometry. Latent Rendering recovers that loss and improves NDS by 3.47 points over the baseline. This is a stronger argument than the aggregate transfer table: forecasting is useful only after the representation has a way to resolve depth along a ray.

With the same image-LiDAR pretraining recipe, BEVFormer RN101 detection rises from 37.7 mAP / 47.7 NDS after ImageNet classification pretraining to 42.6 / 51.8 with ViDAR. A 3D-detection-pretrained initialization rises from 41.5 / 51.7 to 45.8 / 54.8. UniAD motion forecasting improves from minADE 0.75 to 0.67 m and from minFDE 1.08 to 0.99 m; future occupancy IoU rises from 62.8 to 65.4 nearby and 40.1 to 42.1 far away. Open-loop planning averages 0.91 m L2 and 0.23% collision rate over the reported three-second horizon, compared with UniAD's 1.12 m and 0.27%; the paper notes that its averaging protocol is not identical to every prior baseline's timestamp reporting.

![ViDAR's ego-motion-conditioned future decoder](/assets/images/vidar-visual-point-cloud-forecasting-for-autonomous-driving-source-figure-5.webp)
*Fig 3: The Future Decoder rolls BEV features forward while conditioning each step on the planned ego-motion and the previous latent. | source: [ViDAR, Figure 5](https://arxiv.org/abs/2312.17655)*

## High-Level Takeaways

- ViDAR's central contribution is a depth-aware latent rendering operator that fixes the ray-shaped failure of naive point-cloud forecasting.
- Future point-cloud supervision gives a visual BEV encoder a reason to model both geometry and temporal change.
- The downstream gains are broad, but the rendering ablation shows that the forecasting target can hurt without the right geometric bottleneck.
- The method still relies on a limited nuScenes image-LiDAR pretraining corpus and open-loop evaluation; control-conditioned point-cloud rollouts need closed-loop validation before they support a planning claim.
