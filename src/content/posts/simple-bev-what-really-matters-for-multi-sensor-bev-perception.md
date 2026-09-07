---
title: 'Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?'
date: '2022-06-16T09:00:00.000Z'
section: paper-shorts
postSlug: simple-bev-what-really-matters-for-multi-sensor-bev-perception
legacyPath: /paper shorts/2022/06/16/simple-bev-what-really-matters-for-multi-sensor-bev-perception.html
tags:
  - Bird's-Eye View
  - Radar-Camera Fusion
  - Autonomous Driving
field: 'BEV Perception & Mapping'
topics:
  - autonomy
  - learning
summary: '2022 – Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?'
---

**ArXiv:** [2206.07959](https://arxiv.org/abs/2206.07959)

**Project and code:** [Simple-BEV](https://simple-bev.github.io/)

## Summary

> Simple-BEV holds the training recipe still long enough to ask what the lifting operator contributes. In a matched camera-only vehicle-segmentation study, bilinear sampling reaches 47.4 IoU, close to learned deformable alternatives, while effective batch size and input resolution move the score by far more. Adding three aligned radar sweeps raises IoU to 55.7; LiDAR reaches 60.8. The paper's practical decision is therefore broader than “which BEV transformer wins”: first price optimization scale and metric sensing, then pay for a more complex lift.

## Core Insights

### Pull a feature for every voxel instead of pushing pixels along rays

Simple-BEV begins with a metric volume spanning 100×10×100 m and discretized into 200×8×200 voxels. A ResNet-101 produces features for six cameras. For every voxel, the model projects its 3D coordinate into each camera feature map and bilinearly samples the valid features. It averages those views, folds the height axis into channels, and sends the resulting BEV map through a ResNet-18 segmentation network. Radar or LiDAR can be rasterized into the same grid and concatenated before the height collapse.

![Simple-BEV source Figure 1: ray splatting versus voxel-to-image bilinear sampling](/assets/images/simple-bev-what-really-matters-for-multi-sensor-bev-perception-source-figure-1.webp)
*Fig 1: Splatting starts from image pixels and pushes features along rays; Simple-BEV starts from each 3D voxel and pulls a bilinear image feature at its projected coordinate. | source: [Simple-BEV, Figure 1](https://arxiv.org/abs/2206.07959)*

The distinction changes how the two operators cover distance. Splatting can place multiple samples into nearby voxels but may leave far voxels with few or no samples at fixed depth intervals. Pull-based sampling gives every voxel a feature, although far voxels sample densely packed image regions. The paper's distance breakdown shows splatting ahead at close range and bilinear sampling ahead at medium and long range. That is a geometric explanation for why a parameter-free operator can remain competitive.

![Simple-BEV source Figure 2: IoU by distance for splatting and bilinear sampling](/assets/images/simple-bev-what-really-matters-for-multi-sensor-bev-perception-source-figure-2.webp)
*Fig 2: The distance curves show the geometric trade-off: splatting helps nearby voxels, while pull-based sampling is stronger at medium and long range. | source: [Simple-BEV, Figure 2](https://arxiv.org/abs/2206.07959)*

### The matched lifting table is smaller than the training effects

The camera-only experiments use nuScenes vehicle segmentation: 28,130 training samples and 6,019 validation samples, six cameras, and IoU as the metric. The ResNet-101 is initialized from COCO detection; the BEV ResNet-18 starts from scratch. Training runs for 25,000 iterations with AdamW and a one-cycle schedule. The 200×8×200 volume and 200×200 output are held fixed while the authors vary lifting, resolution, batch size, and augmentation.

| Camera-only choice | IoU | Interpretation |
| --- | ---: | --- |
| Unweighted splatting | 43.1 | A simple ray fill is a viable lower baseline. |
| Depth-based splatting | 44.4 | Predicted depth adds only 1.3 points here. |
| Deformable attention | 46.5 | Learned offsets and weights help. |
| Bilinear sampling | 47.4 | Parameter-free geometry is competitive. |
| Multi-scale deformable attention | 48.9 | Best lifting score, with a larger systems cost. |

The matched table keeps backbone, resolution, batch size, and augmentations constant. Multi-scale deformable attention buys 1.5 points over bilinear sampling, but uses 59M rather than 42M parameters, requires a custom CUDA kernel, trains about a day longer, and is 0.5 FPS slower at test time. The difference is real; it is smaller than the nearly 14-point gain in the batch sweep. That sweep holds 25,000 iterations fixed, so batch 40 processes about 20 times as many examples as batch 2. The gain therefore mixes effective-batch and optimization effects with more training data seen; it is not an isolated batch-size intervention.

Resolution creates another explicit frontier. At 672×1200, the model reaches 49.3 IoU, but takes 133 ms versus 83 ms for the 47.4 IoU model at 448×800 and needs nearly twice the training time. At the highest tested resolution, performance falls again, plausibly because the feature scale no longer matches the backbone's pretraining. The paper does not claim that resolution should always be maximized.

### Radar is useful when its input contract is preserved

The radar experiment uses the same synchronized nuScenes vehicle-segmentation setup. Each radar return contributes position, velocity, and metadata channels. Three sweeps at t, t-1, and t-2 are aligned to the current frame; the BEV raster is concatenated with RGB features before compression. Camera-only IoU is 47.4, camera plus radar is 55.7, and camera plus LiDAR is 60.8.

The radar ablations explain why earlier negative results do not settle the question. Removing return metadata lowers IoU by 0.7; using nuScenes-filtered returns instead of raw returns lowers it by 2.0; using one sweep instead of three lowers it by 2.4. The useful signal is not a binary occupancy mask. Velocity can separate moving objects from background, and accumulation helps overcome radar's extreme sparsity.

The final state-of-the-art table should be read with care. Simple-BEV reports 47.4 IoU for RGB at 448×800 and 55.7 for RGB+radar, with 42M parameters and 7.3 FPS on a V100, compared with 2.3 FPS for BEVFormer in that setup. The authors explicitly say that published comparisons mix backbones, categories, augmentations, and training schedules. Their controlled claim is about factors within one reproducible model, not a universal ranking of all BEV systems.

## High-Level Takeaways

- Bilinear sampling is a credible baseline because every metric voxel receives a projected feature; the learned lifting gap is 1.5 IoU in the matched table, while batch size and resolution move the result much more.
- Radar's 55.7 IoU depends on a concrete contract: all return metadata, raw rather than filtered points, and three aligned sweeps. “Radar” without those choices is not one experiment.
- Resolution and effective batch size trade accuracy against training and latency. The 49.3 IoU 672×1200 model is slower than the 47.4 IoU 448×800 model, and the biggest batch requires gradient accumulation across eight A100 GPUs.
- The paper studies vehicle segmentation with synchronized nuScenes sensors and leaves temporal modeling, 3D detection, and deployment under calibration or weather faults open.
