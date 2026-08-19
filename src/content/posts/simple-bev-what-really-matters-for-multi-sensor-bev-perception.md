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

## 2022 – Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?

**arXiv:** [2206.07959](https://arxiv.org/abs/2206.07959)

**Project and code:** [Simple-BEV](https://simple-bev.github.io/)

## Summary

> BEV papers often attribute gains to the operator that lifts perspective-view image features into a bird's-eye-view grid. Simple-BEV holds more of the training recipe fixed and finds a different ordering of importance. Across matched camera-only models on nuScenes vehicle segmentation, input resolution and effective batch size move IoU far more than the choice among depth splatting, deformable attention, and parameter-free bilinear sampling.

## Core Insights

The paper also tests a sensor choice that camera-only comparisons omit. Rasterizing three sweeps of raw radar returns into the BEV grid and concatenating them with camera features raises IoU from 47.4 to 55.7; LiDAR reaches 60.8. The lesson is not that radar replaces LiDAR, but that sparse metric measurements can remove enough geometric ambiguity to matter when fused with dense visual features.

Simple-BEV begins with a $100\text{ m}\times10\text{ m}\times100\text{ m}$ 3D volume discretized to $200\times8\times200$. A ResNet-101 produces features for six cameras. For every voxel, the model projects its 3D coordinate into each image and bilinearly samples the corresponding feature, then averages valid observations across cameras. The vertical axis is folded into channels, optional radar or LiDAR features are concatenated, and a BEV ResNet-18 predicts vehicle occupancy with auxiliary centerness and offset heads.

![Lift-Splat pushes image features along rays, while Simple-BEV pulls a projected image feature for every 3D voxel](/assets/images/simple-bev-lifting.png)
_The parameter-free lifter starts from each voxel and samples the projected image location, guaranteeing one feature per visible voxel. Cropped from Figure 1 of the [paper](https://arxiv.org/abs/2206.07959)._

The lifting comparison is deliberately matched on resolution, batch size, backbone, and augmentations. Multi-scale deformable attention reaches 48.9 IoU, bilinear sampling 47.4, deformable attention 46.5, depth-based splatting 44.4, and unweighted splatting 43.1. The best learned operator buys 1.5 points over bilinear sampling while adding parameters, a custom CUDA kernel, slower training, and lower inference speed. Distance-stratified results add nuance: splatting is better nearby, while bilinear sampling is better at medium and long range.

| Factor | Controlled result | Decision implication |
| --- | --- | --- |
| Effective batch size | 2 to 40 improves IoU by nearly 14 points | Optimization can dominate the apparent architecture gain |
| Input resolution | Best result is 49.3 at $672\times1200$; $448\times800$ gives 47.4 with 83 ms vs 133 ms | Resolution buys accuracy but creates a clear latency frontier |
| Crop and resize augmentation | 45.8 to 47.4 IoU | Geometry-consistent image augmentation matters |
| Random reference camera | 46.8 to 47.4 IoU | Rotating the BEV frame reduces orientation bias |
| Camera to camera+radar | 47.4 to 55.7 IoU | Sparse metric sensing is more valuable than another lifting variant in this setup |

Radar succeeds only with the right input contract. Keeping all return metadata rather than binary occupancy adds 0.7 points, disabling the nuScenes outlier filter adds 2.0, and aggregating three time-aligned sweeps instead of one adds 2.6. These results explain why a sparse sensor can appear useless under an impoverished preprocessing pipeline.

## High-Level Takeaways

- Simple-BEV informs where to spend a BEV perception budget: lifting research, input fidelity, optimization scale, or another sensor. Its atomic visual unit is a sampled image feature attached to a metric voxel; radar adds sparse BEV cells with position, velocity, and return metadata. Parameters are shared across cameras, and sensor features fuse only after the vertical dimension is collapsed. For nuScenes vehicle segmentation, the evidence favors securing resolution, effective batch size, and radar preprocessing before replacing a simple geometric lifter.
- The expensive commitment is the sensor-and-compute stack. A larger batch of 40 is obtained through gradient accumulation across eight A100 GPUs and takes about five seconds per optimizer step. Higher image resolution improves IoU but also raises latency and training time. Radar adds calibration, synchronization, temporal alignment, and failure handling that a single benchmark IoU does not price.
- The missing test is a modern matched-budget replication. Re-run simple sampling, depth lifting, and deformable attention with current backbones across detection, occupancy, mapping, and adverse weather; match wall-clock training, latency, and parameter count; then ablate radar quality and calibration noise. The paper's conclusion should be revised if learned lifting produces consistent gains after these controls or if radar's advantage vanishes outside vehicle segmentation.
- Simple-BEV is a controlled-baseline paper: it asks whether architectural novelty still wins after training details and readily available metric sensors are treated as first-class variables.
- The task is binary vehicle segmentation on the nuScenes validation split, with no temporal camera model, 3D detection, map prediction, or closed-loop driving evaluation. The state-of-the-art table is not fully controlled, and the hardware cost of large effective batches is substantial.
- Before inventing a more elaborate camera-to-BEV lift, fix resolution and optimization—and use sparse metric sensing when the vehicle already has it.
