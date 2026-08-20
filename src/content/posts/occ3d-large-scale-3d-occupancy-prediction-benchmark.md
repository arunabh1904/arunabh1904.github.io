---
title: 'Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving'
date: '2023-04-27T00:00:00.000Z'
section: paper-shorts
postSlug: occ3d-large-scale-3d-occupancy-prediction-benchmark
legacyPath: /paper shorts/2023/04/27/occ3d-large-scale-3d-occupancy-prediction-benchmark.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2023 – Occ3D: visibility-aware dense 3D occupancy benchmarks'
---
## 2023 – Occ3D

**arXiv:** [2304.14365](https://arxiv.org/abs/2304.14365)

**Project and data:** [Occ3D](https://tsinghua-mars-lab.github.io/Occ3D/)

**Code:** [Tsinghua-MARS-Lab/Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D)

### Method and reported result

Occ3D turns camera-based scene reconstruction into a benchmarkable semantic occupancy task. Its label pipeline aggregates LiDAR across frames, densifies voxels, reasons about visibility and occlusion, and uses image evidence to refine boundaries. The release provides Occ3D-nuScenes and Occ3D-Waymo plus CTF-Occ, a coarse-to-fine baseline that spends computation on uncertain voxels.

## Summary

> Occupancy represents geometry that a fixed object vocabulary and bounding boxes miss, but the ground truth is itself a reconstruction. Calibration, motion compensation, visibility, and taxonomy errors enter before model training begins.

## Core Insights

CTF-Occ improves over BEVFormer by 1.65 mIoU on Occ3D-nuScenes and improves the reported Occ3D-Waymo baseline by 1.97 mIoU. On Waymo, top-k uncertain-token selection with online hard-example mining reaches 18.43 mIoU, compared with 14.06 when neither targeted selection nor hard-example mining is active.

| Label stage | Purpose | Failure risk |
| --- | --- | --- |
| Multi-frame densification | Fill sparse 3D space | Pose and motion error. |
| Occlusion reasoning | Mark observed versus hidden voxels | Incorrect visibility boundaries. |
| Image-guided refinement | Sharpen object geometry | Camera-LiDAR calibration error. |

The authors explicitly flag calibration error, non-rigid or unannotated moving objects, and limited semantic categories. Out-of-vocabulary objects are grouped as general objects rather than resolved individually.

## High-Level Takeaways

- Occupancy adds free space, occlusion, and irregular geometry to a scene representation that boxes cannot express.
- Visibility labels matter because an unobserved voxel is not evidence of free space.
- Benchmark quality depends on the same calibration and temporal alignment that the learned projector is expected to handle.
- Occupancy complements object tracks and vector maps; it does not encode identity, intent, or lane topology by itself.
