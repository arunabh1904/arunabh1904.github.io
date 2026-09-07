---
title: 'UniLION: Towards a Unified Autonomous Driving Model with Linear Group RNNs'
date: '2025-11-03T00:00:00.000Z'
section: paper-shorts
postSlug: unilion-towards-unified-autonomous-driving-model-with-linear-group-rnns
legacyPath: /paper shorts/2025/11/03/unilion-towards-unified-autonomous-driving-model-with-linear-group-rnns.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2025 – UniLION: one linear-RNN backbone across sensors, time, and driving tasks'
---

**arXiv:** [2511.01768](https://arxiv.org/abs/2511.01768)

## Summary

> UniLION turns sparse LiDAR, camera, and temporal voxels into one 3D backbone built from grouped linear RNNs, then fans a shared BEV representation into perception, prediction, and planning heads. Its strongest evidence is a sensor-ablation rather than a single leaderboard number: one model trained with LiDAR, cameras, and history can run after inputs disappear, although a LiDAR-only specialist remains better when LiDAR-only deployment is known in advance.

## Core Insights

### A grouped linear recurrence replaces explicit fusion at token level

UniLION keeps modality-specific front ends, but moves the fusion decision into a shared sparse 3D backbone. Dynamic voxelization produces LiDAR voxels; a camera encoder estimates depth and lifts the top four depth candidates per image feature into camera voxels. Temporal voxels from four consecutive frames are then concatenated with the current voxels. The backbone partitions sparse features into non-overlapping 3D windows, sorts them along alternating spatial axes, and groups up to 4,096 tokens for a linear group RNN. Four blocks progressively use windows `(13,13,32)`, `(13,13,16)`, `(13,13,8)`, and `(13,13,4)` with group sizes 4,096, 2,048, 1,024, and 512.

That choice changes what “fusion” means. There is no camera–LiDAR cross-attention module or temporal alignment module after token construction; the recurrence mixes the concatenated voxel sequence while voxel merging and expansion change resolution. A 3D spatial descriptor and voxel generation compensate for two weaknesses of flattening: nearby voxels can become far apart in a 1D order, and sparse foreground evidence can vanish during downsampling.

![UniLION compares explicit fusion pipelines with one backbone and parallel task heads](/assets/images/unilion-towards-unified-autonomous-driving-model-with-linear-group-rnns-paper-figure.webp)
*Fig 1: The comparison contrasts explicit multi-modal and temporal fusion with UniLION's shared 3D backbone and decoupled task heads. | source: [UniLION: Towards a Unified Autonomous Driving Model with Linear Group RNNs, Figure 1](https://arxiv.org/abs/2511.01768)*

The spatial descriptor is not cosmetic. In the paper's Figure 4, voxels numbered 01 and 34 are adjacent in the 2D grid but become separated by the scan order along the x axis. The descriptor supplies spatial coordinates so the recurrent operator does not have to infer all locality from sequence position alone. The design therefore buys long-range token interaction through linear-time recurrence while adding explicit spatial information where the sequence representation is lossy.

![Flattening a 3D window can separate adjacent voxels in sequence order](/assets/images/unilion-towards-unified-autonomous-driving-model-with-linear-group-rnns-source-figure-4.webp)
*Fig 2: The example shows why spatial descriptors are needed when a 2D neighborhood is flattened into a 1D recurrent sequence. | source: [UniLION: Towards a Unified Autonomous Driving Model with Linear Group RNNs, Figure 4](https://arxiv.org/abs/2511.01768)*

### One rich training model provides fallbacks, with a distribution cost

The cleanest unification test is Table VIII. The authors train one model with LiDAR, cameras, and temporal input (LCT), then remove streams only at inference without masked-modality training. The full LCT-to-LCT configuration reaches 75.4 NDS, 76.5 AMOTA, 73.3 map mIoU, and 51.3 RayIoU. Removing history while retaining LiDAR and cameras gives 74.9 NDS, 76.2 AMOTA, 72.2 mIoU, and 50.7 RayIoU. Removing cameras and history and running the same model on LiDAR gives 70.6 NDS, 70.2 AMOTA, 68.6 mIoU, and 43.4 RayIoU.

That last number is useful precisely because it is not the best LiDAR-only result. A LiDAR-only model trained and tested as L reaches 72.3 NDS, 72.6 AMOTA, 71.7 mIoU, and 46.8 RayIoU. Rich-sensor pretraining gives a graceful fallback, but it also creates a train–test mismatch when the missing sensor is known. The paper demonstrates architectural compatibility, not that one universal checkpoint dominates every deployment-specific specialist.

On the full nuScenes validation setup, the LCT model reaches 73.2 mAP and 75.4 NDS for detection, 76.5 AMOTA for tracking, 73.3 map mIoU, and 51.3 RayIoU. Planning is 0.65 m average L2 and 0.18% collision without ego status; a separate ego-status row reaches 0.55 m and 0.06%, so those figures should not be compared as if they used the same inputs.

### Multi-task sharing improves balance but does not remove interference

UniLION trains detection, tracking, map segmentation, occupancy, motion, and planning from the shared BEV state, but the task heads still impose competing gradients. Adding dynamic loss balancing changes the R50 ablation from 73.3 NDS, 74.1 AMOTA, 71.2 mIoU, and 50.4 RayIoU to 73.6, 75.0, 71.8, and 50.2 respectively. The gain is broad but not monotonic: occupancy slightly falls while detection, tracking, and mapping improve.

The training schedule also matters. Perception is trained first, temporal variants are initialized from single-frame perception, and motion and planning are trained after the temporal perception model is frozen. “Unified model” therefore describes parameter reuse and a common representation more strongly than it describes one jointly optimized end-to-end objective. The benchmarks are nuScenes and Occ3D-nuScenes, with six cameras and 2 Hz annotations; the paper does not report a matched hardware latency, memory, or throughput comparison that would quantify the practical benefit of the linear operator.

## High-Level Takeaways

- UniLION makes sensor and task unification a backbone decision: heterogeneous sparse voxels share one recurrent 3D state before task-specific heads read it.
- A rich-sensor checkpoint can survive missing history or cameras, but the LiDAR-only fallback remains weaker than a LiDAR-specialized model; graceful degradation needs an explicit acceptance threshold.
- Spatial descriptors and voxel generation are the mechanisms that keep linear sequence mixing from losing local geometry and sparse foreground evidence.
- Multi-task loss balancing shifts the compromise among tasks rather than eliminating interference.
- The paper establishes broad benchmark coverage and input flexibility; matched efficiency, calibration failure, and sensor-corruption tests are still needed to establish a system-level advantage.
