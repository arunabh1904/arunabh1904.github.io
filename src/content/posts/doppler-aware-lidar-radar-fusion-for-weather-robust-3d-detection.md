---
title: 'Doppler-Aware LiDAR-RADAR Fusion for Weather-Robust 3D Detection'
date: '2025-10-23T00:00:00.000Z'
section: paper-shorts
postSlug: doppler-aware-lidar-radar-fusion-for-weather-robust-3d-detection
legacyPath: /paper shorts/2025/10/23/doppler-aware-lidar-radar-fusion-for-weather-robust-3d-detection.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2025 – DLRFusion: preserving radar Doppler during LiDAR fusion'
---

**Paper:** [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Chae_Doppler-Aware_LiDAR-RADAR_Fusion_for_Weather-Robust_3D_Detection_ICCV_2025_paper.html)

**Code:** [yujeong-star/DLRFusion](https://github.com/yujeong-star/DLRFusion)

## Summary

> DLRFusion treats radar power and Doppler as different signals with different failure modes. It keeps them in separate sparse branches, uses Doppler to refine radar power, lets the refined power update LiDAR through local voxel attention, and adds a direct Doppler–LiDAR path. On K-RADAR this structured interaction reaches 73.2 BEV AP and 45.7 3D AP at IoU 0.5, but the radar bandwidth limitation means the model is using compensated relative Doppler rather than calibrated absolute velocity.

## Core Insights

### Keep Doppler legible before fusion

The radar preprocessing begins with a range–azimuth–Doppler tensor, polar-to-Cartesian conversion, and CFAR filtering that retains the top 10% of radar points by power. Each retained point carries coordinates, power, and Doppler. Because the paper does not have the radar bandwidth needed to recover absolute velocity, it estimates an ego-motion component from the most frequent fixed Doppler bin and subtracts that mean. Power is log-scaled; compensated Doppler and power become separate sparse tensors.

That caveat matters. The model can exploit motion-related contrast, but its Doppler is not a full tracked velocity measurement. The design is therefore about preserving a useful signal through fusion, not solving velocity estimation as a separate task.

![DLRFusion framework with separate LiDAR, radar power, and Doppler branches](/assets/images/dlrfusion-paper-figure.webp)
*Fig 1: DLRFusion encodes LiDAR, radar power, and compensated Doppler separately, then repeats three modality interactions across sparse and BEV stages; the displayed framework is the paper’s Figure 2. | source: [Doppler-Aware LiDAR-RADAR Fusion for Weather-Robust 3D Detection, Figure 2](https://openaccess.thecvf.com/content/ICCV2025/html/Chae_Doppler-Aware_LiDAR-RADAR_Fusion_for_Weather-Robust_3D_Detection_ICCV_2025_paper.html)*

### The interaction order encodes a hypothesis

Each of three MPII stages first uses Doppler to update radar power. Its kernelized attention combines a motion-emphasis term, which favors similar projected features, with a separation term that keeps meaningful differences from collapsing. The resulting radar-power feature updates LiDAR voxels through KNN attention. A direct Doppler–LiDAR path runs in parallel because motion cues can be lost if they have to pass through power first. The refined branches are then sent through separate BEV encoders and repeated at the next stage.

The ablation supports this causal ordering. At IoU 0.5, power–LiDAR interaction alone gives 71.6 BEV AP / 37.9 3D AP; adding direct Doppler–LiDAR changes this to 71.2 / 37.9, while adding Doppler–power produces the full 73.2 / 45.7. In other words, the largest step comes from using Doppler to decide how radar objectness should be transferred before LiDAR fusion, rather than simply adding another edge to the graph.

### Weather results are strong under a clear protocol

K-RADAR supplies seven weather conditions. The authors evaluate the Sedan class in a driving corridor with x∈(0,72 m), y∈(−6.4,6.4 m), z∈(−2,6 m), 0.4 m voxels, and three MPII stages; training uses Adam with batch size 4. At IoU 0.5, DLRFusion’s total is 73.2 BEV AP / 45.7 3D AP, compared with 71.3 / 40.4 for LOD-PDR. At IoU 0.3, it reaches 82.9 / 74.8 versus 82.1 / 73.2. The 3D improvement is more pronounced than the BEV improvement, which is consistent with a cue that helps decide whether returns belong to an object in 3D rather than merely improving its ground-plane footprint.

The radar-only encoding ablation also reaches 49.9 BEV AP / 19.9 3D AP at IoU 0.5 only when preprocessing and separate power/Doppler encoding are both enabled. The comparison is not a proof that every radar stack needs the same kernels, but it does isolate the cost of collapsing Doppler into one undifferentiated feature channel.

## High-Level Takeaways

- DLRFusion’s main idea is representational: power measures radar evidence, while Doppler supplies motion-related evidence, so they should not be concatenated before the network can use the distinction.
- The strongest component ablation is the Doppler–power path, which makes the later LiDAR update conditional on motion-aware radar evidence.
- K-RADAR’s seven-weather evaluation supports robustness under the paper’s preprocessing and Sedan metric, while the absent bandwidth limits the meaning of “velocity.”
- The larger gain in 3D AP than BEV AP is a useful clue about where Doppler helps, but it should be checked by object speed, weather, and false-positive type.
- A next experiment should add calibrated velocity and temporal tracking, then test whether iterative interaction improves persistence without carrying radar ghosts from one frame to the next.
