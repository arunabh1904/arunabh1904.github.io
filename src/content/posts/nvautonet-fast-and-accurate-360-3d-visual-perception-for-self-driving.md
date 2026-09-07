---
title: 'NVAutoNet: Fast and Accurate 360° 3D Visual Perception for Self Driving'
date: '2023-03-23T00:00:00.000Z'
section: paper-shorts
postSlug: nvautonet-fast-and-accurate-360-3d-visual-perception-for-self-driving
legacyPath: /paper shorts/2023/03/23/nvautonet-fast-and-accurate-360-3d-visual-perception-for-self-driving.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2023 – NVAutoNet: production-oriented camera-to-BEV perception'
---

**arXiv:** [2303.12976](https://arxiv.org/abs/2303.12976)

**Paper:** [WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Pham_NVAutoNet_Fast_and_Accurate_360deg_3D_Visual_Perception_for_Self_WACV_2024_paper.html)

## Summary

> NVAutoNet is a systems-oriented camera-to-BEV network for three outputs that a vehicle needs together: 3D obstacles, freespace, and parking spaces. Lightweight CNNs and TensorRT provide the execution path; a column-wise MLP lifts image features directly into an irregular polar BEV, and precomputed BEV indices avoid repeated projection work. On eight cameras it measures 18.72 ms / 53 FPS on NVIDIA DRIVE Orin, but the accuracy evidence comes from a large proprietary dataset with noisy LiDAR-derived labels.

## Core Insights

### The view transform is designed to compile

NVAutoNet avoids a dense lifted 3D volume and avoids attention over a large set of BEV queries. Each image column is projected using camera intrinsics and extrinsics, and the resulting polar BEV points fit a polynomial curve that maps radial distance to angular position. A small shared MLP transforms the column features over logarithmic radial bins; BEV indices are then looked up and accumulated into a global polar grid.

This choice is tied to the operating range. A 0.25 m Cartesian grid out to 200 m is expensive, so the network uses fine angular samples and logarithmic radial spacing. The lookup table can be precomputed when calibration is fixed; the paper reports only 0.3 ms per image for the transformation’s indexing overhead. Calibration is still part of the input contract: the same mechanism can adapt to another mounting configuration only when the camera parameters and fine-tuning data are appropriate.

![NVAutoNet camera-to-BEV multitask pipeline](/assets/images/nvautonet-fast-and-accurate-360-3d-visual-perception-for-self-driving-paper-figure.webp)
*Fig 1: NVAutoNet extracts multi-camera CNN features, lifts them into a shared BEV, and decodes obstacles, freespace, and parking outputs; this is the paper’s Figure 1. | source: [NVAutoNet: Fast and Accurate 360° 3D Visual Perception for Self Driving, Figure 1](https://arxiv.org/abs/2303.12976)*

### One BEV supports several task contracts

The shared BEV backbone feeds lightweight task heads. Obstacles use set prediction with one-to-one matching and a greedy candidate restriction rather than NMS. Freespace is represented as a radial distance map: each angular bin predicts the nearest boundary, which is compact and directly useful to a planner. Parking spaces use oriented boxes and a profile label for angled, parallel, and perpendicular maneuvers. An adaptive loss-balancing procedure rescales task losses by their epoch-level totals, then uses manually chosen task priorities in a second training round.

![NVAutoNet polynomial camera-to-BEV projection and lookup indexing](/assets/images/nvautonet-fast-and-accurate-360-3d-visual-perception-for-self-driving-source-figure-2.webp)
*Fig 2: The source’s Figure 2 explains how calibrated camera pixels become polynomial polar curves and then precomputed BEV indices for efficient feature lifting. | source: [NVAutoNet: Fast and Accurate 360° 3D Visual Perception for Self Driving, Figure 2](https://arxiv.org/abs/2303.12976)*

The multitask design is practical, but its shared representation does not make every task equally easy. In the obstacle evaluation, the overall mAP is 0.465 and the safety-region mAP is 0.595. Vehicle AP is 0.638, while person AP is 0.351; orientation error averages 6.542° for vehicles but 53.4° for persons. For freespace, near and front regions perform better because they have denser camera coverage. Parking mean IoU for true positives is about 86%, with parallel spaces hardest because their widths are inconsistently labeled.

### Embedded speed comes with a data boundary

The in-house corpus contains 2.2M training scenes, 400K validation scenes, and 177K test scenes from real, simulated, and augmented-reality data. It uses eight cameras, five obstacle classes, three freespace classes, and three parking classes. TensorRT on DRIVE Orin measures 18.72 ms end to end, or 53 FPS, including camera encoders, uplifting/fusion, and BEV heads.

The platform transfer experiment is a useful practical check. A model pretrained on the car platform reaches 0.168 mAP on the truck validation set without truck fine-tuning, while fine-tuning it reaches 0.286–0.300 as the truck data grows from 50K to 150K scenes. A truck-only model reaches 0.146–0.210 over the same sizes. The evidence supports inexpensive adaptation, but labels are generated from LiDAR at a different mounting height than the cameras, and the proprietary corpus prevents a matched public benchmark that isolates the architecture from data scale.

## High-Level Takeaways

- NVAutoNet’s key contribution is a deployment contract: simple column lifting, polar BEV indexing, CNN backbones, and set-style heads fit an embedded inference budget.
- The irregular grid spends resolution where radial accuracy matters while avoiding the cost of a uniformly dense 200 m Cartesian map.
- Multitask sharing is useful only when the output representation matches the downstream consumer: radial freespace and oriented parking boxes are different objects than 3D obstacle cuboids.
- The Orin measurement is strong systems evidence, while the proprietary labels and task definitions limit direct comparison with public BEV benchmarks.
- Sensor-mount adaptation should be evaluated with explicit calibration perturbations and held-out vehicle platforms, alongside accuracy at long range and rare classes.
