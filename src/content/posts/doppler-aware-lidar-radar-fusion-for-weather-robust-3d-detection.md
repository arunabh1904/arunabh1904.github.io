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
## 2025 – Doppler-Aware LiDAR-RADAR Fusion

**Paper:** [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Chae_Doppler-Aware_LiDAR-RADAR_Fusion_for_Weather-Robust_3D_Detection_ICCV_2025_paper.html)

**Code:** [yujeong-star/DLRFusion](https://github.com/yujeong-star/DLRFusion)

### Method and reported result

DLRFusion does not collapse every radar measurement into one feature tensor. It encodes LiDAR, radar power, and radar Doppler in separate sparse branches, then uses a multi-path iterative interaction block to let Doppler refine radar power and LiDAR features before detection.

## Summary

> Doppler is not just another radar channel. It is a motion cue with different noise and semantics, so the fusion operator should decide where it changes object evidence instead of burying it inside early concatenation.

## Core Insights

On K-RADAR, DLRFusion reports 73.2 BEV AP and 45.7 3D AP at IoU 0.5 across weather conditions. The second-best compared system, LOD-PDR, reports 71.3 and 40.4 respectively, so the 3D AP gain is 5.3 absolute points, or about 13% relative. The paper's fusion ablation is also diagnostic: separate encoding followed by the proposed interaction reaches 45.7 3D AP, versus 36.9 for concatenation and 31.0 for a BEVFusion-style variant.

| Fusion strategy | BEV AP | 3D AP |
| --- | ---: | ---: |
| Concatenation | 71.4 | 36.9 |
| BEVFusion-style | 72.0 | 31.0 |
| DLRFusion | 73.2 | 45.7 |

The limitation is temporal. K-RADAR's restricted Doppler range does not support reliable absolute object velocity, so the paper uses simple motion compensation and evaluates single-frame detection rather than tracking.

## High-Level Takeaways

- Signal-specific branches matter when channels carry structurally different evidence rather than interchangeable features.
- The largest reported gain is in 3D localization, consistent with Doppler helping the model decide which returns belong to moving objects.
- Weather robustness should be sliced by condition; an aggregate score can hide a modality that only helps in a narrow regime.
- The next test is whether calibrated velocity and temporal fusion preserve the gain without amplifying radar ghosts across frames.
