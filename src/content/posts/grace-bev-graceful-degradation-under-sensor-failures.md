---
title: 'Grace-BEV: Graceful Degradation under Sensor Failures'
date: '2026-05-29T04:00:00.000Z'
section: paper-shorts
postSlug: grace-bev-graceful-degradation-under-sensor-failures
legacyPath: /paper shorts/2026/05/29/grace-bev-graceful-degradation-under-sensor-failures.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2026 – Grace-BEV: reliability-aware camera-LiDAR fusion under sensor failure'
---
## 2026 – Grace-BEV

**arXiv:** [2605.30983](https://arxiv.org/abs/2605.30983)

### Method and reported result

Grace-BEV treats sensor failure as a reliability-estimation problem rather than a binary missing-modality switch. TrustGateRouter estimates how much each camera or LiDAR feature should be trusted, while a FailSafe Fusion Block recalibrates the joint BEV representation. A three-phase training schedule introduces modality dropout and degraded inputs without discarding clean-mode capability.

## Summary

The important shift is from availability to health. A stream can be present yet misleading because of weather, glare, corruption, or sparse returns. Reliability must therefore condition fusion before erroneous evidence dominates the shared scene state.

## Core Insights

The paper evaluates both complete removal and corruption on nuScenes-R and nuScenes-C. Its ablations show why concatenation is not a fallback policy: a concatenation baseline reports 66.9 clean mAP but 0.0 when LiDAR is removed. Reliability-aware routing trades a small amount of peak simplicity for usable degraded modes.

| Operating slice | Reported mAP | What it tests |
| --- | ---: | --- |
| Clean camera + LiDAR | 68.3 | Nominal fusion quality. |
| LiDAR dropped | 32.8 | Whether the camera path remains calibrated. |
| Camera dropped | 58.4 | Whether LiDAR can carry the detector. |
| Added latency | 40.76 to 40.81 ms | Router overhead in the reported setup. |

## High-Level Takeaways

- Grace-BEV is useful when a conditional model must cross transitions between healthy, degraded, and missing sensors. The trust score should be audited as a safety-facing signal: calibration, detection delay, false confidence during correlated corruption, and recovery after a sensor returns matter as much as average mAP.
- The paper does not eliminate the specialist-model question. A fair deployment decision still compares conditional fusion against separately optimized fallbacks under the same memory and worst-case latency budget.
- MetaBEV and UniBEV condition on modality availability; Grace-BEV makes estimated reliability part of the fusion interface.
- Graceful degradation requires the model to represent sensor health, not merely notice that a tensor is absent.
