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

**arXiv:** [2605.30983](https://arxiv.org/abs/2605.30983)

## Summary

> Grace-BEV treats sensor failure as a reliability-estimation problem. A LiDAR-guided expert preserves nominal geometric precision, a pure-vision expert supplies a fallback, and a TrustGate Router estimates how much the LiDAR path should influence the aligned BEV representation. A FailSafe Fusion Block then gates features spatially. On the reported nuScenes-R and nuScenes-C tests, the model recovers from LiDAR blackouts while adding only 0.05 ms to a BEVFusion-MIT latency measurement, but its trust policy still needs testing under correlated and unseen failures.

## Core Insights

### Availability is not the same as health

Static concatenation assumes that a present tensor is useful. A camera can be glare-filled, a LiDAR scan can be sparse or corrupted, and zero-filling a missing modality changes feature statistics in ways a fusion layer may not understand. Grace-BEV stays in the aligned BEV space and creates two experts with different contracts. Expert A uses LiDAR geometry to guide the camera view transform; Expert B builds a BEV representation from images alone. The first is more precise when geometry is healthy, while the second defines a usable lower bound when LiDAR is not trustworthy.

The TrustGate Router reads global average and maximum statistics of the LiDAR BEV feature, compresses them through a small MLP, and emits a sample-level trust score. That score softly interpolates the two experts. The FailSafe Fusion Block then predicts an element-wise sigmoid gate from the routed features, allowing individual BEV regions and channels to suppress corrupted evidence instead of applying one global switch.

![Grace-BEV dual-expert architecture with TrustGate routing and FailSafe Fusion](/assets/images/grace-bev-graceful-degradation-under-sensor-failures-source-figure-2.webp)
*Fig 1: Grace-BEV routes between a LiDAR-guided expert and a pure-vision expert, then uses element-wise gating before the detection head; this is the paper’s Figure 2. | source: [Grace-BEV: Graceful Degradation under Sensor Failures, Figure 2](https://arxiv.org/abs/2605.30983)*

### Training must teach the router both exits

Modality dropout alone does not guarantee a balanced model. The paper uses three phases: one epoch of pure-vision pretraining to establish Expert B, three epochs freezing the backbones while training the router and fusion modules under modality dropout, and three epochs of end-to-end alignment. The balanced dropout allocation gives the model examples of full input, LiDAR-only, and camera-only states, so it cannot learn that one modality is always the shortcut.

The component ablation makes the roles separable. Under the default dropout protocol, the plain baseline is 56.1 mAP / 62.2 NDS on clean data and 12.1 / 14.4 after LiDAR dropout. Adding only TrustGate raises clean performance to 67.2 / 69.9 and LiDAR-drop performance to 28.5 / 35.2. Adding only the FailSafe block gives 67.5 / 70.3 and 15.5 / 20.2. Combining both reaches 68.3 / 71.2 and 32.8 / 39.9, showing that estimating reliability and filtering features solve different parts of the failure.

![Grace-BEV qualitative comparisons under LiDAR, camera, and limited-FOV failures](/assets/images/grace-bev-graceful-degradation-under-sensor-failures-source-figure-4.webp)
*Fig 2: The source’s Figure 4 compares BEVFusion-MIT and Grace-BEV under LiDAR blackout, camera blackout, and limited field of view; the fallback preserves detections when one stream fails. | source: [Grace-BEV: Graceful Degradation under Sensor Failures, Figure 4](https://arxiv.org/abs/2605.30983)*

### Robustness needs several corruption protocols

The main evaluation uses nuScenes-R for sensor impairment and nuScenes-C for weather. In the catastrophic LiDAR-drop setting, standard LSS-based baselines collapse to 0.0 mAP while Grace-BEV reaches as high as 34.7 mAP in the main comparison. In the default component setting, the full model reaches 32.8 mAP under LiDAR drop and 58.4 mAP under camera drop. These values are different experimental slices, so they should not be merged into one headline number.

The weather table supplies a useful out-of-training check. On BEVFusion-AD under Rainy corruption, the baseline is 68.4 mAP and the Grace-BEV plugin reaches 70.7; on BEVFusion-MIT under Sunlight, it moves from 64.1 to 65.7. Clean gains across the compared baselines range from 0.4 to 1.4 points. The router and fusion modules add only 40.76→40.81 ms in the reported BEVFusion-MIT measurement on one A100, but correlated camera/LiDAR failures and trust-score calibration remain untested boundaries.

## High-Level Takeaways

- Grace-BEV’s key design decision is to represent sensor health before fusion, then let a spatial gate suppress the unhealthy stream.
- The pure-vision expert is a real fallback only because the training schedule gives it a dedicated pretraining phase and explicit dropout cases.
- TrustGate and FailSafe Fusion are complementary: the scalar route chooses the expert mixture, while the element-wise gate handles local corruption.
- Weather gains and blackout gains test different failure distributions; a single clean mAP number cannot stand in for either.
- A deployment study should calibrate trust scores, test simultaneous or correlated failures, and measure recovery when a sensor returns under the same worst-case latency budget.
