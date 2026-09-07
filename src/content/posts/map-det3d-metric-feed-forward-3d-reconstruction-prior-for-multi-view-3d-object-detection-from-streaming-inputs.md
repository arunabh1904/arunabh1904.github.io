---
title: "Map-Det3D: Metric Feed-Forward 3D Reconstruction Prior for Multi-view 3D Object Detection from Streaming Inputs"
date: '2026-08-12T00:00:00.000Z'
section: paper-shorts
postSlug: map-det3d-metric-feed-forward-3d-reconstruction-prior-for-multi-view-3d-object-detection-from-streaming-inputs
legacyPath: /paper shorts/2026/08/12/map-det3d-metric-feed-forward-3d-reconstruction-prior-for-multi-view-3d-object-detection-from-streaming-inputs.html
tags:
  - Autonomous Driving
  - 3D Detection
  - Reconstruction Priors
field: 'BEV Perception & Mapping'
summary: "2026 – Map-Det3D: Metric Feed-Forward 3D Reconstruction Prior for Multi-view 3D Object Detection from Streaming Inputs"
---

**arXiv:** [2608.12179](https://arxiv.org/abs/2608.12179) · **Project:** [Map-Det3D](https://royyang0714.github.io/Map-Det3D)

## Summary

> Map-Det3D adapts MapAnything's metric reconstruction prior into a detector for streaming monocular video. A short window supplies multiple views; the detector predicts box geometry up to scale, then uses the backbone's scale factor to recover metric centers and dimensions. The evidence is class-agnostic indoor detection on CA-1M and zero-shot ScanNet, not autonomous-driving detection. Its main insight is that reconstruction features need object-aware adaptation: simply adding temporal views to a frozen backbone produces almost no gain.

## Core Insights

### Separate an object's geometry from the window's metric scale

At each time step, Map-Det3D processes the current image and four earlier frames, returning boxes for the current frame only. MapAnything fuses the views through a transformer and predicts a shared scale factor from a learned scale token. Camera intrinsics and poses are optional inputs, but the strongest ablation configuration uses both; no depth sensor is required at inference.

The detection head regresses unscaled horizontal coordinates, log-depth, log-dimensions, rotation, and objectness. Multiplying coordinates and positive exponentiated sizes by the shared scale factor produces metric boxes. This couples the objects to one window-level geometric interpretation instead of asking every detected object to recover metric depth independently.

![Map-Det3D shares reconstruction features and a scale token across temporal views](/assets/images/map-det3d-metric-feed-forward-3d-reconstruction-prior-for-multi-view-3d-object-detection-from-streaming-inputs-source-figure-2.webp)
*Fig 1: The shared multi-view transformer supplies object features, while the separate scale head converts unscaled geometry to metric units. The trainable components show where a reconstruction model is adapted into a detector. | source: [Map-Det3D, Figure 2](https://arxiv.org/abs/2608.12179)*

“Direct 3D” does not mean the architecture contains no 2D boxes. Dense 2D proposals initialize object queries; reference boxes guide deformable attention and Hungarian matching. The distinction is the final geometry parameterization, which avoids lifting a detected pixel center through a separately regressed range. The training loss still includes auxiliary 2D box supervision alongside disentangled 3D corner losses.

### Temporal evidence becomes useful after the backbone learns object structure

The 50,000-step CA-1M ablation begins at 11.7 AP15 with a frozen single-view setup. Adding multiple views gives 11.8, and unfreezing only the scale head gives 11.9. The prior can reconstruct geometry, yet its original features do not automatically expose the information a box decoder needs.

With both the scale head and multi-view transformer adapted, the single-view result is 14.5; adding temporal views under that same adaptation raises it to 17.2. This is the cleaner comparison for temporal evidence. Camera conditioning supplies another contribution: under the frozen multi-view setting, intrinsics raise AP15 from 11.8 to 12.6, and adding poses reaches 17.3. Combining adaptation, temporal views, intrinsics, and poses reaches 21.2.

The direct-3D head comparison is narrower: with the multi-view transformer and scale head frozen, it improves CA-1M AP15 from 15.6 to 17.3 and ScanNet AP15 from 10.8 to 12.6. These ablations support complementary choices rather than attributing the whole gain to one scale token.

### Better transfer does not imply category understanding or depth-sensor parity

The final model trains for 100,000 steps on CA-1M, whose exhaustive annotations cover indoor objects beyond a short class list. All results are evaluated class-agnostically. A box indicates an object hypothesis; the model does not assign closed-set or open-vocabulary semantic labels.

| CA-1M detector | AP25 | AP50 |
| --- | ---: | ---: |
| CuTR, monocular | 13.5 | 2.4 |
| ImVoxelNet, offline multi-view | 10.1 | 2.3 |
| Map-Det3D, online multi-view | 16.9 | 3.5 |
| FCAF, point cloud | 29.3 | 11.2 |

Map-Det3D improves on the listed image-only baselines while remaining below the depth-based FCAF result, particularly at stricter overlap. AP25 and AP50 refer to 0.25 and 0.50 cuboid IoU thresholds. This evaluation includes boxes regardless of visibility or truncation, which also affects comparison with other detection protocols.

On the selected ScanNet200 zero-shot benchmark, the final model reaches 15.2 AP15 and 9.7 AP25, versus CuTR's 4.3 and 2.1. That controls the detector-training dataset, but the models still differ in reconstruction pretraining and architecture; it does not isolate architecture alone.

### A bounded history limits cost without making streaming free

The temporal/runtime table reports 14.3 FPS and 5.8 GB for one frame, 8.3 FPS and 6.4 GB for five, and 6.3 FPS and 6.7 GB for seven. Five frames give 21.2 CA-1M AP15 in that ablation; seven give 21.1. More context adds cost without improving the measured result. The separate offline five-view row predicts all five frames together and should not be described as the causal streaming setting.

For scene-level evaluation, the authors add simple IoU-based tracking in world coordinates and retain a larger associated box when an object becomes more visible. This reaches 22.7 AP25 on ScanNetV2, versus 11.8 for RGB-only BoxFusion; depth-using BoxFusion reaches 24.6. The heuristic can improve incomplete boxes but does not establish robust tracking of arbitrary dynamic objects.

Indoor-only training, class-agnostic output, optional camera metadata, and a relatively expensive backbone define the supported application. Outdoor driving and semantic detection remain extensions to test.

## High-Level Takeaways

- A shared reconstruction scale can coordinate metric box geometry across temporal views, while 2D proposals remain useful for attention and matching.
- Adapt the reconstruction representation for objects before expecting additional views to help detection.
- Keep camera-conditioning, training length, IoU threshold, and per-frame versus per-scene evaluation attached to each result.
- The demonstrated transfer is indoor and class-agnostic; reconstruction priors are promising geometry, not evidence of an already general driving detector.
