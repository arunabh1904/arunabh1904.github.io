---
title: 'UniBEV: Robust Multimodal Detection with Uniform BEV Encoders'
date: '2023-09-25T04:00:00.000Z'
section: paper-shorts
postSlug: unibev-robust-multimodal-detection-with-uniform-bev-encoders
legacyPath: /paper shorts/2023/09/25/unibev-robust-multimodal-detection-with-uniform-bev-encoders.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2023 – UniBEV: one detector for camera, LiDAR, and fused operating modes'
---

**ArXiv:** [2309.14516](https://arxiv.org/abs/2309.14516)

**Code:** [UniBEV](https://github.com/tudelft-iv/UniBEV)

## Summary

> UniBEV treats a missing sensor as a normal operating mode. Camera and LiDAR features are each converted into BEV by the same deformable-attention interface, then Channel Normalized Weights (CNW) fuses the channels that are actually available. Modality dropout trains one set of weights on camera-only, LiDAR-only, and fused inputs. On nuScenes validation, that model reaches 64.2 mAP with both sensors, 58.2 with LiDAR only, and 35.0 with cameras only, averaging 52.5 across modes. Its main lesson is that robustness comes from a compatible representation and training distribution together; CNW alone cannot teach a detector what a failed sensor looks like.

## Core Insights

### Align the BEV construction before tuning the fusion rule

A fused detector usually assumes that both feature branches exist and that their channels have the same meaning. UniBEV starts earlier. It gives camera and LiDAR branches a shared grid of learnable BEV queries with 3D reference points, then uses three layers of deformable self- and cross-attention to construct each modality's BEV feature map. Camera references are projected into image coordinates; LiDAR references are projected into the native spatial feature map. The query grid is the common scaffold, while each branch still samples its own sensor representation.

![UniBEV source Figure 2: uniform BEV encoders and channel-normalized fusion](/assets/images/unibev-robust-multimodal-detection-with-uniform-bev-encoders-paper-figure.webp)
*Fig 1: Shared BEV queries guide separate deformable encoders for camera and LiDAR features; CNW then fuses the aligned outputs before the detector head. | source: [UniBEV, Figure 2](https://arxiv.org/abs/2309.14516)*

This differs from the usual BEVFusion path, where LSS explicitly predicts camera depth while the LiDAR branch already lives in metric space. It also differs from MetaBEV's deformable fusion block: a stronger fusion module cannot fully repair a mismatch created earlier in the two BEV encoders. UniBEV's claim is therefore about alignment and operating-mode continuity, not about a more elaborate decoder.

### Normalize the channels that remain

Concatenation preserves both feature tensors when both sensors exist, but it creates a fixed-width input when one tensor disappears by filling its channels with zeros. A detector trained on complete inputs can interpret those zeros as an unfamiliar distribution. UniBEV compares concatenation with channel-wise averaging and with CNW.

CNW assigns each modality a learned vector of channel weights. Before fusion, the weights are normalized across the modalities that are present, so a camera-only input receives all of its own channel weight and a LiDAR-only input does the same. With two sensors, a channel can favor camera, favor LiDAR, or remain near an average. The output width stays fixed without a placeholder tensor. The paper also shows that the average channel activation distributions of the two branches are similar, so CNW is not merely correcting a scale mismatch.

![UniBEV source Figure 1: BEVFusion, MetaBEV, and UniBEV fusion interfaces](/assets/images/unibev-robust-multimodal-detection-with-uniform-bev-encoders-source-figure-1.webp)
*Fig 2: The source comparison distinguishes concatenation, deformable fusion, and UniBEV's uniform encoders with channel-normalized weighting. | source: [UniBEV, Figure 1](https://arxiv.org/abs/2309.14516)*

The fusion ablation isolates the modest but consistent advantage of CNW:

| Fusion rule | L+C mAP | LiDAR-only mAP | Camera-only mAP | Average mAP |
| --- | ---: | ---: | ---: | ---: |
| Concatenation | 63.8 | 57.6 | 34.4 | 51.9 |
| Average | 64.1 | 57.6 | 35.1 | 52.3 |
| CNW | 64.2 | 58.2 | 35.0 | 52.5 |

The table makes the decision boundary clear. Normalized averaging prevents a missing stream from changing the feature width; learned weights recover some modality preference. The gain from average to CNW is only 0.2 average mAP, so the larger architectural decision is training and alignment, not the last fusion operator.

### Modality dropout is the actual fallback mechanism

UniBEV uses ResNet-101 with FPN for images and VoxelNet for LiDAR. The authors initialize those branches from FCOS3D and CenterPoint, train for 36 epochs on four A40 GPUs, and retrain the comparison models with the same no-augmentation pipeline because MetaBEV code was unavailable. The default dropout probability is 0.5; of the remaining dropped-modality cases, half keep LiDAR and half keep cameras. Training therefore sees both sensors in 50% of iterations and each single sensor in 25%.

With one set of weights, UniBEV reaches 68.5 NDS/64.2 mAP on L+C, 65.3/58.2 on LiDAR only, and 42.4/35.0 on cameras only. The average is 58.7 NDS/52.5 mAP, above BEVFusion's 43.5 mAP average and MetaBEV's 48.7 under the paper's retrained protocol. Inference speed is 1.6 FPS on a V100 with both modalities, compared with 0.7 for BEVFusion and 1.4 for MetaBEV; camera-only and LiDAR-only inputs are 2.5 and 3.9 FPS.

The probability ablation exposes an asymmetry. When camera-only training is absent, camera-only mAP is just 3.0; increasing camera-only exposure raises it by 33 points. LiDAR-only performance is more forgiving and reaches 45.5 even with no LiDAR-only iterations. The stronger modality can be learned through fused examples; the weaker camera branch needs to see its fallback mode directly. Unified BEV queries provide only a small additional gain over separate queries, 52.5 versus 52.2 average mAP, while also reducing parameters.

## High-Level Takeaways

- UniBEV's robustness is a three-part contract: shared BEV reference points, fixed-width normalized fusion, and explicit training on each input mode.
- CNW improves the average mAP from 51.9 for concatenation to 52.5, but the larger failure is zero-filling a missing branch without changing the training distribution.
- Camera-only fallback is the hard case: its mAP collapses to 3.0 when camera-only iterations are removed, while LiDAR-only training can still reach 45.5 without LiDAR-only examples.
- The benchmark covers complete modality absence and a controlled dropout schedule. Weather, calibration drift, timing faults, and a sensor that is present but unreliable remain separate reliability problems.
