---
title: "RCBEVDet: Radar-Camera Fusion in Bird's-Eye View for 3D Object Detection"
date: '2024-03-25T00:00:00.000Z'
section: paper-shorts
postSlug: rcbevdet-radar-camera-fusion-in-bev
legacyPath: /paper shorts/2024/03/25/rcbevdet-radar-camera-fusion-in-bev.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: "2024 – RCBEVDet: Radar-Camera Fusion in Bird's-Eye View for 3D Object Detection"
---

**arXiv:** [2403.16440](https://arxiv.org/abs/2403.16440)

**Code:** [VDIGPKU/RCBEVDet](https://github.com/VDIGPKU/RCBEVDet)

## Summary

> RCBEVDet treats radar as a measurement with its own structure, uncertainty, and motion signal. A point branch preserves sparse returns, a transformer branch supplies context, RCS-aware scattering turns signal strength into a spatial prior, and deformable cross-attention aligns the radar and camera BEV features before fusion.
>
> The paper's useful lesson is architectural: radar should not be forced through a LiDAR encoder and then fused as if both sensors had the same failure modes. The reported gains come from several small decisions that address sparsity, scattering, and spatial misalignment together.

## Core Insights

### Preserve radar structure before making a grid

RadarBEVNet has a point-based stream and a transformer-based stream. The point stream retains local return attributes; the transformer stream models relationships among sparse points with dynamic spatial attention. Injection and extraction blocks let the streams exchange information after each has built its own representation. The ablation explains why the exchange matters: adding the transformer alone gives only a marginal gain because independently processed radar features do not merge cleanly, while the injection/extraction module adds 0.6 NDS and 0.8 mAP on top of the RCS-aware backbone.

RCS-aware scattering uses radar cross section as an object-size prior when spreading one return across BEV cells. Its Gaussian-like spatial weight is scaled by the return's RCS and range, so a strong return can support a broader spatial hypothesis than a weak one. This is a useful inductive bias for radar's sparse footprint, but it remains a prior rather than an instance segmentation target.

![RCBEVDet's camera and radar paths before dynamic fusion](/assets/images/rcbevdet-radar-camera-fusion-in-bev-source-figure-2.webp)
*Fig 1: Camera BEV features and RadarBEVNet features meet in CAMF after dual-stream radar encoding and RCS-aware scattering. | source: [RCBEVDet, Figure 2](https://arxiv.org/abs/2403.16440)*

### Align modalities where they actually disagree

The Cross-Attention Multi-layer Fusion (CAMF) module uses deformable cross-attention in both directions: radar queries sample camera BEV features and camera queries sample radar BEV features at learned offsets. Channel and spatial fusion then combines the aligned features. This separates two questions that a simple concatenation conflates: which nearby cells correspond, and how should the evidence be mixed after correspondence is uncertain?

The component ablations make that causal story legible:

| Incremental configuration on nuScenes val | NDS | mAP |
| --- | ---: | ---: |
| BEVDepth camera baseline | 47.5 | 35.1 |
| + temporal input | 51.9 | 40.5 |
| + PointPillar + BEVFusion | 53.6 | 42.3 |
| + RadarBEVNet | 55.7 | 45.3 |
| + CAMF | 56.4 | 45.6 |
| + temporal supervision | 56.8 | 45.3 |

The last row raises NDS while lowering mAP by 0.3, a reminder that the composite detection score and class-averaged precision reward different behavior. On the nuScenes test set, RCBEVDet with a V2-99 camera backbone reaches 63.9 NDS and 55.0 mAP, compared with 60.5 and 51.5 for the listed BEVDepth V2-99 baseline. On View-of-Delft it reports 69.80 mAP in the defined region of interest.

![RCBEVDet's reported accuracy-speed comparison](/assets/images/rcbevdet-radar-camera-fusion-in-bev-source-figure-1.webp)
*Fig 2: The paper's real-time comparison on nuScenes validation places RCBEVDet on an accuracy-speed frontier measured on one RTX 3090. | source: [RCBEVDet, Figure 1](https://arxiv.org/abs/2403.16440)*

### Robustness is part of the fusion interface

The sensor-drop experiment randomly removes camera views or radar inputs and trains with the same dropout strategy used for comparison. In the paper's car-mAP table, RCBEVDet degrades less than CRN across the three radar-failure settings: CRN drops by 4.5, 11.8, and 25.0 points, while RCBEVDet drops by 0.9, 6.4, and 10.4. The authors attribute this to CAMF's learned alignment, which can still extract useful evidence when one stream is incomplete. These are controlled dropout results, not a substitute for weather, multipath, time synchronization, or calibration stress tests.

The speed numbers also depend on the configuration: the validation table reports 18.2 FPS for the Swin-T 256 x 704 setting and 28.3 FPS for the R18 setting on the stated hardware. A fair deployment comparison therefore needs the same camera resolution, backbone, radar preprocessing, and failure policy.

## High-Level Takeaways

- Radar-specific encoding is justified by sparse returns, RCS, radial velocity, and modality-specific noise.
- Deformable cross-attention addresses correspondence uncertainty before fusion; channel and spatial mixing alone cannot solve a misregistered BEV.
- The component table supports a cumulative design, but the small final mAP trade shows why NDS and mAP should be reported together.
- The next decisive test is matched corruption and latency evaluation across radar dropout, camera dropout, multipath, and calibration drift, with simple pillar and temporal baselines included.
