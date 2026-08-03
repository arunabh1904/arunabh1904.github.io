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
## 2024 – RCBEVDet

**arXiv:** [2403.16440](https://arxiv.org/abs/2403.16440)

**Code:** [VDIGPKU/RCBEVDet](https://github.com/VDIGPKU/RCBEVDet)

**Summary:** RCBEVDet treats radar as a distinct sensor rather than a poor substitute for LiDAR. A point branch preserves individual returns; a transformer branch models relationships among sparse returns; Radar Cross Section informs how features are scattered into BEV; and deformable cross-attention aligns the resulting radar map with camera BEV before fusion.

That decomposition matters for production camera-radar stacks. Radar supplies range and radial velocity in conditions where images weaken, but its azimuth noise, sparsity, multipath, and weak elevation make a LiDAR encoder a bad default.

## Paper Insights

RadarBEVNet couples two encoders. The point path retains local measurement attributes, while the transformer path supplies context; injection and extraction blocks let them exchange features. RCS-aware scattering uses return strength as a prior when spreading sparse points into BEV. The Cross-Attention Multi-layer Fusion module then offsets spatial misalignment before channel and spatial fusion.

On nuScenes test, the paper reports that RCBEVDet improves its BEVDepth baseline by 3.4 NDS and 3.5 mAP. It surpasses the compared CRN configuration by 1.5 NDS with a smaller image backbone and reports 21–28 FPS depending on configuration. On View-of-Delft, which uses 4D radar, it reports 69.80 mAP in the region of interest. A sensor-dropout experiment is included, but the main benchmark gains do not by themselves establish adverse-weather reliability.

![Figure 2 from RCBEVDet, showing a dual-stream radar encoder, RCS-aware BEV scattering, and camera-radar cross-attention](/assets/images/rcbevdet-paper-figure-2.png)
_The radar path preserves measurement-specific structure before meeting camera features in BEV. Source: [RCBEVDet](https://arxiv.org/abs/2403.16440), Figure 2._

| Radar property | Modeling response | Why it matters |
| --- | --- | --- |
| Sparse returns | Point and transformer branches | Retains measurements while adding context. |
| Radar Cross Section | RCS-aware BEV scattering | Uses signal strength as a spatial extent prior. |
| Azimuth and calibration error | Deformable cross-attention | Learns a local alignment instead of assuming exact cell correspondence. |
| Radial velocity | Preserved radar attributes | Contributes motion evidence cameras infer only indirectly. |

## Decision Lens

RCBEVDet informs whether a unified camera-radar model should reuse a generic point-cloud encoder or pay for radar-specific structure. Its atomic unit is a radar return enriched by pointwise and contextual processing. Camera and radar backbones stay separate; sharing begins after both enter BEV.

The missing experiment compares radar-specific and LiDAR-derived encoders under adverse weather, matched latency, and calibrated uncertainty, with range- and velocity-bucketed metrics. At 10× temporal accumulation, ghost returns and association cost can dominate unless the model tracks measurement age and ego motion. The radar-specific path would fail if a simpler pillar encoder plus temporal fusion matches long-range velocity and corruption performance within the same compute budget.

**Context:** CRN establishes radar-guided camera lifting and real-time camera-radar BEV; RCBEVDet sharpens the radar encoder and alignment path. Later 4D-radar work moves more geometry into the radar stream.

**Takeaway:** Sensor unification should standardize the interface to radar without discarding the attributes and error modes that make radar valuable.
