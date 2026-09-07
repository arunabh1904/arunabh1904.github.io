---
title: 'SOLOFusion: Time Will Tell for Temporal Multi-View 3D Object Detection'
date: '2022-10-05T04:00:00.000Z'
section: paper-shorts
postSlug: solofusion-temporal-multiview-3d-object-detection
legacyPath: /paper shorts/2022/10/05/solofusion-temporal-multiview-3d-object-detection.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2022 – Time Will Tell: New Outlooks and a Baseline for Temporal Multi-View 3D Object Detection (SOLOFusion)'
---
## 2022 – SOLOFusion

**arXiv:** [2210.02443](https://arxiv.org/abs/2210.02443)

**Code:** [Divadi/SOLOFusion](https://github.com/Divadi/SOLOFusion)

## Summary

> Its central insight is a budget trade: temporal baseline and feature resolution can compensate for each other, but neither should be maximized uniformly.

## Core Insights

### Spend resolution on the short baseline and time on the long one

SOLOFusion frames temporal camera detection as stereo with a moving baseline. The short-term path uses high-resolution features from two nearby frames and matches a small set of depth hypotheses. The long-term path warps a running sequence of low-resolution BEV features into the current frame and builds a cost volume. The combination gives fine correspondence where image detail matters and larger baselines where distant depth needs more parallax.

The architecture figure is a two-speed pipeline. Follow the short path through high-resolution plane-sweep features and the long path through low-resolution BEV warping, then see them meet before detection. Gaussian-spaced top-k sampling concentrates short-term computation near likely depths while retaining nearby alternatives; the model does not match every possible depth plane at full resolution.

![SOLOFusion: Time Will Tell for Temporal Multi-View 3D Object Detection source figure: The framework of SOLOFusion.](/assets/images/solofusion-temporal-multiview-3d-object-detection-paper-figure.webp)
*Fig 1: High-resolution short-term plane-sweep fusion and low-resolution long-term BEV warping provide complementary temporal baselines before the detection head. | source: [SOLOFusion, Figure 2](https://arxiv.org/abs/2210.02443)*

![Figure 5 from SOLOFusion: Time Will Tell for Temporal Multi-View 3D Object Detection](/assets/images/solofusion-temporal-multiview-3d-object-detection-source-figure-5.webp)
*Fig 2: Localization potential increases as more timesteps create a larger temporal baseline; camera heatmaps use different scales, so compare trends within each panel. | source: [SOLOFusion, Figure 5](https://arxiv.org/abs/2210.02443)*

![Figure 1 from SOLOFusion: Time Will Tell for Temporal Multi-View 3D Object Detection](/assets/images/solofusion-temporal-multiview-3d-object-detection-source-figure-1.webp)
*Fig 3: Depth hypotheses project farther apart between frames as the temporal baseline grows, increasing the signal available to multi-view depth matching. | source: [SOLOFusion, Figure 1](https://arxiv.org/abs/2210.02443)*


### Long history improves observability before it saturates

The temporal comparison moves from single-frame BEVDepth at 32.1 mAP/34.9 NDS to short high-resolution only at 34.3/38.9, long low-resolution only at 38.6/47.9, and both paths at 40.4/49.5. The long path carries more information about depth because frame spacing changes the baseline, while the short path protects fine spatial correspondence.

History length is nonlinear: moving from one to sixteen previous frames raises mAP from 31.6 to 37.7 and improves translation error from 0.734 to 0.655; 41 frames no longer helps. Matching all 112 depth hypotheses falls to 2.9 FPS and 8.5 GB, while seven guided hypotheses keep memory at 3.3 GB. The top-k choice is therefore an accuracy–memory contract, not a cosmetic approximation.

On nuScenes validation, the ResNet-50 model reports 42.7 mAP and 53.4 NDS. The larger test model reaches 54.0 mAP and 61.9 NDS. These results support the representation trade, but the pipeline still carries dense BEV state and its history depends on ego-motion alignment.

### A temporal baseline is also a calibration test

Longer baselines amplify pose error, rolling-shutter mismatch, timestamp drift, and object motion. Report depth and detection by range, frame interval, and motion regime, and inject the actual inference delay. A sparse recurrent query model may match the long-term path with less dense memory, but it inherits a different object-birth contract.

## High-Level Takeaways

SOLOFusion is a temporal budget design: high-resolution short history supplies detail, low-resolution long history supplies baseline, and guided depth samples control memory. Its localization-potential curve and 112-to-7 hypothesis comparison make the trade measurable, while pose and synchronization bound the claim.

Sweep resolution, frame spacing, history, and depth hypotheses under one latency/memory envelope. Report range-bucketed depth, actor recall, pose drift, and inference delay. BEVDet4D supplies the short-history baseline; Sparse4D v2 and StreamPETR compress long history into recurrent instances rather than dense BEV state.
