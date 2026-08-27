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

### Method and reported result

SOLOFusion frames temporal camera detection as stereo with a moving baseline. Short, high-resolution history resolves fine correspondence; long, low-resolution history supplies larger baselines that make distant depth easier to observe. The model combines both rather than spending high resolution over the full temporal window.

## Summary

> Its central insight is a budget trade: temporal baseline and feature resolution can compensate for each other, but neither should be maximized uniformly.

## Core Insights

The long-term path warps a running sequence of low-resolution BEV features into the present frame and builds a cost volume. The short-term path performs higher-resolution stereo around a few depth hypotheses chosen from the monocular depth distribution. Gaussian-spaced top-k sampling concentrates compute near likely depth while retaining nearby alternatives.

![SOLOFusion: Time Will Tell for Temporal Multi-View 3D Object Detection source figure: The framework of SOLOFusion.](/assets/images/solofusion-temporal-multiview-3d-object-detection-paper-figure.webp)
*Fig 1: SOLOFusion combines high-resolution short-term plane-sweep fusion for depth and BEV features with low-resolution long-term warping across earlier frames. | source: [SOLOFusion: Time Will Tell for Temporal Multi-View 3D Object Detection](https://arxiv.org/abs/2210.02443)*

![Figure 5 from SOLOFusion: Time Will Tell for Temporal Multi-View 3D Object Detection](/assets/images/solofusion-temporal-multiview-3d-object-detection-source-figure-5.webp)
*Fig 2: Visualization of relative increase in localization potential from using multiple timesteps. Note that each camera heatmap has a different scale. | source: [SOLOFusion: Time Will Tell for Temporal Multi-View 3D Object Detection](https://arxiv.org/abs/2210.02443)*

![Figure 1 from SOLOFusion: Time Will Tell for Temporal Multi-View 3D Object Detection](/assets/images/solofusion-temporal-multiview-3d-object-detection-source-figure-1.webp)
*Fig 3: The depth hypothesis projections onto the source view are further apart, making multi-view depth estimation easier when compared to the source view. | source: [SOLOFusion: Time Will Tell for Temporal Multi-View 3D Object Detection](https://arxiv.org/abs/2210.02443)*


| Temporal design | FPS | Memory | mAP | NDS |
| --- | ---: | ---: | ---: | ---: |
| Single-frame BEVDepth | 17.6 | 3.3 GB | 32.1 | 34.9 |
| Short high-resolution only | 12.2 | 3.3 GB | 34.3 | 38.9 |
| Long low-resolution only | 15.9 | 3.6 GB | 38.6 | 47.9 |
| Both paths | 11.4 | 3.6 GB | 40.4 | 49.5 |

History length matters nonlinearly. Moving from one to sixteen previous frames raises mAP from 31.6 to 37.7 and improves translation error from 0.734 to 0.655; 41 frames no longer helps. Matching every one of 112 depth hypotheses falls to 2.9 FPS and 8.5 GB, while seven guided hypotheses keep the memory at 3.3 GB.

On nuScenes validation, the ResNet-50 model reports 42.7 mAP and 53.4 NDS. The larger test model reaches 54.0 mAP and 61.9 NDS. These results support the representation trade, but the pipeline still carries dense BEV state and its history depends on ego-motion alignment.

## High-Level Takeaways

- SOLOFusion informs how to spend a fixed temporal budget between spatial detail and baseline diversity. Its atomic units are depth hypotheses and aligned BEV cells. The short path buys local precision; the long path buys observability and velocity without replaying high-resolution images at every timestamp.
- The matched experiment should sweep resolution, frame spacing, and history under one latency and memory envelope, with range-bucketed depth and actor recall. SOLOFusion loses if a sparse recurrent query model matches long-range localization with less dense state, or if pose drift makes long baselines unreliable. At 10× history, stored BEV features and warp bandwidth become the limits even when each frame is coarse.
- BEVDet4D uses a short recurrent BEV. SOLOFusion explains why long temporal baselines help camera depth; Sparse4D v2 and StreamPETR instead compress long history into recurrent instances.
- Long history is most useful when it changes depth observability; the efficient design spends detail on the short baseline and duration on the coarse one.
