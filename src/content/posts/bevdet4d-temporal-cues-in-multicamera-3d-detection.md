---
title: 'BEVDet4D: Temporal Cues in Multi-Camera 3D Detection'
date: '2022-03-31T04:00:00.000Z'
section: paper-shorts
postSlug: bevdet4d-temporal-cues-in-multicamera-3d-detection
legacyPath: /paper shorts/2022/03/31/bevdet4d-temporal-cues-in-multicamera-3d-detection.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2022 – BEVDet4D: align and fuse adjacent camera BEV features for motion-aware detection'
---
## 2022 – BEVDet4D

**arXiv:** [2203.17054](https://arxiv.org/abs/2203.17054)

**Code:** [HuangJunJie2017/BEVDet](https://github.com/HuangJunJie2017/BEVDet)

**Summary:** BEVDet4D adds a simple temporal path to BEVDet: transform the previous frame's BEV feature into the current ego frame, concatenate it with the current feature, and process the pair with an extra BEV encoder. Two adjacent observations turn object velocity into a spatial displacement that the detector can learn directly.

This is the dense-grid temporal baseline against which more elaborate BEV memories and sparse object memories should be judged.

## Paper Insights

Ego alignment is critical because a static scene otherwise appears to move with the vehicle. The extra encoder gives local convolutions a way to interpret paired features after warping. In the tiny configuration, the paper reports velocity error falling from 0.909 to 0.337 and NDS increasing from 39.2 to 47.6; the test result reaches 56.9 NDS.

| Component | Purpose | Failure mode |
| --- | --- | --- |
| Previous BEV | Supplies temporal evidence | Carries stale false positives. |
| Ego transform | Aligns static scene | Pose error accumulates. |
| Concatenation | Preserves both timestamps | Doubles temporal feature bandwidth. |
| Extra BEV encoder | Learns displacement cues | Dense cost grows with grid area. |

## Decision Lens

BEVDet4D is appropriate when one short history fixes velocity and flicker without the complexity of recurrent attention. Evaluate moving actors separately: ego warping aligns the road, not independently moving objects. Timestamp, rolling-shutter, and inference-delay errors should be injected explicitly.

Longer history is not automatically better. Dense grids consume memory linearly in spatial extent and can preserve actors after they leave the scene.

**Context:** BEVFormer uses temporal attention over a dense BEV; StreamPETR and Sparse4D instead keep object-centric state.

**Takeaway:** A warped previous BEV is a strong temporal baseline because it makes motion visible, but its state is dense, short-lived, and only ego-motion aligned.
