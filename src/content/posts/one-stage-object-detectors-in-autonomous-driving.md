---
title: 'One-Stage Object Detectors in Autonomous Driving'
date: '2026-08-19T09:00:00.000Z'
section: paper-shorts
postSlug: one-stage-object-detectors-in-autonomous-driving
legacyPath: /paper shorts/2026/08/19/one-stage-object-detectors-in-autonomous-driving.html
tags:
  - Object Detection
  - Autonomous Driving
  - Survey
field: 'Vision Foundations'
summary: '2026 – a survey of one-stage detector design and the limits of cross-paper speed-accuracy comparisons'
---

## 2026 – One-Stage Object Detectors in Autonomous Driving

**arXiv:** [2608.19014](https://arxiv.org/abs/2608.19014)

## Summary

> This paper is a survey and comparative analysis, not a new detector. Its useful contribution is a decision map for one-stage design: assignment, feature fusion, score calibration, loss, post-processing, and scaling each move the speed-accuracy trade-off. Its reported AP, FPS, parameter, and FLOP values are collected from different papers and protocols, so they are evidence about the literature's design choices rather than a matched autonomous-driving leaderboard.

## Core Insights

### One-stage is a placement in the pipeline, not one architecture

“One-stage” describes where the proposal stage went; it does not identify one architecture. The survey's taxonomy separates default-box methods such as SSD, RetinaNet, and EfficientDet; grid-based YOLO variants; keypoint and center representations such as CornerNet and CenterNet; per-location anchor-free regression such as FCOS; and recent NMS-free training such as YOLOv10. The design decisions are orthogonal. A detector can change its label assignment while retaining a feature pyramid, or change its score calibration while keeping anchors.

![Chronology of the one-stage detectors organized by the survey.](/assets/images/one-stage-detectors-timeline.webp)
*Fig 1: The timeline shows branches of assignment, feature fusion, and efficiency design rather than one model replacing another. | source: [One-Stage Object Detectors in Autonomous Driving, Figure 1](https://arxiv.org/abs/2608.19014)*

The survey connects these choices to driving constraints. Multi-scale features and feature pyramids help distant pedestrians, cyclists, and signs; focal or quality-aware losses address the foreground-background and classification-localization imbalance; anchor-free heads remove anchor tuning but can create crowded-scene or keypoint-grouping problems; and NMS-free training targets post-processing latency. EfficientDet is used as the efficiency-oriented example, while YOLOX, FCOS, GFL, VFNet, RTMDet, and YOLOv10 illustrate other branches.

### The survey turns detector comparisons into a deployment checklist

The paper's comparison is most useful as a checklist for a real evaluation:

| Axis | What the survey records | What a deployment comparison must hold fixed |
| --- | --- | --- |
| Detection quality | AP or mAP from the original studies | Dataset, classes, IoU rule, resolution, augmentation |
| Throughput | FPS or latency where reported | Hardware, precision, batch size, runtime, pre/post-processing |
| Small objects | Feature-pyramid choices and reported weaknesses | Range-stratified recall for pedestrians, cyclists, and signs |
| Efficiency | Parameters, FLOPs, and model scale | Memory, energy, and worst-case latency on the target device |
| Driving relevance | KITTI, Waymo, nuScenes, BDD100K, Cityscapes, Argoverse | Same sensor contract and adverse-condition slices |

The survey explicitly warns that its speed-accuracy plot mixes sources. Its points are useful for locating candidate papers, but a point's horizontal position can reflect a different GPU, image size, precision mode, or definition of FPS. The qualitative radar makes the same point in another form:

![Qualitative radar comparison of representative one-stage detectors.](/assets/images/one-stage-object-detectors-in-autonomous-driving-source-figure-4.webp)
*Fig 2: The radar scores summarize reported accuracy, speed, efficiency, deployment readiness, and small-object handling on a qualitative five-point scale. | source: [One-Stage Object Detectors in Autonomous Driving, Figure 4](https://arxiv.org/abs/2608.19014)*

The source's Figure 3 is a reading aid, not a new experiment. A detector that appears far right on an FPS axis may still have unacceptable tail latency after decoding and NMS, while a high AP value from COCO says little about a rare, distant cyclist in rain.

![Speed-accuracy points collected across the survey's cited detector papers.](/assets/images/one-stage-detectors-speed-accuracy.webp)
*Fig 3: The survey's cross-paper speed-accuracy plot should be read as a literature map, because datasets, input sizes, hardware, and reporting conventions differ. | source: [One-Stage Object Detectors in Autonomous Driving, Figure 3](https://arxiv.org/abs/2608.19014)*

### Cross-paper plots cannot answer a safety question

The survey does not run a controlled benchmark on KITTI, Waymo, BDD100K, or another driving dataset, and it does not measure a single detector on a common device. Its strongest conclusion is therefore methodological: mAP and FPS should be joined by class-, range-, weather-, and occlusion-stratified recall, calibration, energy, and tail latency. The paper's future directions—small-object handling, adverse-weather robustness, edge optimization, deployment-centric metrics, and integration with tracking and planning—follow from that missing protocol.

## High-Level Takeaways

- One-stage detection is a family of independent choices about assignment, representation, fusion, loss, scaling, and post-processing.
- The survey is a useful taxonomy and reading list; its collected numbers do not rank detectors for an autonomous-driving stack.
- Anchor-free heads remove anchor design, but they do not guarantee lower latency or better crowded-scene localization.
- A defensible AV comparison needs one dataset, one sensor contract, one device, and explicit tail-latency and degradation slices.
- The next useful artifact is a matched benchmark that reports safety-critical per-class recall alongside AP, energy, calibration, and end-to-end latency.
