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

> This survey organizes one-stage detectors by assignment, representation, feature fusion, loss design, and deployment trade-offs. Its useful conclusion is methodological: AP, frames per second, parameters, and input size must be read together and within one dataset-hardware protocol. The paper does not train a detector or run a matched autonomous-driving benchmark; its performance table combines values from Pascal VOC and COCO, different resolutions, hardware, and reporting conventions. It is therefore a map of design choices, not evidence that one listed family is safest or fastest for a driving stack.

## Core Insights

### One-stage detection now names several design families

The original distinction was procedural: two-stage detectors generate proposals before classification and box refinement, while one-stage detectors predict dense classes and boxes directly. The survey shows how much diversity now sits inside the second category. SSD and RetinaNet use default boxes; early YOLO variants use grid-based anchors; FCOS predicts per-location boxes with centerness; CenterNet turns objects into keypoints; YOLOX combines an anchor-free head with dynamic assignment; and YOLOv10 removes non-maximum suppression through consistent one-to-many and one-to-one training assignments.

![Timeline of one-stage detector families from early YOLO and SSD through anchor-free and NMS-free designs](/assets/images/one-stage-detectors-timeline.webp)
*The chronology is best read as branching design decisions—assignment, feature fusion, loss, post-processing, and scaling—not as one model family replacing another. source: [One-Stage Object Detectors](https://arxiv.org/abs/2608.19014)*

![Figure 4 from One-Stage Object Detectors in Autonomous Driving](/assets/images/one-stage-object-detectors-in-autonomous-driving-source-figure-4.webp)
*Figure 4 Fig. 4: Radar chart comparing representative one-stage detectors across multiple dimensions. source: [One-Stage Object Detectors in Autonomous Driving](https://arxiv.org/abs/2608.19014)*


Feature fusion and optimization often matter as much as the output parameterization. [EfficientDet](/paper%20shorts/2020/04/01/efficientdet-scalable-and-efficient-object-detection.html) couples BiFPN with compound scaling, RetinaNet introduces focal loss for foreground-background imbalance, GFL represents box coordinates as distributions, and VFNet aligns classification confidence with localization quality. These changes move the deployment frontier without making anchor-based versus anchor-free a sufficient selection rule.

| Selection axis | What the survey records | Required deployment control |
| --- | --- | --- |
| Detection quality | AP or mAP from original papers | Same dataset, class set, IoU rule, resolution, and test-time augmentation |
| Throughput | FPS or latency when reported | Same hardware, precision, batch size, pre/post-processing, and runtime |
| Small-object behavior | Multi-scale features and reported weaknesses | Range-stratified pedestrian, cyclist, sign, and occlusion recall |
| Efficiency | Parameters, FLOPs, and model scale | Measured memory, energy, and worst-case latency on target hardware |
| Driving relevance | KITTI, Waymo, nuScenes, BDD100K, and Argoverse coverage | Per-class and adverse-condition results under the intended sensor contract |

### The comparison table is not a leaderboard

The survey explicitly marks its performance table as cross-paper reporting rather than a controlled benchmark. For example, it places YOLO and EfficientDet COCO numbers beside SSD and DSSD Pascal VOC numbers, with several missing FPS and parameter entries. Its qualitative radar chart is also a survey-derived five-point assessment, not a new measurement. The defensible use is to shortlist mechanisms and baselines before running a hardware- and dataset-matched evaluation.

![Cross-paper speed-accuracy scatter for one-stage detectors collected by the survey](/assets/images/one-stage-detectors-speed-accuracy.webp)
*This plot is a reading aid, not a leaderboard: points mix datasets, input sizes, hardware, runtimes, and reporting conventions, so their relative positions do not establish a matched speed-accuracy frontier. source: [One-Stage Object Detectors](https://arxiv.org/abs/2608.19014)*

The survey's autonomous-driving discussion identifies the right failure categories—small and distant objects, occlusion, adverse weather, edge compute, and missing deployment-centric metrics—but it does not quantify them under one protocol. A detector choice for driving should therefore be driven by critical-class recall, calibration, degradation tests, and end-to-end latency, not a global AP/FPS pair copied from unrelated papers.

## High-Level Takeaways

- One-stage detection is no longer one architecture: anchor assignment, feature pyramid design, score-localization alignment, post-processing, and scaling policy are independent decisions.
- The survey is most useful as a taxonomy and reading list. Its heterogeneous reported numbers cannot support a detector ranking for autonomous driving.
- A production comparison should hold the input, dataset, hardware, precision, runtime, and post-processing fixed, then report range- and condition-stratified recall for safety-critical classes.
- Anchor-free prediction removes anchor tuning but does not automatically improve latency, crowded-scene localization, or small-object recall.
- The survey's deployment claims would become decision-grade only after a matched benchmark on driving data and target hardware, including tail latency, energy, calibration, weather, and occlusion slices.
