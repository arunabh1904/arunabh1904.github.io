---
title: 'EfficientDet: Scalable and Efficient Object Detection'
date: '2020-04-01T04:00:00.000Z'
section: paper-shorts
postSlug: efficientdet-scalable-and-efficient-object-detection
legacyPath: >-
  /paper
  shorts/2020/04/01/efficientdet-scalable-and-efficient-object-detection.html
tags:
  - Other
field: 'Vision Foundations'
summary: EfficientDet paired BiFPN feature fusion with compound scaling to make detector accuracy and latency easier to trade off.
---
## 2020 – EfficientDet: Scalable and Efficient Object Detection

**arXiv:** [1911.09070](https://arxiv.org/abs/1911.09070)

**GitHub:** [google/automl](https://github.com/google/automl/tree/master/efficientdet)

**Project page / Google AI Blog:** [EfficientDet: Towards Scalable and Efficient Object Detection](https://research.google/blog/efficientdet-towards-scalable-and-efficient-object-detection/)

**Conference:** CVPR 2020

## Paper Insights

EfficientDet adapts EfficientNet-style scaling to object detection. Its first ingredient is BiFPN, a weighted bidirectional feature pyramid that repeatedly fuses multi-scale features while learning how much each input matters. Its second ingredient is compound scaling across backbone, feature network, prediction heads, and input resolution. The model family D0 through D7 gives a smooth compute-accuracy ladder. The evidence is COCO detection performance versus FLOPs and parameters, where EfficientDet reaches strong accuracy at much lower cost than prior detectors. The caveat is that efficiency depends on the whole recipe: backbone, feature fusion, scaling, and training settings all contribute. The paper matters because it made detector scaling systematic rather than hand-tuned.

![Figure 2 from EfficientDet: feature pyramid variants compared against BiFPN](/assets/images/efficientdet-paper-figure-2-bifpn.png)
_Figure 2 from the [EfficientDet paper](https://arxiv.org/abs/1911.09070), via ar5iv._

**Summary:** EfficientDet asks the same question for object detection that EfficientNet asked for classification: how far can careful scaling take a CNN family? The authors build on EfficientNet backbones, add a Bi-directional Feature Pyramid Network (BiFPN) for fast multi-scale feature fusion, and scale the entire detector with one coefficient $\phi$.

BiFPN is the key detector-specific piece. It learns lightweight per-edge weights, so the network can emphasize useful feature paths instead of relying on hand-designed merge heuristics. Compound scaling then grows depth, width, and input resolution across the backbone, feature network, and prediction heads. The result is the D0-D7 family, where each model shares one recipe but targets a different FLOP and latency budget.

**Evals / Latency benchmarks (COCO test-dev, single scale):**

| Model | Params | FLOPs | AP | GPU latency† | CPU latency† |
| ----- | ------ | ----- | --- | ------------ | ------------ |
| EfficientDet-D0 | 3.9 M | 2.5 B | 33.8 | 16 ms | 0.32 s |
| EfficientDet-D1 | 6.6 M | 6.1 B | 39.6 | 20 ms | 0.74 s |
| EfficientDet-D4 | 21 M | 55 B | 49.4 | 74 ms | 4.8 s |
| EfficientDet-D7 | 52 M | 325 B | 52.2 | 262 ms | 24 s |

Compared with its contemporaries, D0 matches YOLOv3 accuracy with 28× fewer FLOPs, while D7 beats AmoebaNet + NAS-FPN by 1.5 AP using far fewer parameters and multiply-adds.

†Latency measured on Titan V GPU and single-thread Xeon CPU in the paper's ablation study.

**Critiques & limitations:** EfficientDet's strength is its AP-per-FLOP story. BiFPN removes a lot of manual feature-fusion guesswork, and compound scaling gives practitioners a ready-made detector suite for different devices. The pipeline is still anchor-based and more complex than many modern anchor-free detectors. It also depends heavily on EfficientNet backbones, and transformer-based detectors now surpass its top-end accuracy at very large budgets.

## Decision Lens

EfficientDet informs how to allocate a detector's latency budget across backbone size, feature fusion, input resolution, and prediction heads. The operative units are multiscale feature-map locations and anchors, not isolated image patches.

BiFPN compresses a pyramid of backbone features through repeated, learned weighted fusion, while compound scaling grows the full detector rather than one component. The reported model family establishes a favorable accuracy-efficiency frontier across several budgets; it does not prove that the same scaling coefficients remain optimal on different hardware or anchor-free heads.

The key missing study is a latency-matched factorial ablation of BiFPN topology, fusion weights, resolution, backbone depth/width, and head capacity. At 10× resolution, activation memory and pyramid fusion dominate cost. Compound scaling would be falsified as the useful design rule if a single-axis or hardware-aware scaling policy consistently matched AP at lower measured latency and energy.

**Takeaway:** EfficientDet showed that scaling every part of the detector matters as much as making it big. Thoughtful feature fusion and end-to-end scaling unlocked large gains in speed and accuracy and continue to influence modern detection pipelines.
