---
title: 'EfficientDet: Scalable and Efficient Object Detection'
date: '2019-11-20T00:00:00.000Z'
section: paper-shorts
postSlug: efficientdet-scalable-and-efficient-object-detection
legacyPath: >-
  /paper
  shorts/2020/04/01/efficientdet-scalable-and-efficient-object-detection.html
tags:
  - Other
field: 'Vision Foundations'
summary: "2020 – EfficientDet: Scalable and Efficient Object Detection"
---
## 2020 – EfficientDet: Scalable and Efficient Object Detection

**arXiv:** [1911.09070](https://arxiv.org/abs/1911.09070)

**GitHub:** [google/automl](https://github.com/google/automl/tree/master/efficientdet)

**Project page / Google AI Blog:** [EfficientDet: Towards Scalable and Efficient Object Detection](https://research.google/blog/efficientdet-towards-scalable-and-efficient-object-detection/)

**Conference:** CVPR 2020

## Summary

> EfficientDet makes detector scaling a joint design problem. BiFPN repeatedly fuses features from coarse and fine pyramid levels with learned normalized weights, while one compound coefficient scales the EfficientNet backbone, BiFPN, prediction heads, and input resolution together. The resulting D0–D7/D7x family moves along an explicit accuracy-compute curve. On COCO test-dev, D7x reaches 55.1 AP with 77M parameters and 410B FLOPs in a single-model, single-scale evaluation. The result depends on the whole recipe: feature fusion, backbone, scale, and training schedule.

## Core Insights

### BiFPN spends capacity on repeated cross-scale fusion

A plain FPN sends information top-down. PANet adds a bottom-up path, while NAS-FPN searches for a topology that is harder to interpret and expensive to discover. BiFPN keeps both directions, removes nodes that only have one input, adds same-level skip edges, and repeats the bidirectional block. Every feature is processed with depthwise separable convolution. The network therefore spends its capacity on fusing genuinely different resolutions rather than on pass-through nodes.

![EfficientDet compares top-down FPN, PANet, NAS-FPN, and the repeated bidirectional BiFPN topology.](/assets/images/efficientdet-paper-figure-2-bifpn.png)
*Fig 1: BiFPN combines top-down and bottom-up cross-scale paths, removes one-input nodes, and repeats a compact fusion block. | source: [EfficientDet, Figure 2](https://arxiv.org/abs/1911.09070)*

The fusion itself is learned. For inputs I_i, fast normalized fusion uses non-negative scalar weights and computes the weighted sum divided by the sum of weights plus a small epsilon. It behaves like softmax fusion but avoids the expensive exponentials. The ablation reports 1.26–1.31 times GPU speedup with almost unchanged AP. This is a useful systems detail: the feature pyramid is not only a graph of connections; its normalization rule affects deployment cost.

The compound equations describe the D0–D6 progression. D6 already uses 1280-pixel input, the B6 backbone, a 384-channel BiFPN, eight BiFPN layers, and five box/class layers, so the input formula should not be extrapolated unchanged to D7. D7 keeps B6 and that head but raises the input to 1536; D7x keeps the 1536-pixel input, swaps in B7, and adds the P8 feature level. These are deliberate endpoint choices, not a claim that one linear formula exactly generates every family member.

### The table separates detector scale from accelerator latency

The paper's Table 2 reports single-model, single-scale COCO results:

| Model | Test-dev AP | Params | FLOPs | Titan V latency | V100 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| EfficientDet-D0 (512) | 34.6 | 3.9M | 2.5B | 12 ms | 10.2 ms |
| EfficientDet-D1 (640) | 40.5 | 6.6M | 6.1B | 16 ms | 13.5 ms |
| EfficientDet-D4 (1024) | 49.7 | 21M | 55B | 65 ms | 42.8 ms |
| EfficientDet-D6 (1280) | 52.6 | 52M | 226B | 169 ms | 92.8 ms |
| EfficientDet-D7 (1536) | 53.7 | 52M | 325B | 232 ms | 122 ms |
| EfficientDet-D7x (1536) | 55.1 | 77M | 410B | 285 ms | 153 ms |

The headline D7x result is 4 AP above the earlier best detector in the paper's comparison and uses 7.4 times fewer FLOPs. In Figure 1's plotted checkpoints, D0 matches YOLOv3 at 28 times fewer FLOPs, while larger models continue to gain AP as the budget rises. The table above reports the later test-dev checkpoint values and keeps the two GPU latency columns separate.

The training schedule is part of the result. D0–D6 use 300 epochs on 32 TPUv3 cores; D7/D7x use 600 epochs on 128 TPUv3 cores. The source's validation curve shows EfficientDet-D1 moving from 34.6 AP at 30 epochs to 40.2 at 300 and 40.5 at 600, while RetinaNet-R50 moves from 35.5 to 39.2 and 39.5.

![EfficientDet's accuracy-compute frontier on COCO single-model, single-scale evaluation.](/assets/images/efficientdet-scalable-and-efficient-object-detection-source-figure-1.webp)
*Fig 2: EfficientDet moves from D0 to D7 along a higher AP-per-FLOP curve than the compared detectors. | source: [EfficientDet, Figure 1](https://arxiv.org/abs/1911.09070)*

![Validation AP versus training epochs for EfficientDet-D1 and RetinaNet-R50.](/assets/images/efficientdet-scalable-and-efficient-object-detection-source-figure-7.webp)
*Fig 3: Both models improve with longer training; EfficientDet-D1 continues improving through the 300- and 600-epoch schedules. | source: [EfficientDet, Figure 7](https://arxiv.org/abs/1911.09070)*

### Training recipe and device decide where the curve is useful

The ablations show that both parts matter. Under RetinaNet settings, replacing ResNet-50 with EfficientNet-B3 raises AP from 37.0 to 40.3; replacing FPN with BiFPN then raises it to 44.4. The paper's cross-paper leaderboard is still heterogeneous, and the authors reproduce RetinaNet with their trainer while taking several other baselines from their papers. Latency also depends on hardware and post-processing. D7x is efficient relative to its contemporaries, but 410B FLOPs and 285 ms on the reported Titan V are a poor fit for every edge system.

## High-Level Takeaways

- BiFPN's useful idea is repeated bidirectional fusion with learned, cheap normalization of cross-scale inputs.
- Compound scaling allocates capacity across backbone, pyramid, heads, and resolution instead of enlarging one component in isolation.
- D7x reaches 55.1 AP with 77M parameters and 410B FLOPs under the paper's single-model, single-scale COCO protocol.
- The accuracy curve is inseparable from long training and the EfficientNet backbone; the best scale is a hardware and data decision.
