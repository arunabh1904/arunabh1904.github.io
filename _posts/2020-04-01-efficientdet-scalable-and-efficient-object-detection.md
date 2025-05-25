---
layout: content
title: "EfficientDet: Scalable and Efficient Object Detection"
date: 2020-04-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Computer Vision
---

## 2020 – EfficientDet: Scalable and Efficient Object Detection

**arXiv:** [1911.09070](https://arxiv.org/abs/1911.09070)

**GitHub:** [google/automl](https://github.com/google/automl/tree/master/efficientdet)

**Project page / Google AI Blog:** [EfficientDet: Towards Scalable and Efficient Object Detection](https://research.google/blog/efficientdet-towards-scalable-and-efficient-object-detection/)

**Conference:** CVPR 2020

**Summary (abstract in plain English):** EfficientDet revisits object-detector design under tight efficiency constraints. Building on EfficientNet backbones, the authors introduce Bi-directional Feature Pyramid Networks (BiFPN) for fast, weighted multi-scale fusion and a compound-scaling rule that grows depth, width and input resolution of all detector components with a single coefficient \(\phi\). This yields the D0→D7 family, each tuned to a specific FLOP or latency budget while sharing the same architecture and training recipe.

**Novel insights:**
- Weighted BiFPN learns per-edge weights so the network can emphasise informative paths with little overhead.
- Compound scaling extends EfficientNet's idea to the whole detector stack for smoother accuracy–efficiency trade-offs.
- One recipe fits many regimes, providing SOTA from 2.5 B FLOPs (D0) up to 325 B FLOPs (D7).

**Evals / Latency benchmarks (COCO test-dev, single scale):**

| Model | Params | FLOPs | AP | GPU latency† | CPU latency† |
| ----- | ------ | ----- | --- | ------------ | ------------ |
| EfficientDet-D0 | 3.9 M | 2.5 B | 33.8 | 16 ms | 0.32 s |
| EfficientDet-D1 | 6.6 M | 6.1 B | 39.6 | 20 ms | 0.74 s |
| EfficientDet-D4 | 21 M | 55 B | 49.4 | 74 ms | 4.8 s |
| EfficientDet-D7 | 52 M | 325 B | 52.2 | 262 ms | 24 s |

Compared with its contemporaries, D0 matches YOLOv3 accuracy with 28× fewer FLOPs, while D7 beats AmoebaNet + NAS-FPN by 1.5 AP using far fewer parameters and multiply-adds.

†Latency measured on Titan V GPU and single-thread Xeon CPU in the paper's ablation study.

**Critiques & limitations:**
- **What I liked:** Elegant BiFPN delivers accuracy and speed without manual feature-merge heuristics. Compound scaling supplies a ready-made detector suite for any device class. Strong AP-per-FLOP repositioned CNN detectors on the Pareto frontier. Open-sourced code and weights accelerated community adoption.
- **Limitations:** Anchor-based pipeline with focal loss remains complex next to modern anchor-free methods. Dependence on EfficientNet backbones means newer backbones need extra tuning. Transformers now surpass its top-end accuracy at very large budgets. Weighted edges can complicate some inference runtimes.

**Take-home message:** EfficientDet showed that scaling every part of the detector matters as much as making it big. Thoughtful feature fusion and end-to-end scaling unlocked large gains in speed and accuracy and continue to influence modern detection pipelines.

