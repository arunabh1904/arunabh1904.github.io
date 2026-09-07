---
title: 'End-to-End Object Detection with Transformers'
date: '2020-05-26T00:00:00.000Z'
section: paper-shorts
postSlug: end-to-end-object-detection-with-transformers
legacyPath: /paper shorts/2020/05/26/end-to-end-object-detection-with-transformers.html
tags: [Other]
field: 'Vision Foundations'
summary: '2020 – DETR: object detection as direct set prediction'
---
## 2020 – DETR

**arXiv:** [2005.12872](https://arxiv.org/abs/2005.12872)

**Code:** [facebookresearch/detr](https://github.com/facebookresearch/detr)

## Summary

> DETR treats detection as prediction of a fixed-size unordered set. A CNN creates an image feature map; a transformer encoder supplies global context; learned object queries are decoded in parallel into boxes and classes; and Hungarian matching gives each target one owner. This removes proposal engineering and NMS from the inference contract. On COCO validation, the ResNet-50 model reaches 42.0 AP with 86 GFLOPs and 28 FPS, close to the 40.2 AP Faster R-CNN-FPN baseline with the same backbone and 180 GFLOPs. The cost is a long optimization schedule and weaker small-object AP.

## Core Insights

### Mechanism

The backbone feature map is flattened and combined with spatial positional encodings. The encoder lets every feature location reason globally; the decoder starts from N learned object queries and updates them through self-attention and encoder-decoder attention. A shared feed-forward head predicts a class, including a no-object class, and normalized center, width, and height for each slot.

Training makes the set contract explicit. Hungarian matching chooses the permutation of predictions that minimizes a cost built from class probability, L1 box distance, and generalized IoU. The matched pairs receive the classification and box losses; unmatched slots are trained toward no-object, with that class downweighted. Auxiliary losses after decoder layers help the slots become useful before the final layer.

![DETR predicts a set of detections in parallel and uses bipartite matching during training.](/assets/images/end-to-end-object-detection-with-transformers-paper-figure.webp)
*Fig 1: The model turns detection into parallel set prediction, with bipartite matching assigning predictions to ground-truth boxes. | source: [End-to-End Object Detection with Transformers, Figure 1](https://arxiv.org/abs/2005.12872)*

This is why NMS is absent from the model definition. Duplicate suppression is learned through the matching loss and decoder self-attention. The queries are not fixed semantic labels: they are exchangeable slots whose meanings emerge from the training distribution.

### Evidence

The main COCO validation table reports:

| Model | AP | AP$_S$ | AP$_L$ | GFLOPs | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Faster R-CNN-FPN | 40.2 | 24.2 | 52.0 | 180 | 26 |
| DETR | 42.0 | 20.5 | 61.1 | 86 | 28 |
| DETR-DC5 | 43.3 | 22.5 | 61.1 | 187 | 12 |
| DETR-R101 | 43.5 | 21.9 | 61.8 | 152 | 20 |

The comparison reveals the shape of the trade-off. DETR's large-object AP is 7.8 points above Faster R-CNN-FPN, but its small-object AP is 5.5 points lower. Doubling the feature resolution in DETR-DC5 improves small-object performance while making encoder attention much more expensive. The baseline ablations use a 300-epoch schedule with a learning-rate drop at 200; the long schedule used for the main Faster R-CNN comparison runs for 500 epochs with a drop at 400 and adds 1.5 AP. Training the baseline takes about three days on 16 V100 GPUs.

The encoder ablation gives a more causal explanation than the headline AP. With zero encoder layers, AP is 36.7; three layers reach 40.1, six reach 40.6, and twelve reach 41.6. The authors interpret the gain as global scene reasoning that separates instances before the decoder extracts them. Later decoder layers add 8.2 AP between the first and final layer. NMS helps the first layer, where queries have not yet communicated, but slightly hurts the final layers by removing true positives.

![Each decoder slot develops modes for locations and box sizes across the COCO validation set.](/assets/images/end-to-end-object-detection-with-transformers-source-figure-7.webp)
*Fig 2: Across images, the 20 shown query slots specialize in different center locations and box-size modes, including a common image-wide-box mode. | source: [End-to-End Object Detection with Transformers, Figure 7](https://arxiv.org/abs/2005.12872)*

### Boundary

A fixed query budget bounds how many objects can be represented, and convergence is sensitive to the optimizer, schedule, augmentation, and encoder resolution. The paper's evidence is 2D COCO detection; a learned query has no metric position or camera calibration. The panoptic extension shows that the set can feed a shared mask head, but it does not make the original detector a 3D or autonomous-driving system.

![DETR-R101 produces aligned masks for things and stuff with a small panoptic head.](/assets/images/end-to-end-object-detection-with-transformers-source-figure-9.webp)
*Fig 3: DETR-R101's panoptic head produces unified qualitative predictions for object and stuff regions. | source: [End-to-End Object Detection with Transformers, Figure 9](https://arxiv.org/abs/2005.12872)*

## High-Level Takeaways

- DETR's contribution is the training contract between a fixed set of queries and one-to-one matching, not the use of attention by itself.
- The contract removes NMS and proposal heuristics, but shifts work into matching, query learning, and long optimization.
- Global encoder attention helps separate instances; the ablation rises from 36.7 AP with no encoder to 41.6 with twelve layers.
- The large-object gain and small-object deficit explain why later DETR variants add multi-scale or sparse feature sampling.
