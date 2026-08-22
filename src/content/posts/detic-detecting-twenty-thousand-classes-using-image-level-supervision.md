---
title: 'Detic: Detecting Twenty-thousand Classes using Image-level Supervision'
date: '2022-01-07T00:00:00.000Z'
section: paper-shorts
postSlug: detic-detecting-twenty-thousand-classes-using-image-level-supervision
legacyPath: /paper shorts/2022/01/07/detic-detecting-twenty-thousand-classes-using-image-level-supervision.html
tags:
  - Vision-Language Models
  - Open-Vocabulary Detection
field: 'Vision-Language Models'
summary: '2022 – Detic: Detecting Twenty-thousand Classes using Image-level Supervision'
---

## 2022 – Detic: Detecting Twenty-thousand Classes using Image-level Supervision

**arXiv:** [2201.02605](https://arxiv.org/abs/2201.02605)

**Code:** [facebookresearch/Detic](https://github.com/facebookresearch/Detic)

## Summary

> Detic expands a detector's vocabulary with image classification data, even when those classes have no bounding-box annotations. On open-vocabulary LVIS, the paper reports gains of 2.4 mAP across all classes and 8.3 mAP on novel classes. The same recipe trains a detector over all 21,000 ImageNet categories and transfers it to new datasets without fine-tuning.

## Core Insights

![Detic comparison of category coverage across detection, classification, and caption datasets](/assets/images/detic-paper-figure-1.png)
_Classification and caption datasets cover many more categories than box-annotated detection data. Detic uses that cheaper image-level supervision to train the detector's classifier. Source: [Detic](https://arxiv.org/abs/2201.02605), Figure 1._

The expensive label in detection is the box, not the category name. Detic keeps box-supervised data for localization, then uses image-level labels to expand the classifier. It avoids assigning every image label to a model-predicted box through a complex pseudo-labeling rule.

This cleanly separates two jobs. Box annotations teach where objects are. Image-level labels teach which categories the classifier can name. The tradeoff is that a class without boxes receives weak localization supervision, so vocabulary growth does not guarantee equally strong boxes for every category.

## High-Level Takeaways

- Detic uses cheap image labels to widen a detector without requiring boxes for every class.
- The reported LVIS gains are largest on novel classes, where the added vocabulary matters most.
- Image-level supervision expands recognition; localization quality still depends on box-supervised transfer.
