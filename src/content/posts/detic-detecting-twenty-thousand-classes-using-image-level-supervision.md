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

> Detic separates the two jobs inside a detector: boxes teach the model where objects are, while image-level labels teach its classifier what those objects are called. It applies image supervision to one stable, large proposal instead of guessing which predicted box owns an image label. With a ResNet-50 CenterNet2 baseline on open-vocabulary LVIS, the default max-size loss moves novel-class mask mAP from 16.3 to 24.6 and overall mask mAP from 30.0 to 32.4. The same idea scales to a 21K-class ImageNet classifier and transfers to Objects365 and OpenImages without fine-tuning, although it remains well below dataset-specific box-supervised models.

## Core Insights

### A larger vocabulary can reuse an existing localizer

The motivation is visible before the method is. LVIS has roughly 1,200 detection categories, while ImageNet-21K has 21,000 classes and about 14 million images. The paper's Figure 1 shows the mismatch as a long tail: box datasets run out of examples for many names just as image-classification data keeps going. The detector often already proposes a region that contains an unfamiliar object; its classifier simply has no useful supervision for naming it.

![Detic comparison of category coverage across detection, classification, and caption datasets](/assets/images/detic-paper-figure-1.png)
*Fig 1: LVIS has far fewer images per category than ImageNet and Conceptual Captions, motivating image-level supervision for vocabulary expansion. | source: [Detic, Figure 1](https://arxiv.org/abs/2201.02605)*

Detic therefore keeps localization and classification on separate diets. Images with boxes use the ordinary RPN, box-regression, and classification losses. An image-labeled example updates only the classifier: the method chooses the largest proposal and applies the image label to that feature. This choice is deliberately boring. Prediction-based weak supervision tries to select a proposal whose current class score is high, creating a chicken-and-egg loop: a weak detector makes a bad assignment, then learns from that bad assignment. The max-size proposal is more likely to cover the main object, and it tends to remain the same as the detector changes during training.

![Detic assigned boxes during training](/assets/images/detic-detecting-twenty-thousand-classes-using-image-level-supervision-source-figure-4.webp)
*Fig 2: Prediction-based training jumps between blue candidate boxes and can select a region that misses the object; Detic's red max-size assignment is more stable and usually covers it. On an annotated IN-L subset, max-size covered 92.8% of target objects versus 69.0% for the prediction-based method. | source: [Detic, Figure 4](https://arxiv.org/abs/2201.02605)*

### Stable weak assignments outperform predicted ones

That design explains the paper's most useful ablation. On open-vocabulary LVIS with the 997 ImageNet classes overlapping LVIS, the box-only baseline reaches 30.0 overall mask mAP and 16.3 novel-class mask mAP. The max-size variant reaches 32.4 and 24.6, respectively. Image-box is almost as strong at 32.4/23.8, while prediction-based methods top out at 31.2/20.4 in the reported table. The improvement is not magic localization from image labels: it comes from making the classifier recognize names that the proposal network can already place. For multi-object Conceptual Captions, the same max-size method reaches 30.9 overall and 19.5 novel mask mAP; the fact that it still works when the image is not a single centered object is useful, but it also shows why the assignment is only a weak localization signal.

The effect is not tied to CLIP weights. With a trained classifier that cannot recognize novel classes by itself, Detic still reaches 17.4 novel mask mAP after image-level co-training. FastText and OpenCLIP improve that to 19.2 and 19.4, while CLIP gives 24.9. In other words, the image-level loss is the contribution; language-derived classifier weights make the open-vocabulary starting point stronger.

### Vocabulary transfer still trails target-domain box supervision

The large-vocabulary experiment makes the boundary clear. The Swin-B model uses all 21K ImageNet classes and a modified Federated Loss that samples 50 vocabulary classes per iteration. Without retraining, it reaches 21.5 box mAP and 20.0 rare-class mAP on Objects365, and 55.2/68.8 box AP50 overall/rare on OpenImages. Those are 70–80% of the paper's dataset-specific oracles, which use box labels from the target datasets. Figure 5 shows why the result is still meaningful: the detector can name unfamiliar objects such as raccoons, potholes, and many varieties of food, but confidence and boundaries are uneven.

![Detic 21K-class qualitative detections](/assets/images/detic-detecting-twenty-thousand-classes-using-image-level-supervision-source-figure-5.webp)
*Fig 3: Qualitative 21K-class predictions on OpenImages and Objects365; purple labels are LVIS classes and green labels are novel classes. The examples show vocabulary transfer, not a claim that image labels supply precise boxes. | source: [Detic, Figure 5](https://arxiv.org/abs/2201.02605)*

## High-Level Takeaways

- The practical move is to widen the classifier while leaving box regression and proposal learning on detection data. This makes cheap image labels useful without pretending they contain coordinates.
- The max-size assignment wins because it avoids dependence on an already-good novel-class detector: on IN-L, it covers 92.8% of annotated target objects versus 69.0% for a prediction-based assignment.
- On open-vocabulary LVIS, image-level co-training raises novel mask mAP from 16.3 to 24.6 with the default max-size loss, while the fully box-supervised reference is 25.5. That reference uses stronger supervision; the comparison does not isolate the cause of the remaining gap.
- Detic's 21K-class transfer is best read as vocabulary breadth with partial localization transfer: 20.0 rare-class mAP on Objects365 is useful, but the 22.5 dataset-specific oracle shows where boxes still matter.
