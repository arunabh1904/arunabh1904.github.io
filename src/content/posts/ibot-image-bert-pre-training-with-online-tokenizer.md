---
title: 'iBOT: Image BERT Pre-Training with Online Tokenizer'
date: '2021-11-15T00:00:00.000Z'
section: paper-shorts
postSlug: ibot-image-bert-pre-training-with-online-tokenizer
legacyPath: /paper shorts/2021/11/15/ibot-image-bert-pre-training-with-online-tokenizer.html
tags:
  - Self-Supervised Learning
  - Masked Image Modeling
field: 'Vision Foundations'
topics:
  - learning
summary: '2021 – iBOT: Image BERT Pre-Training with Online Tokenizer'
---

## 2021 – iBOT: Image BERT Pre-Training with Online Tokenizer

**arXiv:** [2111.07832](https://arxiv.org/abs/2111.07832)

**Code:** [bytedance/ibot](https://github.com/bytedance/ibot)

**Conference:** ICLR 2022

## Summary

> iBOT turns masked image modeling into self-distillation at the patch level. Instead of reconstructing RGB pixels or committing to a separately trained visual tokenizer, a momentum teacher converts the unmasked image into soft patch targets while the student sees masked patches. The tokenizer is therefore “online”: its vocabulary changes with the representation being learned.

## Core Insights

![iBOT combines cross-view class-token self-distillation with masked patch-token prediction from an online teacher](/assets/images/ibot-image-bert-pre-training-online-tokenizer-paper-figure.png)
_The class-token branch preserves DINO-style cross-view alignment; the masked-image branch asks student patch tokens to match teacher patch distributions at the hidden locations. Source: [iBOT](https://arxiv.org/abs/2111.07832)._

The student and teacher share a ViT architecture. The student receives a corrupted view; the exponential-moving-average teacher receives the corresponding unmasked view. A class-token loss aligns different crops of the same image, while a masked-image loss aligns the student's hidden patch tokens with the teacher's soft patch distributions at the same spatial positions. The two heads share parameters, tying global semantics to local prediction.

This target choice avoids a recurring masked-modeling dilemma. Pixel reconstruction spends capacity on color and texture details that may not matter downstream. A frozen discrete tokenizer imports someone else's representation and failure modes. iBOT instead lets the target improve during training. The risk is circularity: if the teacher's patch partition is poor, the student faithfully reproduces a poor target. Momentum, centering, and global self-distillation stabilize that loop.

| Pretraining and model | ImageNet linear | ImageNet fine-tune | Interpretation |
| --- | ---: | ---: | --- |
| ImageNet-1K, ViT-L/16 | 81.0% | 84.8% | Strong local and global features without external labels |
| ImageNet-22K, ViT-L/16, 224 px | 82.3% | 86.6% | Larger data is important for the larger backbone |
| ImageNet-22K, ViT-L/16, 512 px | — | 87.8% | Resolution adaptation adds a further supervised fine-tuning gain |

The paper also evaluates object detection, instance segmentation, semantic segmentation, and low-shot classification. Those transfers are central to the method's claim: predicting teacher patch tokens should preserve spatial structure better than a global-only objective. The evidence supports broad transfer, but mixes changes in data, backbone size, resolution, and fine-tuning protocol across tables, so the headline results are not one clean attribution study.

## High-Level Takeaways

- iBOT informs what a masked vision model should predict. Its atomic unit is a masked patch position paired with a soft target from an unmasked momentum teacher. That makes it attractive when dense transfer matters and a fixed visual codebook would be expensive or domain-mismatched.
- The critical control is target quality at equal compute. Pixel reconstruction, a frozen tokenizer, and an online tokenizer should be compared with the same backbone, augmentations, schedule, and downstream protocol. The paper supplies many such ablations, but the best scale result also benefits from ImageNet-22K. At ten times the data, teacher inference and storing two networks become systems costs, while target drift can complicate distributed training.
- iBOT supplies the patch-level half of the later [DINOv2](/paper%20shorts/2023/04/14/dinov2-learning-robust-visual-features-without-supervision.html) recipe. DINOv2 keeps DINO's image-level agreement, uses iBOT-style masked patch prediction, and shifts much more attention to data curation and training stability.
- iBOT joins DINO-style global self-distillation with BERT-like masked prediction over image patches.
- Its strongest large-model result depends on ImageNet-22K, and an online teacher can propagate its own representation errors.
- Masked image modeling need not reconstruct pixels or use a frozen codebook; the model's momentum teacher can generate semantic patch targets while it learns.
