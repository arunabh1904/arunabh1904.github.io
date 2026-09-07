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

> iBOT makes masked image modeling a self-distillation problem with an online visual tokenizer. A momentum teacher sees an unmasked view and supplies soft patch-token distributions; a student sees masked patches and predicts those targets. A DINO-style cross-view class-token loss gives the teacher global semantics, while the patch loss preserves local structure. The tokenizer is learned jointly with the representation, so the method avoids a separately pretrained discrete codebook.

## Core Insights

### An EMA teacher turns masked prediction into soft token distillation

For two augmented views, the student receives block-masked patches and the teacher receives the corresponding unmasked views. When multi-crop is enabled, the default recipe uses two 224×224 global crops and ten 96×96 local crops. iBOT applies MIM to the global crops only and randomizes the prediction ratio: a zero ratio makes a sample DINO-like, while a positive ratio masks both global crops. The class-token branch distills across views. The masked-image branch compares the student's outputs at masked locations with the teacher's soft patch distributions at those same locations. The teacher is an exponential-moving-average copy of the student, so it changes during pre-training instead of remaining a fixed tokenizer.

![iBOT combines cross-view class-token distillation with in-view masked patch-token distillation.](/assets/images/ibot-image-bert-pre-training-online-tokenizer-paper-figure.png)
*Fig 1: The student is masked while the momentum teacher supplies class-token and patch-token targets; the two losses are cross-view semantic alignment and in-view masked prediction. | source: [iBOT, Figure 3](https://arxiv.org/abs/2111.07832)*

The online target addresses two limitations of earlier masked modeling. Pixel reconstruction spends capacity on color and texture detail, while a frozen discrete tokenizer imports a vocabulary trained with another architecture and dataset. iBOT instead lets the target representation become more semantic through class-token self-distillation. The projection heads for class and patch tokens share parameters, allowing that global semantic signal to shape the patch-token space. The target remains a distribution rather than a hard token id, which preserves ambiguity in image patches.

![Masked image modeling uses a visual tokenizer to provide targets for the student's hidden patches.](/assets/images/ibot-image-bert-pre-training-with-online-tokenizer-source-figure-2.webp)
*Fig 2: Masked patches are predicted from a tokenizer's visual token distributions rather than from raw pixels. | source: [iBOT, Figure 2](https://arxiv.org/abs/2111.07832)*

### Soft patch targets support both classification and dense transfer

The default setup uses ViT-S/16, ViT-B/16, ViT-L/16, or Swin-T, 224×224 images, 196 patch tokens for ViT, and an 8192-dimensional shared three-layer projection head. ImageNet-1K pre-training uses batch size 1024; ViT-S runs for 800 epochs, ViT-B for 400, and ViT-L for 250. The masking ratio is zero with probability 0.5 and uniformly sampled from 0.1 to 0.5 otherwise.

| Pre-training and model | Linear probe | ImageNet fine-tune |
| --- | ---: | ---: |
| ImageNet-1K, ViT-L/16 | 81.0% | 84.8% |
| ImageNet-22K, ViT-L/16, 224 px | 82.3% | 86.6% |
| ImageNet-22K, ViT-L/16, 512 px | — | 87.8% |

The scale result needs its protocol attached: the 82.3% linear score uses frozen features, while 86.6% and 87.8% are supervised fine-tuning results under the ImageNet-22K setup. On ImageNet-1K, iBOT ViT-S/16 reaches 77.9% linear probing and 82.3% fine-tuning; ViT-B/16 reaches 79.5% and 84.0%. With only 1% and 10% labels, the small model reaches 61.9% and 75.1%, compared with DINO's 60.3% and 74.3%.

The dense transfer result supports the patch-level claim. With Cascade Mask R-CNN on COCO, iBOT ViT-S/16 reaches 49.4 box AP and 42.6 mask AP, compared with 46.2 and 40.1 for the supervised ViT-S baseline. On ADE20K with UPerNet, iBOT ViT-B reaches 50.0 mIoU, compared with 46.8 for DINO and 46.6 for the supervised baseline. The paper's visual analysis shows patch tokens grouping parts such as vehicle headlights and dog ears, as well as textures such as stripes and curls.

![Linear probing accuracy rises with iBOT model size in the paper's ImageNet comparison.](/assets/images/ibot-image-bert-pre-training-with-online-tokenizer-source-figure-1.webp)
*Fig 3: ImageNet linear accuracy versus parameter count for iBOT and other self-supervised baselines. | source: [iBOT, Figure 1](https://arxiv.org/abs/2111.07832)*

### The tokenizer depends on a moving teacher and a matched protocol

The online teacher creates a moving-target risk: a weak teacher can propagate a weak patch vocabulary. It also adds a teacher forward pass on unmasked views, but the EMA teacher receives no backward pass, so saying that it doubles representation compute is too strong; the overhead depends on the crop and masking pipeline. The ablations show why soft targets matter: with the paper's small ViT-S setting, iBOT reaches 69.1 k-NN and 74.2 linear accuracy, while the DINO comparison is 67.9 and 72.5; hard-label variants are lower. The strongest headline gains also change pre-training data, model size, resolution, and fine-tuning recipes, so they are not a single clean attribution of patch masking. The method's value is most directly tested by matched comparisons of pixel targets, a frozen tokenizer, and online soft targets under the same backbone and schedule.

## High-Level Takeaways

- iBOT couples global cross-view self-distillation with local masked patch prediction.
- Its online tokenizer learns with the model and emits soft distributions, which retain ambiguity that hard visual codebooks discard.
- The dense-transfer results support the claim that patch tokens preserve spatial semantics, but the large ImageNet headline numbers also change data and fine-tuning protocol.
- Momentum teachers stabilize a useful target while introducing extra compute and a risk of reinforcing representation errors.
