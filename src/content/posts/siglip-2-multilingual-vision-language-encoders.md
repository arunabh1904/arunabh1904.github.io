---
title: 'SigLIP 2: Multilingual Vision-Language Encoders with Dense Features'
date: '2025-02-20T00:00:00.000Z'
section: paper-shorts
postSlug: siglip-2-multilingual-vision-language-encoders
legacyPath: /paper shorts/2025/02/20/siglip-2-multilingual-vision-language-encoders.html
tags:
  - Vision-Language Models
  - Contrastive Learning
field: 'Vision-Language Models'
summary: '2025 – SigLIP 2: Multilingual Vision-Language Encoders with Dense Features'
---

## 2025 – SigLIP 2: Multilingual Vision-Language Encoders with Dense Features

**arXiv:** [2502.14786](https://arxiv.org/abs/2502.14786)

**Models and code:** [google-research/big_vision](https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/image_text)

## Summary

> SigLIP 2 keeps SigLIP's pairwise sigmoid image–text objective and widens the training signal around it. A captioning and referring-expression decoder teaches language and localization, self-distillation and masked prediction preserve local features, and a 109-language mixture plus NaFlex improves coverage of multilingual and aspect-sensitive inputs. The paper reports gains over SigLIP across retrieval, classification, dense prediction, and localization, but because these ingredients are introduced together the aggregate improvement is easier to observe than to attribute to one loss.

## Core Insights

### One encoder is trained against several kinds of evidence

SigLIP turns every image–text pairing in a batch into binary matching problems and optimizes a sigmoid loss rather than a softmax contrastive denominator. SigLIP 2 retains that interface so an existing dual-encoder application can swap weights. The new recipe adds a LocCa-style transformer decoder over unpooled visual features for captioning, dense captioning, and referring expressions. During the last 20% of pretraining, local-to-global self-distillation and masked patch prediction add pressure to keep individual patches meaningful instead of only making the pooled image embedding useful.

![Figure 1: SigLIP 2 training recipe](/assets/images/siglip-2-paper-figure-1.svg)
*Fig 1: This source Figure 1 combines SigLIP's sigmoid alignment with decoder-based captioning and referring expressions, then adds self-distillation and masked prediction for local features. | source: [SigLIP 2: Multilingual Vision-Language Encoders, Figure 1](https://arxiv.org/abs/2502.14786)*

The data mixture is also part of the representation. WebLI contributes 10 billion images and 12 billion alt-texts across 109 languages; the default mixture is 90% English web pages and 10% non-English pages, with filtering for representation bias. The multilingual proportion is modest, but it prevents a separate language-specific encoder from being the only route to cross-lingual retrieval.

### Preserve the image's geometry when the task needs it

The fixed-resolution checkpoints still resize images to square grids. NaFlex instead chooses a sequence length, resizes to a patch grid that stays close to the input aspect ratio, interpolates the positional embedding to that non-square grid, and masks padding. One checkpoint can therefore process several resolutions and aspect ratios. This is a systems choice with a representation consequence: a document or screen can spend tokens on its long axis without first warping the page.

![Figure 2: Per-language retrieval on Crossmodal-3600](/assets/images/siglip-2-multilingual-vision-language-encoders-source-figure-2.webp)
*Fig 2: This source Figure 2 compares per-language image–text retrieval for SigLIP, SigLIP 2, and mSigLIP across Crossmodal-3600's 36 languages; SigLIP 2 nearly matches the multilingual specialist while retaining stronger English results. | source: [SigLIP 2: Multilingual Vision-Language Encoders, Figure 2](https://arxiv.org/abs/2502.14786)*

![Figure 3: NaFlex and fixed-resolution SigLIP 2](/assets/images/siglip-2-multilingual-vision-language-encoders-source-figure-3.webp)
*Fig 3: This source Figure 3 compares one aspect-preserving NaFlex checkpoint with separate square-input checkpoints as sequence length changes; NaFlex interpolates between trained resolutions but does not extrapolate reliably beyond them. | source: [SigLIP 2: Multilingual Vision-Language Encoders, Figure 3](https://arxiv.org/abs/2502.14786)*

The plots show why “variable resolution” needs a boundary. NaFlex wins on most OCR and screen retrieval tasks, where distortion is costly. On natural-image benchmarks, the distilled fixed-resolution B model can still win, and the authors explicitly report poor extrapolation beyond the sequence lengths used in training. Flexibility reduces checkpoint count; it does not remove the need to train the resolution regime a deployment will use.

### Dense features are the decisive transfer test

| Frozen encoder | PASCAL mIoU | ADE20K mIoU | NYUv2 depth RMSE |
| --- | ---: | ---: | ---: |
| SigLIP So/14, 224px | 72.0 | 37.6 | 0.576 |
| SigLIP 2 So/14, 224px | 77.1 | 41.8 | 0.493 |
| SigLIP So/14, 384px | 73.8 | 40.8 | 0.563 |
| SigLIP 2 So/14, 384px | 78.1 | 45.4 | 0.466 |

These are frozen-representation probes, with segmentation reported as mIoU and depth as RMSE. The gain is not merely a larger pooled embedding: the patch features support better segmentation and depth. Localization shows the same shape. At 256 tokens, the B model's RefCOCO validation accuracy rises from 64.05 for SigLIP to 83.76 for SigLIP 2; the L model rises from 67.33 to 86.04. The decoder-based pretraining is a plausible mechanism, but the paper's combined recipe does not isolate how much comes from LocCa versus self-distillation, masked prediction, multilingual data, or curation.

## High-Level Takeaways

- SigLIP 2 preserves the cheap dual-encoder interface while training the visual backbone against language, local patches, and grounded regions.
- The strongest evidence for the expanded objective is in dense and localization transfer: at 224px, ADE20K rises from 37.6 to 41.8 mIoU and NYUv2 RMSE falls from 0.576 to 0.493.
- NaFlex is useful when aspect ratio matters, especially for OCR and screens, but the paper reports interpolation rather than reliable resolution extrapolation.
- Multilingual coverage and debiasing are data-mixture decisions; they improve breadth, but the aggregate paper does not provide a clean one-ingredient attribution for the full gain.
