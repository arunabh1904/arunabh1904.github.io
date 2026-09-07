---
title: 'Learning Transferable Visual Models From Natural Language Supervision'
date: '2021-02-26T00:00:00.000Z'
section: paper-shorts
postSlug: learning-transferable-visual-models-from-natural-language-supervision
legacyPath: >-
  /paper
  shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html
tags:
  - Vision-Language Models
  - Multimodal Pretraining
field: 'Vision-Language Models'
summary: '2021 – Learning Transferable Visual Models From Natural Language Supervision (CLIP)'
---

## 2021 – Learning Transferable Visual Models From Natural Language Supervision (CLIP)

**arXiv:** [2103.00020](https://arxiv.org/abs/2103.00020)

**GitHub:** [openai/CLIP](https://github.com/openai/CLIP) · [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)

**Project:** [OpenAI CLIP announcement](https://openai.com/index/clip/)

## Summary

> CLIP learns visual representations from natural-language captions instead of a fixed class list. It trains image and text encoders to identify the matching pair inside a batch, then turns class descriptions into a classifier at inference time. On the paper’s largest model, zero-shot ImageNet accuracy reaches 76.2% without using ImageNet’s 1.28M labeled images, and a 27-dataset study finds the representation beats a supervised ResNet-50 linear probe on 16 datasets. The same open vocabulary also imports web-data noise, prompt sensitivity, and social bias.

## Core Insights

### Contrastive pairs turn language into a dynamic classifier

![CLIP contrastive pretraining and zero-shot prediction](/assets/images/clip-paper-figure-1-contrastive-pretraining.png)
*Fig 1: CLIP learns one image and one text embedding for every pair in a batch, scores all image-text combinations, and later uses class descriptions as the rows of a zero-shot classifier. | source: [CLIP, Figure 1](https://arxiv.org/abs/2103.00020)*

For a batch of $N$ image-caption pairs, CLIP produces $N$ image embeddings and $N$ text embeddings, L2-normalizes them, and computes an $N\times N$ cosine-similarity matrix. The diagonal entries are the observed pairs; the other $N^2-N$ entries are in-batch negatives. A learned temperature scales the logits, and the loss averages cross-entropy in both directions: image-to-text and text-to-image. The encoders interact only through this similarity matrix, rather than through cross-attention over every image region and word.

At test time, the labels do not need to be the closed-set symbols seen during training. The text encoder embeds prompts such as “a photo of a {label},” and the image encoder chooses the class whose text embedding has the highest similarity. The classifier is therefore synthesized from language for each task. A phrase can describe a breed, an action, or a place without adding a new output head, provided the web data and the prompt give the visual encoder enough evidence to learn it.

### The corpus makes the interface possible

CLIP’s dataset, WebImageText (WIT), contains about 400M image-text pairs gathered from publicly available internet sources. To widen the concept vocabulary, the authors search for pairs containing one of roughly 500,000 queries and cap each query at 20,000 pairs. This is a scalable collection rule rather than a clean annotation process: captions can be incomplete, duplicated, culturally narrow, or unrelated to the visible image. The text remains valuable because it names concepts that a 1,000-class label set cannot cover, but the model inherits the distribution that produced those captions.

The authors first tried caption generation and a bag-of-words prediction baseline. Figure 2 shows the practical reason for the final objective: the Transformer language-model baseline learns ImageNet classes more slowly than the simpler bag-of-words predictor, and replacing prediction with batch contrastive learning improves efficiency again. CLIP therefore optimizes an easier pair-identification target instead of reproducing every word in the accompanying description. This choice sacrifices dense language generation for a representation that can be reused through text prompts.

![CLIP’s zero-shot transfer efficiency as the image corpus grows](/assets/images/learning-transferable-visual-models-from-natural-language-supervision-source-figure-2-white.png)
*Fig 2: The source efficiency plot shows zero-shot ImageNet accuracy rising faster for CLIP’s contrastive objective than for the captioning baselines as more image-text pairs are processed. | source: [CLIP, Figure 2](https://arxiv.org/abs/2103.00020)*

The training sweep includes ResNet-50, ResNet-101, three compute-scaled ResNets, and ViT-B/32, ViT-B/16, and ViT-L/14. The text encoder is a 12-layer, 512-wide Transformer with a 49,152-token lower-cased BPE vocabulary and a 76-token context limit. All models train from scratch for 32 epochs with a batch size of 32,768, Adam with decoupled weight decay, a cosine learning-rate schedule, mixed precision, and a learned temperature whose logit scale is clipped for stability. The best reported default is ViT-L/14 with one extra epoch at 336-pixel resolution; the largest ResNet takes 18 days on 592 V100 GPUs, and the largest Vision Transformer takes 12 days on 256 V100 GPUs.

### Zero-shot breadth is the result, not just ImageNet accuracy

![CLIP zero-shot transfer compared with a supervised ResNet-50 linear probe](/assets/images/learning-transferable-visual-models-from-natural-language-supervision-source-figure-5-white.png)
*Fig 3: Each bar compares zero-shot CLIP with a linear classifier fitted to ResNet-50 features across the paper’s 27-dataset suite; CLIP is ahead on 16 datasets, including ImageNet. | source: [CLIP, Figure 5](https://arxiv.org/abs/2103.00020)*

| Evidence | Reported result | What it establishes |
| --- | ---: | --- |
| ImageNet zero-shot | 76.2% top-1 | matches the original supervised ResNet-50 without ImageNet labels |
| Broad linear-probe comparison | ahead on 16 of 27 datasets | transfer is not confined to ordinary object recognition |
| Prompt engineering | nearly +5 points on average | the natural-language interface is part of the score |
| Natural distribution shifts | robustness gap reduced by up to 75% | zero-shot evaluation can improve effective robustness |
| ImageNet adaptation | 85.4%, +9.2 points | in-distribution tuning slightly reduces average robustness |

The 27 datasets span OCR and text recognition, fine-grained objects, actions, geo-localization, scene recognition, and visual reasoning. CLIP is strongest where ImageNet’s single-label vocabulary is a poor description of the task: the paper highlights OCR, traffic signs, cars, and video actions. It still loses on some low-resolution datasets and on tasks whose labels or visual statistics are far from WIT. The comparison is also a representation study: the supervised baseline is a linear probe on ResNet-50 features, not a separately retrained model for every task.

Prompt wording changes the classifier. Adding context such as “a photo of” and ensembling several templates improves average zero-shot accuracy by almost five points over contextless class names. This is not a nuisance outside the model; it is the mechanism by which a developer tells the text encoder what kind of visual evidence matters. A reported class score therefore bundles representation quality, label semantics, and prompt design.

The robustness analysis compares zero-shot CLIP with ImageNet-trained models on seven natural distribution shifts. The best zero-shot model shrinks the gap between ImageNet and shifted data by up to 75%, but adapting CLIP to ImageNet raises ImageNet accuracy to 85.4% while slightly lowering average shifted-data robustness. The paper’s bias analysis also shows why “open vocabulary” needs a governance boundary: changing the candidate labels or the language used to describe people changes which harmful associations can be emitted. Benchmark breadth does not make arbitrary class design safe.

### Decision test and boundary

Use CLIP when an application needs a reusable image/text embedding interface and open-vocabulary transfer. Evaluate prompts, calibration, rare concepts, distribution shifts, and dataset overlap together; a single ImageNet number hides the main trade-off. The method is less attractive when dense grounding, counting, or reliable spatial reasoning is required, because its only cross-modal operation is a global dot product. WIT’s scale is also inseparable from its noise and bias. CLIP’s durable contribution is the language-defined classifier interface, not a guarantee that internet captions provide neutral supervision.

## High-Level Takeaways

- CLIP trains image and text encoders with symmetric in-batch contrastive loss.
- Natural-language class prompts synthesize a zero-shot classifier without task-specific output heads.
- Broad transfer and shift robustness emerge at web scale, but prompt wording and task coverage affect the score.
- Open vocabulary imports the web corpus’s noise and social bias; label design is part of deployment behavior.
