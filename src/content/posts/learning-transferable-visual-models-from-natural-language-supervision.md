---
title: Learning Transferable Visual Models From Natural Language Supervision
date: '2021-02-26T00:00:00.000Z'
section: paper-shorts
postSlug: learning-transferable-visual-models-from-natural-language-supervision
legacyPath: >-
  /paper
  shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html
tags:
  - Other
field: 'Vision-Language Models'
summary: "2021 – Learning Transferable Visual Models From Natural Language Supervision"
---

## 2021 – Learning Transferable Visual Models From Natural Language Supervision (CLIP)

**arXiv:** [2103.00020](https://arxiv.org/abs/2103.00020)

**GitHub:** [openai/CLIP](https://github.com/openai/CLIP) · [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)

**Project:** [OpenAI CLIP announcement](https://openai.com/index/clip/)

## Summary

> CLIP learns visual representations from natural-language captions instead of a fixed class list. It trains image and text encoders to identify the matching pair inside a batch, then turns class descriptions into a classifier at inference time. On the paper’s largest model, zero-shot ImageNet accuracy reaches 76.2% without training on ImageNet’s labeled training split, and a 27-dataset study finds its zero-shot classifier beats a linear classifier fitted to supervised ResNet-50 features on 16 datasets. The same open vocabulary also imports web-data noise, prompt sensitivity, and social bias.

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

The training sweep includes ResNet-50, ResNet-101, three compute-scaled ResNets, and ViT-B/32, ViT-B/16, and ViT-L/14. The base text encoder is a 12-layer, 512-wide Transformer with a 49,152-token lower-cased BPE vocabulary and a 76-token context limit. The ResNet scaling sweep also widens the text encoder proportionally while holding its depth fixed. All models train from scratch for 32 epochs with a batch size of 32,768, Adam with decoupled weight decay, a cosine learning-rate schedule, mixed precision, and a learned temperature whose logit scale is clipped for stability. The best reported default is ViT-L/14 with one extra epoch at 336-pixel resolution; the largest ResNet takes 18 days on 592 V100 GPUs, and the largest Vision Transformer takes 12 days on 256 V100 GPUs.

### Zero-shot breadth is the result, not just ImageNet accuracy

![CLIP zero-shot transfer compared with a supervised ResNet-50 linear probe](/assets/images/learning-transferable-visual-models-from-natural-language-supervision-source-figure-5-white.png)
*Fig 3: Each bar compares zero-shot CLIP with a linear classifier fitted to ResNet-50 features across the paper’s 27-dataset suite; CLIP is ahead on 16 datasets, including ImageNet. | source: [CLIP, Figure 5](https://arxiv.org/abs/2103.00020)*

| Evidence | Reported result | What it establishes |
| --- | ---: | --- |
| ImageNet zero-shot | 76.2% top-1 | matches the original supervised ResNet-50 without ImageNet labels |
| Broad linear-probe comparison | ahead on 16 of 27 datasets | transfer is not confined to ordinary object recognition |
| Prompt engineering and ensembling | nearly +5 points averaged over 36 datasets in Figure 4 | a separate evaluation from the 27-dataset comparison |
| Natural distribution shifts | effective-robustness gap reduced by up to 75% | relative to the ImageNet-accuracy trend, not the raw accuracy drop |
| ImageNet linear probe on frozen CLIP features | 85.4%, +9.2 points | average shifted-data accuracy slightly decreases |

The 27 datasets span OCR and text recognition, fine-grained objects, actions, geo-localization, scene recognition, and visual reasoning. CLIP is strongest where ImageNet’s single-label vocabulary is a poor description of the task: the paper highlights OCR, cars, and video actions. In Figure 5, CLIP leads by 28.9 points on StanfordCars and 23.2 on Country211, yet trails by 18.4 on German traffic signs, 18.2 on CLEVR counting, and 37.1 on EuroSAT. Those negative bars expose tasks for which a broad web vocabulary is insufficient. The comparison is also a representation study: the ResNet-50 backbone stays fixed, while a separate linear classifier is fitted for each dataset.

Prompt wording changes the classifier. On ImageNet, adding context such as “a photo of” improves accuracy by 1.3 points and ensembling 80 templates adds another 3.5. Figure 4 separately reports almost five points on average across 36 datasets. Both compare with contextless class names. This is not a nuisance outside the model; it is the mechanism by which a developer tells the text encoder what kind of visual evidence matters. A reported class score therefore bundles representation quality, label semantics, and prompt design.

The robustness analysis compares zero-shot CLIP with ImageNet-trained models on seven natural distribution shifts. The best zero-shot model reduces the effective-robustness gap by up to 75% relative to the accuracy trend fitted to ImageNet-trained models. This measures performance beyond what the in-distribution score predicts, rather than the raw difference between ImageNet and shifted accuracy. Fitting a logistic-regression classifier to frozen CLIP features on ImageNet raises ImageNet accuracy to 85.4% while slightly lowering average accuracy under distribution shift. The paper’s bias analysis also shows that the class list changes the behavior: changing the candidate labels or the language used to describe people changes which harmful associations can be emitted. The supplied class set therefore affects the associations being measured.

### A language-defined classifier still reflects its training distribution

Independent encoders make the image and text embeddings reusable, while the global dot product gives no explicit box, count, or spatial relation. Strong transfer on many recognition tasks can therefore coexist with weak counting and unusual-object performance. The WIT overlap audit also matters: the paper detects median evaluation overlap of 2.2% and mean overlap of 3.2% across the audited datasets. Zero-shot means no supervised training on the benchmark's labeled split, not proof that web pretraining contains no overlapping images. Prompt sensitivity, uneven concept coverage, and the paper's observed social biases all follow the classifier into a new label set.

## High-Level Takeaways

- CLIP trains image and text encoders with symmetric in-batch contrastive loss.
- Natural-language class prompts synthesize a zero-shot classifier without task-specific output heads.
- Broad transfer and shift robustness emerge at web scale, but prompt wording and task coverage affect the score.
- Open vocabulary imports the web corpus’s noise and social bias; label design is part of deployment behavior.
