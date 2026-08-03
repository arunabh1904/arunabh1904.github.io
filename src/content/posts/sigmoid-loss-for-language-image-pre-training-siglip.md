---
title: 'Sigmoid Loss for Language-Image Pre-Training (SigLIP)'
date: '2023-03-27T00:00:00.000Z'
section: paper-shorts
postSlug: sigmoid-loss-for-language-image-pre-training-siglip
legacyPath: /paper shorts/2023/10/01/sigmoid-loss-for-language-image-pre-training-siglip.html
tags:
  - Vision-Language Models
  - Contrastive Learning
field: 'Vision-Language Models'
topics:
  - multimodal
  - learning
summary: '2023 – Sigmoid Loss for Language-Image Pre-Training (SigLIP)'
---

## 2023 – Sigmoid Loss for Language-Image Pre-Training (SigLIP)

**arXiv:** [2303.15343](https://arxiv.org/abs/2303.15343)

**Code:** [google-research/big_vision](https://github.com/google-research/big_vision/tree/main/big_vision/models/proj/siglip)

**Conference:** ICCV 2023 (oral)

SigLIP changes the normalization boundary in image-text contrastive learning. CLIP treats the other items in a batch as classes inside a global softmax. SigLIP treats every image-text combination as an independent positive or negative binary example. The change removes the global denominator, supports chunked cross-device negatives, and reduces the need to make the batch enormous merely to define the loss.

## Paper Insights

![Cross-device SigLIP loss computation accumulates independent pair losses without materializing one global softmax](/assets/images/sigmoid-loss-for-language-image-pre-training-siglip-paper-figure.png)
_Each device keeps local image embeddings and receives chunks of text embeddings. Because pair losses are independent, the system can accumulate negatives without assembling one global similarity matrix. Source: [SigLIP](https://arxiv.org/abs/2303.15343)._

For a batch of $n$ matched pairs, SigLIP labels diagonal image-text pairs positive and all off-diagonal pairs negative:

$$
\mathcal{L}
= -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{n}
\log \sigma\left(z_{ij}\left(t\,x_i^\top y_j+b\right)\right),
$$

where $z_{ij}=1$ for a match and $-1$ otherwise. The learned bias $b$ matters because each image has one positive and $n-1$ negatives; without it, the initial class imbalance can dominate optimization. The temperature scale $t$ controls similarity sharpness.

The strongest result is a scaling correction, not “sigmoid always wins.” Benefits are clearest below roughly 16,000 examples per batch. Both sigmoid and softmax objectives saturate around 32,000, and even a one-million-example batch provides little value. The loss makes smaller distributed systems competitive; it does not make unlimited negatives useful.

| Configuration reported in the paper | Hardware and time | ImageNet zero-shot |
| --- | --- | ---: |
| Locked-image tuning, ViT-B/8 vision encoder | 4 TPUv4 chips, 1 day, batch 32k | 79.8% |
| Locked-image tuning, ViT-g/14 vision encoder | 4 TPUv4 chips, 2 days, batch 20k | 84.5% |
| Unlocked SigLIP B/16 | 16 TPUv4 chips, 3 days, batch 16k | 71.0% |
| B/16 trained from scratch | 32 TPUv4 chips, 5 days, batch 32k | 73.4% |

Locked-image tuning freezes the vision encoder and trains the text side, so those rows should not be read as an end-to-end architecture comparison. They demonstrate that the objective remains effective when alignment is added to an existing visual model.

## Decision Lens

SigLIP informs whether cross-device global softmax normalization is worth its systems cost. Its atomic example is one image-text pair with an independent binary target. That makes it appealing when accelerator memory, all-gather traffic, or cluster size constrains training.

The source results use WebLI, a private web-scale dataset, so public reproduction cannot fully separate the loss from data composition. The decisive systems comparison would hold data order, encoder, optimizer, total negative pairs, and wall-clock budget fixed across cluster topologies. At ten times scale, false negatives and noisy pairs are more likely to dominate than normalization.

[dino.txt](/paper%20shorts/2024/12/20/dinov2-meets-text-dino-txt.html) addresses a different decision: preserve a self-supervised dense vision backbone and attach language alignment afterward. SigLIP is primarily a loss and distributed-training choice; dino.txt is primarily an initialization and freezing choice.

**Context:** SigLIP replaces batch-softmax contrastive learning with independent sigmoid losses over image-text pairs.

**Limits:** The best data are private, most negatives remain uncurated, and the objective's advantage narrows at very large batches.

**Takeaway:** Image-text pretraining does not need one global softmax; pairwise sigmoid loss turns normalization into a local systems choice and reveals that useful batch scaling saturates early.
