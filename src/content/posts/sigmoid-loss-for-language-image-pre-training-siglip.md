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

## Summary

> SigLIP changes where image-text contrastive learning normalizes. CLIP treats the other items in a batch as classes inside a global softmax; SigLIP treats each image-text pair as an independent binary example. That removes the global denominator, makes the distributed loss chunkable, and improves small-batch training. The result is a simpler systems boundary, not a promise that ever-larger batches or arbitrary negatives keep helping.

## Core Insights

### The loss changes a systems dependency

For a batch of matched image-text pairs, let $x_i$ and $y_j$ be normalized image and text embeddings. SigLIP assigns $z_{ij}=+1$ to the matched diagonal pair and $z_{ij}=-1$ to every other pair, then optimizes

$$
\mathcal{L}
= -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{n}
\log \sigma\left(z_{ij}\left(t\,x_i^\top y_j+b\right)\right).
$$

The learnable temperature is $t=\exp(t')$. The additional bias $b$ is initialized to $-10$, while $t'$ starts at $\log 10$. That bias is not cosmetic: a batch contains $n$ positives but $n^2-n$ negatives, so a zero bias makes the initial loss badly imbalanced. The initialization begins near the prior odds of one match among many non-matches and lets optimization move the operating point.

CLIP’s symmetric softmax loss must normalize each image against every text and each text against every image. The denominator couples all entries in the similarity matrix: a device needs a global view before it can finish its local loss. SigLIP instead sums independent pair losses. It keeps the same positive diagonal and negative off-diagonal intuition, but the objective no longer asks the optimizer to select one class from the entire batch.

![Cross-device SigLIP loss computation accumulates independent pair losses without materializing one global softmax](/assets/images/sigmoid-loss-for-language-image-pre-training-siglip-paper-figure.png)
*Fig 1: Efficient loss implementation demonstrated via a mock setup with 3 devices and a global batch size of 12. There are no all-gathers, and at any point in time only the bright yellow square (size 4 × 4) is materialized in memory. | source: [SigLIP, Figure 1](https://arxiv.org/abs/2303.15343)*

Figure 1 is best read as a memory diagram. Each device begins with four image and four text embeddings and evaluates its local 4 × 4 block. It then swaps text chunks, computes the next block, and accumulates the pair losses. After every text chunk has visited every image chunk, a cross-device sum completes the objective. The largest live similarity block stays 4 × 4 in this toy setup, rather than becoming one global 12 × 12 matrix. The loss is still pairwise across the global batch; the implementation changes how that work is scheduled.

### Batch size is useful until it is not

![Figure 2 from SigLIP showing batch-size effects for SigLiT, SigLIP, and mSigLIP](/assets/images/sigmoid-loss-for-language-image-pre-training-siglip-source-figure-2.png)
*Fig 2: Batch-size ablations for SigLiT (18B examples), SigLIP (9B), and mSigLIP (30B). Sigmoid helps at smaller batches, both objectives plateau near 32k, and multilingual scaling beyond 32k hurts XM3600 retrieval. | source: [SigLIP, Figure 2](https://arxiv.org/abs/2303.15343)*

Figure 2 separates three regimes. In locked-image tuning (SigLiT), sigmoid is clearly ahead below roughly 16k examples per batch; as the batch grows, the softmax gap closes. In from-scratch SigLIP, both objectives flatten around 32k, and a 307k batch hurts both. In multilingual mSigLIP, the same 32k plateau appears even though the data cover more than 100 languages. A larger batch can put more languages or harder negatives together, but the paper does not observe a reliable gain beyond that point. The practical conclusion is a resource target: 32k is a strong default, not a lower bound on useful scale.

The paper’s Table 1 makes the tradeoff concrete. Locked-image tuning reaches 79.8% ImageNet zero-shot accuracy with a ViT-B/8 vision checkpoint on four TPUv4 chips in one day, and 84.5% with ViT-g/14 in two days. From-scratch SigLIP B/16 reaches 71.0% at batch 16k on 16 chips in three days, 72.1% at batch 32k in two days, and 73.4% at batch 32k in five days. These are different initialization and training-duration regimes, so the table supports efficiency claims rather than a clean architecture ranking.

### The recipe matters as much as the objective

SigLIP’s systems advantage becomes useful when the rest of training does not destroy the visual representation. With a public pretrained ViT-AugReg-B/16, the authors use a 0.1 learning-rate multiplier for the image tower. Default weight decay on those pretrained weights degrades ImageNet ten-shot transfer; disabling it while retaining decay on randomly initialized text weights stabilizes the representation and reaches 71.0% ImageNet zero-shot accuracy. Large batches also make optimization more fragile: gradient-norm spikes produce large parameter updates, and changing Adam or AdaFactor’s $\beta_2$ from 0.999 to 0.95 is enough to stabilize the runs.

The pairwise view also lets the authors vary the negative-to-positive ratio directly. At batch 16k, one positive is surrounded by about 16k negatives, yet randomly removing negatives hurts, keeping only easy negatives fails, and keeping the hardest negatives almost preserves quality. The matched-pairs experiment suggests that imbalance itself is not the main problem; the information content of hard negatives is more useful than simply counting pair slots. The learned bias shifts as the negative ratio changes, which is exactly what the pairwise formulation makes measurable.

Noise experiments point in the same direction. The authors corrupt images, texts, batch alignment, or combinations of these with increasing probability. Sigmoid-trained models retain an advantage over softmax baselines as corruption rises, but this is an empirical robustness result for their M/16 setup and 3.6 billion seen examples, not a guarantee for every web corpus. Their scaled models also reach 76.2% ImageNet zero-shot and 64.4/47.2 COCO image-to-text/text-to-image recall@1 at B/16, while the 400M-parameter So model reaches 83.2% ImageNet and 70.2/52.0 retrieval recall@1 at 729 patches. mSigLIP reaches 34.9% average XM3600 text-to-image retrieval with a Base model, above the 28.5% prior LiT result cited by the paper.

| Decision signal | Reported evidence | Interpretation |
| --- | ---: | --- |
| Smaller batches | Sigmoid is strongest below about 16k | Removes a softmax penalty when global negatives are scarce |
| Practical plateau | Both losses saturate around 32k | More negatives are not automatically more information |
| Multilingual scaling | mSigLIP reaches 34.9 XM3600 text-to-image retrieval | 32k is sufficient in the reported 36-language study |
| Large-batch stability | $\beta_2=0.95$ stabilizes gradient spikes | Optimizer settings become part of the scaling recipe |
| Noisy pairs | Sigmoid retains an advantage under injected corruption | Suggestive robustness, not a corpus-independent law |

### A better systems boundary, not an unlimited-negative claim

Use SigLIP when the training bottleneck is the global softmax and the available system cannot afford a giant all-gathered similarity matrix. Compare it with a softmax baseline at matched encoders, data, examples seen, optimizer, and batch composition; otherwise the loss and systems changes are confounded. The paper’s private WebLI corpus also limits reproducibility. SigLIP changes normalization and communication, while [dino.txt](/paper%20shorts/2024/12/20/dinov2-meets-text-dino-txt.html) changes the visual initialization and freezing decision. At 32k, data quality, false negatives, optimizer stability, and image resolution may dominate the choice of loss.

## High-Level Takeaways

- SigLIP replaces batch-softmax normalization with independent pairwise sigmoid losses and a learned prior bias.
- Chunked cross-device negatives avoid all-gathering one global similarity matrix while preserving global pair interactions.
- The clearest gains appear below roughly 16k batch size; both objectives saturate near 32k, including multilingual training.
- Weight decay, $\beta_2$, hard-negative composition, and noisy web pairs remain part of the result, so the loss is not an isolated magic switch.
