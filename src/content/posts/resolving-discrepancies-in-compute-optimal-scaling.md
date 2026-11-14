---
title: 'Resolving Discrepancies in Compute-Optimal Scaling of Language Models'
date: '2024-06-27T00:00:00.000Z'
section: paper-shorts
postSlug: resolving-discrepancies-in-compute-optimal-scaling
legacyPath: /paper shorts/2024/06/27/resolving-discrepancies-in-compute-optimal-scaling.html
tags: [Language Models, Scaling Laws, Training Systems]
field: 'Training Systems & Reliability'
summary: '2024 – Resolving Discrepancies in Compute-Optimal Scaling of Language Models'
---

## 2024 – Resolving Discrepancies in Compute-Optimal Scaling of Language Models

**arXiv:** [2406.19146](https://arxiv.org/abs/2406.19146) · **Conference:** NeurIPS 2024

## Summary

> Porian et al. reproduce Kaplan's parameter-heavy compute frontier and progressively correct the experimental protocol. Counting the output head, shortening warmup, and tuning the optimizer move the fitted exponent from about 0.84 to 0.50; matching learning-rate decay to token budget has only a small effect on the allocation. The disagreement between Kaplan and Chinchilla is therefore largely a training-accounting and hyperparameter-tuning problem, not evidence that one smooth law must be universally right.

## Core Insights

### The protocol changes the exponent

The authors train 16 Llama-derived decoder-only models from 5M to 901M parameters on OpenWebText2 and RefinedWeb, evaluate on held-out tokens, and estimate an IsoFLOP frontier. In the Kaplan-style setup, their exponent for optimal model size is about $a=0.835$ on RefinedWeb. They then make three controlled changes:

1. Count the final decoding layer in both model size and FLOPs: $a\approx0.706$.
2. Scale warmup tokens with model size instead of using a long fixed warmup: $a\approx0.602$.
3. Tune learning rate, batch size, and AdamW $\beta_2$ by model size: $a\approx0.497$.

![Source Figure 1 showing the stepwise move from Kaplan to Chinchilla-like scaling](/assets/images/resolving-discrepancies-compute-optimal-scaling-source-figure-1.png)
*Fig 1: Each panel adds one protocol correction. The fitted exponent and the predicted model at Chinchilla's compute budget move toward the Hoffmann et al. law. | source: [Resolving Discrepancies in Compute-Optimal Scaling of Language Models, Figure 1](https://arxiv.org/abs/2406.19146)*

### Warmup can masquerade as a data preference

Kaplan's fixed warmup is long enough that small models reach their apparent compute optimum during warmup. Those models are forced to process extra tokens before they can leave the warmup regime, inflating the inferred data allocation and making the fitted model-size exponent look large. Setting warmup tokens equal to $N$ leaves the optimal point well beyond warmup and removes much of this bias.

Counting the last layer matters for the same reason: its FLOPs are a large fraction of the total for small models, but the Kaplan-style accounting omits them. The paper also tests attention FLOPs in Appendix B and finds a similar qualitative sequence, with a mildly different final model-size prediction.

### Decay is not the missing explanation

After correcting warmup, the authors match cosine decay to each token budget, as Chinchilla hypothesized, but observe little change in the compute-optimal exponent. The final tuned experiment uses a constant learning rate schedule and still matches Chinchilla's roughly $a=0.5$ allocation. Cosine decay does improve the loss itself, including at larger scales; the finding is about the location of the compute frontier, not schedule quality.

The optimizer sweep is not free. Its cost is comparable to a scaling-law experiment, although the compute saved by using a constant schedule largely pays for it. The authors train on a smaller scale than Chinchilla, tune mainly small models, and measure pretraining loss rather than capabilities. Their own limitation section notes that the head contribution and hyperparameter sensitivity will shrink at larger scale. The result is a protocol lesson: when fitting a law at small scale, make the bookkeeping and tuning scale with the model.

## High-Level Takeaways

- Apparent Kaplan-versus-Chinchilla disagreement can arise from FLOP accounting and a mismatched warmup protocol.
- Tuning small-model learning rates, batch sizes, and $\beta_2$ is necessary for a reliable low-scale frontier.
- Learning-rate decay can improve loss without changing the compute-optimal allocation exponent.
- The measured target is pretraining loss; downstream capability and inference cost may choose a different frontier.
- [The Chinchilla note](/paper%20shorts/2022/03/29/training-compute-optimal-large-language-models.html) gives the allocation result, while this paper explains why a small-scale reproduction can miss it.
