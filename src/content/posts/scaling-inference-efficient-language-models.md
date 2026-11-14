---
title: "Scaling Inference-Efficient Language Models"
date: '2025-01-30T00:00:00.000Z'
section: paper-shorts
postSlug: scaling-inference-efficient-language-models
legacyPath: /paper shorts/2025/01/30/scaling-inference-efficient-language-models.html
tags:
  - Inference
  - Scaling Laws
field: "Language Models"
summary: "2025 – Scaling Inference-Efficient Language Models"
---

## 2025 – Scaling Inference-Efficient Language Models

**Paper:** [Bian et al., arXiv:2501.18107v2](https://arxiv.org/abs/2501.18107v2) · **Code:** [open-lm-morph](https://github.com/Waterpine/open-lm-morph) · **Model:** [Morph-1B](https://huggingface.co/NaiveUser/morph-1b)

## Summary

> Parameter count alone is an inadequate proxy for inference latency. Among Morph variants trained on the same 30B tokens, widening the hidden state from 2,048 to 3,072 and reducing depth from 24 to 12 layers lowers measured generation latency from 3.61 to 1.96 seconds while preserving the reported average downstream score of 0.52. The result is tied to a specific A100 serving workload; the contribution is a method for selecting architectures under measured deployment constraints.

## Core Insights

[Chinchilla](/paper%20shorts/2022/03/29/training-compute-optimal-large-language-models.html) allocates training compute between parameters and tokens. This paper adds model shape and a latency constraint. Two models with similar parameter counts can execute differently: layers impose sequential dependencies, while operations within a wider layer can exploit parallel hardware. The useful width–depth trade-off therefore depends on the serving stack as well as the loss.

The authors extend the loss model with an aspect-ratio term:

$$
L(N,D,R)=\left(E+A N^{-\alpha}+B D^{-\beta}\right)\left(1+\epsilon R^\gamma\right),\qquad R=\frac{d_{\mathrm{model}}}{n_{\mathrm{layers}}}.
$$

Here $N$ counts parameters and $D$ training tokens. The fit constrains $\alpha$, $\beta$, and $\gamma$ to be equal. This is an empirical shape correction, not a latency model: latency is separately profiled on the target system. The optimization imposes limits on model size, available data, and measured latency.

### Predict a shortlist, then train and evaluate it

Why not release whichever architecture has the lowest predicted loss? Models with similar losses can differ on downstream tasks. The source workflow uses the fit to rank candidates, then reserves final selection for actual task evaluation.

![Source Figure 6 shows candidate architectures ranked by predicted loss and measured latency before training and downstream evaluation.](/assets/images/inference-efficient-morph-source-figure-6.png)
*Fig 1: The scaling fit narrows the architecture search. Hardware profiling supplies latency, and downstream evaluation after training chooses among shortlisted models; predicted loss does not replace either measurement. | source: [Scaling Inference-Efficient Language Models, Figure 6](https://arxiv.org/html/2501.18107v2#S2.F6)*

[Open the figure at full size](/assets/images/inference-efficient-morph-source-figure-6.png)

The fitting suite contains 63 small models spanning 80M–339M parameters and 1.6B–12.8B training tokens, with shape and token-to-parameter ratio varied. Larger models test extrapolation. Training uses uniformly sampled DCLM-Baseline subsets for one epoch, GPT-NeoX tokenization, a 2,048-token context, and AdamW. “Overtraining” here means more tokens per parameter than the training-compute optimum, rather than repeating a fixed corpus.

Including the shape term reduces the reported loss-prediction MSE from 0.0033 to 0.0006 in the main comparison. More consequentially for selection, the predicted ranking of tested 1B configurations improves from Spearman correlation −0.40 to 1.00. That small candidate set does not establish equally accurate ranking over arbitrary architectures.

### The controlled Morph comparison carries the architectural claim

All three Morph variants use the same 30B training tokens. Table 2 reports:

| Variant | Hidden width / layers | Average task score | Generation latency |
|---|---:|---:|---:|
| Morph-1B-v1 | 2,048 / 24 | 0.52 | 3.61 s |
| Morph-1B-v2 | 2,560 / 16 | 0.52 | 2.57 s |
| Morph-1B | 3,072 / 12 | 0.52 | 1.96 s |

Latency uses Hugging Face generation on one 40GB A100, batch size one, 128 input tokens, and 256 output tokens. The roughly 1.8× ratio compares those complete generations. It is not a claim about every batch size, context length, or GPU. Comparisons against separately released checkpoints also change data and training history; the matched Morph variants provide the cleaner evidence about shape.

The overtraining ablation is equally useful. Removing the 160-tokens-per-parameter fitting data worsens both the original and extended laws when predicting overtrained models. A more expressive formula cannot recover an unobserved regime merely because it fits the available points well. A six-point fit reduces fitting-run cost in one experiment, but includes a deliberately chosen overtrained point; six arbitrary runs are not the demonstrated recipe.

Appendices test A30 hardware, alternative sequence lengths, time to first token, and vLLM, supporting the importance of shape across several settings. Accuracy-preserving training is not demonstrated at 7B scale, and different attention designs or newer kernels can change the frontier. I would re-profile candidate shapes on the intended workload and retain a small downstream evaluation shortlist before committing a larger training budget.

## High-Level Takeaways

- Width and depth affect execution independently of parameter count; measured serving latency belongs in architecture selection.
- Loss prediction is useful for ranking a shortlist, while downstream evaluation remains necessary for the final choice.
- Same-data Morph variants isolate the shape trade-off more clearly than comparisons against independently trained public models.
- Fits intended to guide overtraining need observations in that regime, and hardware or attention changes require renewed profiling.
