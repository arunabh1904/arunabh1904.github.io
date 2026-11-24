---
title: 'LoRA: Low-Rank Adaptation of Large Language Models'
date: '2021-06-17T09:00:00.000Z'
section: paper-shorts
postSlug: lora-low-rank-adaptation-of-large-language-models
legacyPath: /paper shorts/2021/06/17/lora-low-rank-adaptation-of-large-language-models.html
tags:
  - Fine-Tuning
  - Low-Rank Adaptation
field: "Alignment & Post-Training"
summary: "2021 – LoRA: Low-Rank Adaptation of Large Language Models"
---

## 2021 – LoRA: Low-Rank Adaptation of Large Language Models

**Paper:** [Hu et al., arXiv:2106.09685v2](https://arxiv.org/abs/2106.09685v2) · **Code:** [Microsoft LoRA](https://github.com/microsoft/LoRA)

## Summary

> LoRA freezes a pretrained weight matrix and learns a low-rank update. The small update cuts optimizer memory and task-specific checkpoint size while keeping a dense model available for inference. In the paper's Transformer experiments, which projections receive an update matters as much as the rank assigned to each.

## Core Insights

A pretrained linear layer maps an input $x$ through a weight matrix $W_0$. LoRA gives the same input a second path. Matrix $A$ projects down to rank $r$, then matrix $B$ projects back to the output width. The layer adds the result to the base output:

$$
h = W_0 x + (\alpha/r) B A x.
$$

For $W_0$ of shape $d \times k$, the two trainable factors contain $r(d+k)$ parameters rather than $dk$. The authors initialize $A$ with random Gaussian values and $B$ with zeros, so the model starts at the original function. The constraint applies to the update, not the rank of the merged weight.

![A frozen linear path and a narrow trainable path process the same input before their outputs are added.](/assets/images/lora-low-rank-update.svg)
*Fig 1: The narrow path constrains the update to rank at most $r$. Original diagram. | source: [Hu et al., Figure 1 and Section 4.1](https://arxiv.org/html/2106.09685v2#S1.F1)*

### Spend rank where it changes the result

The authors compare adaptation methods across RoBERTa, DeBERTa, GPT-2, and GPT-3. In GPT-3 175B, the 4.7M-parameter LoRA configuration gets 91.7% on MultiNLI-matched versus 89.5% for full tuning, and scores 45.9 versus 44.5 on SAMSum ROUGE-L. On WikiSQL, it gets 73.4% versus 73.8% for full tuning, a gap within the authors' reported fluctuation. These are favorable results for the tested configurations, not evidence that a smaller update always beats a full one.

Section 7 makes the allocation question concrete. With an approximately 18M-parameter budget in GPT-3, WikiSQL validation accuracy is 70.4% when the query projection alone receives rank eight and 73.7% when query and value each receive rank four. Spreading rank across all four attention projections at rank two also reaches 73.7%, and improves the matched MultiNLI score over the query/value allocation. The comparison holds parameter budget fixed; the reader can see when broadening coverage matters more than enlarging one update. It does not settle the best allocation for other models or tasks. The original experiment also leaves MLP adaptation outside its main scope.

### Plan the training and serving paths separately

Section 4.2 reports 1.2 TB versus 350 GB of aggregate training VRAM in the GPT-3 setup. At rank four on query and value, the task checkpoint is about 35 MB against 350 GB for the full weights. Those are two different savings: frozen parameters still have to be available to the model, and a small task checkpoint does not imply the base model fits on a small device.

For a deployment with one fixed adapter, fold the scaled product into $W_0$ before inference. The result uses the original linear operation. Serving different adapters in the same batch requires a different strategy, such as keeping the update path separate, and no longer gets the merged execution guarantee by construction.

## High-Level Takeaways

- Low rank constrains what changes, while the base weight remains available in full.
- Compare target modules at a fixed parameter budget before selecting a larger rank.
- Measure training memory and deployment latency separately. Merging removes the extra branch for a fixed adapter, but changes how adapters can be switched or batched.
