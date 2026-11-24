---
title: 'Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning'
date: '2023-03-28T09:00:00.000Z'
section: paper-shorts
postSlug: scaling-down-to-scale-up-a-guide-to-parameter-efficient-fine-tuning
legacyPath: /paper shorts/2023/03/28/scaling-down-to-scale-up-a-guide-to-parameter-efficient-fine-tuning.html
tags:
  - Fine-Tuning
  - Parameter Efficiency
field: "Alignment & Post-Training"
summary: "2023 – Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning"
---

## 2023 – Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning

**Paper:** [Lialin et al., arXiv:2303.15647v2](https://arxiv.org/abs/2303.15647v2) · **Code:** [PEFT comparison experiments](https://github.com/guitaricet/peft_comparison)

## Summary

> This guide classifies parameter-efficient fine-tuning by what the method changes, then compares methods at T5 scales from about 0.7B to 11B parameters. Its short-run experiments make a useful distinction: a method can save memory while slowing training or inference. Query/value LoRA is a strong baseline in this setting, and tuning layer normalization alone is more competitive than its size suggests.

## Core Insights

The taxonomy starts with where trainable parameters enter the computation. Addition-based methods add elements such as adapter blocks or continuous prompts. Selection-based methods update a subset of existing parameters. Reparameterization-based methods represent a weight change through a smaller trainable object. Hybrid methods combine those choices. [The LoRA note](/paper%20shorts/2021/06/17/lora-low-rank-adaptation-of-large-language-models.html) explains one reparameterization in detail; this guide asks how it compares with the alternatives under a common training budget.

![Taxonomy showing addition-based, selection-based, and reparameterization-based PEFT methods, with adapter and soft-prompt subgroups.](/assets/images/peft-guide-source-figure-2.webp)
*Fig 1: The taxonomy groups methods by what they change. That tells you where to look for extra sequence work, new modules, or an update that can be folded into the base weights. | source: [Lialin et al., Figure 2](https://arxiv.org/html/2303.15647v2#S3.F2)*

Figure license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Converted to WebP.

[Open the figure at full size](/assets/images/peft-guide-source-figure-2.webp)

### Efficiency is more than checkpoint size

Section 11 compares T5-Large, T5-3B, and T5-11B on RTE, COPA, BoolQ, and CNN/DailyMail. The language-understanding runs use a three-value learning-rate sweep, additional seeds for the selected setting, and three epochs or at least 100 updates. Summarization runs for one epoch at one learning rate. Input length is capped at 512 tokens. This protocol asks which methods work without a long tuning campaign; it gives faster-converging methods an advantage.

Table 5 reports that on T5-3B, query/value LoRA trains 3.93M parameters, peaks at 9.6 GB, and scores 75.49 on the paper's aggregate, versus 32.9 GB and 74.83 for full tuning. LoRA on all projections reaches 75.22 while consuming 10.6 GB and delivering 70.2% of full-tuning training throughput, versus 103.0% for query/value LoRA. In the T5-11B row, layer-normalization tuning reaches 73.77 with about 0.12M parameters, compared with 73.25 for full tuning and 76.20 for query/value LoRA. Aggregate scores can mask task differences, and throughput comparisons at 11B include differences in GPU configurations. The full-tuning run uses two 80 GB A100s, whereas most PEFT runs use one 40 GB A100.

### Start with the baseline that matches the constraint

Several hybrid and soft-prompt methods in Section 11 need more care than small parameter counts imply. The authors observe weak prompt-tuning results and slow convergence for both prompt and prefix tuning under the experiment's short schedule. The summarization runs are still improving when training ends. These results describe what happened under the stated budget, not a ceiling on those methods with longer training or a wider sweep.

A practical comparison holds the model, sequence lengths, and batch settings fixed, then records validation quality, peak memory, and tokens per second. Use a simple normalization-only baseline alongside query/value LoRA, and time the actual serving path: merged LoRA, dynamically selected adapters, and sequence-extension methods impose different costs.

## High-Level Takeaways

- The taxonomy locates the change; measurements reveal whether it saves time, memory, or only task-specific storage.
- Query/value LoRA is a strong short-budget baseline in the tested T5 regime.
- Layer-normalization tuning deserves a cheap baseline run before adopting a more complex adapter.
- Recheck method rankings when training budget, hardware, backbone, or serving mode changes.
