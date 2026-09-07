---
title: 'MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training'
date: '2024-03-14T00:00:00.000Z'
section: paper-shorts
postSlug: mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training
legacyPath: /paper shorts/2024/03/14/mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training.html
tags: [Multimodal AI]
field: 'Omni-Model Architectures'
summary: "2024 – MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training"
---

## 2024 – MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training

**arXiv:** [2403.09611](https://arxiv.org/abs/2403.09611)  
**Conference:** Technical report

### Method and reported result

MM1 is a controlled ablation study of multimodal LLM pre-training. It varies image-encoder pretraining, resolution, visual-token count, connector architecture, and data mixture, then scales the selected recipe to dense and mixture-of-experts language models up to 30B parameters. Its central result is a prioritization rule: visual evidence and training data matter more consistently than connector ornamentation in the tested regime.

## Summary

> MM1 is less a proposal for one magical architecture than a guide to spending experiment budget. Preserve more useful visual evidence, choose the data mixture for the evaluation regime, and only then spend time polishing the connector.

## Core Insights

![MM1 model and data ablation axes: image encoder, connector, resolution, visual tokens, and data mixture](/assets/images/mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training-paper-figure.png)
*Fig 1: MM1's ablation plan varies visual encoders, connector types, resolution, image-token count, objectives, and data mixtures around a fixed decoder-only language model. | source: [MM1, Figure 3](https://arxiv.org/abs/2403.09611)*

MM1 starts with a small base configuration so one decision can be changed at a time: a 1.2B decoder-only language model, ViT-L/14 at 336 × 336, a C-Abstractor producing 144 image tokens, and a mixture of 45% captioned images, 45% interleaved image-text documents, and 10% text-only data. That setup makes the figure actionable. The question is not “which connector sounds modern?” but “which part of the visual-to-language path is throwing away the evidence the decoder needs?”

The strongest architecture signal is resolution. In the encoder ablation, moving from 224 to 336 pixels produced roughly a 3% improvement across metrics, while doubling the ViT from L to H produced a usually smaller gain of under 1%. The connector study points in the same direction: increasing resolution or the number of visual tokens helped, while average pooling, attention pooling, and C-Abstractor did not separate cleanly at matched settings. The intuitive tradeoff is detail versus sequence cost. More tokens preserve local evidence, but they also make every multimodal context longer and more expensive.

Data changes what those tokens are used for. Captioned pairs are short and highly image-relevant, so they lift zero-shot performance. Interleaved documents contain longer text, multiple images, and relationships that resemble few-shot context; MM1 reports that at least about 50% interleaved data was important for maintaining strong 4- and 8-shot results. Text-only data helps preserve language performance. In the paper's ablations, a 5:5:1 caption/interleaved/text mix offered a useful balance, and the small VeCap synthetic-caption component improved few-shot performance by 2.4 and 4 percentage points in the reported comparison.

The final scaled recipe uses a ViT-H at 378 × 378 with 144 visual tokens, a C-Abstractor, and a 45% interleaved, 45% paired-caption, 10% text-only mixture. MM1 then trains dense 3B, 7B, and 30B models for about 400B tokens and builds 3B- and 7B-MoE variants. Those results show the recipe scales, but the paper's strongest lesson remains experimental allocation: resolution, token budget, encoder quality, and data composition should be swept before assuming a fancier connector will transfer.

| Lever | What MM1 observed | Practical interpretation |
| --- | --- | --- |
| Resolution | 224 → 336 gave an approximately 3% boost in the encoder ablation | Preserve visual evidence before adding decoder complexity. |
| Visual-token count | More tokens improved zero- and few-shot scores | Budget tokens against context and latency. |
| Connector type | No clear winner among tested connectors | Do not over-invest before fixing representation and data. |
| Data mixture | Caption, interleaved, and text-only data serve different regimes | Choose the mixture for the capability being measured. |

## High-Level Takeaways

- MM1 turns multimodal pretraining into an experiment-allocation problem: test resolution, encoder quality, token count, and data mixture first.
- Its connector conclusion is deliberately local to the tested encoders, token budgets, and tasks; it is not a universal claim that connector design never matters.
- Interleaved data is a mechanism for learning multi-image and long-context behavior, while caption data aligns local visual evidence and text.
- The reported ranking can change when latency, tail memory, higher resolutions, or different downstream tasks become the objective.
- A good replication should keep end-to-end compute and average visual-token budget fixed while sweeping connector, resolution, and mixture separately.
