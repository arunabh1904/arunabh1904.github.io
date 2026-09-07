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

## Summary

> MM1 is a controlled study of where multimodal pretraining budget actually pays off. Around a fixed decoder-only language model, it varies the image encoder, resolution, visual-token count, connector, and data mixture, then scales the selected recipe to dense and MoE models up to 30B. Its practical lesson is a prioritization rule: preserve useful visual evidence and choose the data mixture before polishing the connector.

## Core Insights

### Controlled ablations locate the useful visual capacity

MM1 starts with a deliberately small base configuration: a 1.2B decoder-only language model, ViT-L/14 at 336 × 336, a C-Abstractor producing 144 image tokens, and a 45% captioned, 45% interleaved, 10% text-only data mixture. Each ablation changes one architecture or data choice at a time and evaluates zero-, four-, and eight-shot captioning and VQA.

![MM1 model and data ablation axes: image encoder, connector, resolution, visual tokens, and data mixture](/assets/images/mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training-paper-figure.png)
*Fig 1: MM1's ablation plan varies visual encoders, connector types, resolution, image-token count, objectives, and data mixtures around a fixed decoder-only language model. | source: [MM1, Figure 3](https://arxiv.org/abs/2403.09611)*

The strongest architecture signal is resolution. Moving from 224 to 336 pixels produces roughly a 3% improvement across metrics, while doubling the ViT from L to H gives a usually smaller increase of under 1%. The connector study points in the same direction: increasing resolution or visual-token count helps, while average pooling, attention pooling, and C-Abstractor do not separate cleanly at matched settings. More tokens preserve local evidence, but they make every multimodal context longer and more expensive.

Data changes what those tokens are used for. Captioned pairs are short and highly image-relevant, so they lift zero-shot performance. Interleaved documents contain longer text, multiple images, and relationships resembling few-shot context; MM1 reports that at least about 50% interleaved data is important for maintaining strong four- and eight-shot results. Text-only data helps preserve language performance. In the paper's ablations, a 5:5:1 caption/interleaved/text mix offers a useful balance, while the small VeCap synthetic-caption component improves few-shot performance by 2.4 and 4 percentage points in the reported comparison.

The final scaled recipe uses ViT-H at 378 × 378 with 144 visual tokens, a C-Abstractor, and a 45% interleaved, 45% paired-caption, 10% text-only mixture. MM1 trains dense 3B, 7B, and 30B models for about 400B tokens and builds 3B- and 7B-MoE variants. The reported 30B pretraining model reaches 71.9 VQAv2 and 59.3 OKVQA at 16-shot in Table 3, with comparisons qualified by prompt and data differences. The scaling result supports the recipe; it does not turn the small-scale component ranking into a universal law.

| Lever | What MM1 observed | Practical interpretation |
| --- | --- | --- |
| Resolution | 224 → 336 gives an approximately 3% boost in the encoder ablation | Preserve visual evidence before adding decoder complexity. |
| Visual-token count | More tokens improve zero- and few-shot scores | Budget tokens against context and latency. |
| Connector type | No clear winner among tested connectors | Do not over-invest before fixing representation and data. |
| Data mixture | Caption, interleaved, and text-only data serve different regimes | Choose the mixture for the capability being measured. |

## High-Level Takeaways

- MM1 turns multimodal pretraining into experiment allocation: test resolution, encoder quality, token count, and data mixture first.
- Interleaved data is a mechanism for multi-image and long-context behavior, while caption data aligns local visual evidence and text.
- Its connector conclusion is local to the tested encoders, token budgets, and tasks; stronger encoders or latency-constrained settings may change the ranking.
- The reported 30B numbers are few-shot pretraining results with protocol-specific prompts, so they should not be compared to instruction-tuned scores as if they were the same task.
- The paper's surprise is that the apparently unglamorous visual bandwidth and data decisions dominate the connector choice in the tested regime.
