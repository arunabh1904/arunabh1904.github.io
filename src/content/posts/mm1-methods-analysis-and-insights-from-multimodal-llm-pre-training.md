---
title: 'MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training'
date: '2024-03-14T09:00:00.000Z'
section: paper-shorts
postSlug: mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training
legacyPath: /paper shorts/2024/03/14/mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training.html
tags: [Multimodal AI]
field: Omni-Models
summary: '2024 – MM1: a controlled study of multimodal pre-training choices.'
---

## 2024 – MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training

**arXiv:** [2403.09611](https://arxiv.org/abs/2403.09611)  
**Conference:** Technical report

**Summary:** MM1 is a large ablation study of multimodal LLM pre-training. It varies the image encoder, resolution, visual-token count, vision-language connector, and data mixture before scaling the selected recipe to dense and MoE models up to 30B parameters.

## Paper Insights

The central result is a prioritization rule: image encoder quality, image resolution, visual-token count, and the mix of image-caption, interleaved image-text, and text-only data mattered much more than connector design. That makes MM1 a paper about experiment allocation. Before inventing a new projector, test the representation and data decisions that dominate the result.

| Decision | Signal from MM1 | Why it matters |
| --- | --- | --- |
| Visual representation | Encoder, resolution, and token count dominate | These are the first levers to sweep. |
| Data mixture | Mixed image-caption, interleaved, and text-only data is important | Capability is shaped by the training distribution. |
| Connector | Smaller effect in the reported ablations | Avoid spending the whole budget on connector variants. |

**Limits:** The conclusions come from MM1's model family and data pipeline; they are a strong experimental prior, not a universal ranking for every architecture.

**Takeaway:** In multimodal pre-training, spend early runs on the visual representation and mixture before polishing the connector.

