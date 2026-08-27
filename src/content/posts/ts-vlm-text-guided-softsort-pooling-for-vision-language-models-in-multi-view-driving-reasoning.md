---
title: 'TS-VLM: Text-Guided SoftSort Pooling for Vision-Language Models in Multi-View Driving Reasoning'
date: '2025-05-19T03:37:15.000Z'
section: paper-shorts
postSlug: ts-vlm-text-guided-softsort-pooling-for-vision-language-models-in-multi-view-driving-reasoning
legacyPath: /paper shorts/2025/05/19/ts-vlm-text-guided-softsort-pooling-for-vision-language-models-in-multi-view-driving-reasoning.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2025 – TS-VLM: Text-Guided SoftSort Pooling for Vision-Language Models in Multi-View Driving Reasoning"
---
## 2025 – TS-VLM

**arXiv:** [2505.12670](https://arxiv.org/abs/2505.12670)

## Summary

> TS-VLM replaces costly cross-view attention with Text-Guided SoftSort Pooling. A question ranks multi-view visual features by their semantic relevance, then pools them into the language path. On DriveLM, the paper reports BLEU-4 56.82, METEOR 41.91, ROUGE-L 74.64, and CIDEr 3.39; its smallest model has 20.1 million parameters and is reported to reduce compute by up to 90%. Those are reasoning metrics, not evidence of a vehicle control policy.

## Core Insights

The design asks a useful question before fusion: which camera views should matter for this query? Instead of paying attention cost across every token and view, TGSSP uses text semantics to order and aggregate features. The output is a query-adaptive view summary, so the model can favor a rear or side camera when the language task requires it without learning a full dense attention map.

![TS-VLM: Text-Guided SoftSort Pooling for Vision-Language Models in Multi-View Driving Reasoning source figure: The overall architecture of TS-VLM.](/assets/images/ts-vlm-text-guided-softsort-pooling-for-vision-language-models-in-multi-view-driving-reasoning-paper-figure.webp)
*Fig 1: TS-VLM uses the question text to softly sort and pool multi-view image tokens, then feeds the selected visual representation into a text-to-text answer model. | source: [TS-VLM: Text-Guided SoftSort Pooling for Vision-Language Models in Multi-View Driving Reasoning](https://arxiv.org/abs/2505.12670)*

![Figure 1 from TS-VLM: Text-Guided SoftSort Pooling for Vision-Language Models in Multi-View Driving Reasoning](/assets/images/ts-vlm-text-guided-softsort-pooling-for-vision-language-models-in-multi-view-driving-reasoning-source-figure-1.webp)
*Fig 2: Model performance vs. model size on the DriveLM benchmark across four metrics: BLEU-4, METEOR, ROUGE-L, and CIDEr. Each circle represents a model, where the x-axis indicates model size (in millions of parameters, log scale), and the y-axis shows the performance score (↑ = better). | source: [TS-VLM: Text-Guided SoftSort Pooling for Vision-Language Models in Multi-View Driving Reasoning](https://arxiv.org/abs/2505.12670)*


The saving depends on the comparison contract. The abstract does not report the number of views, visual tokens per view, hardware, latency distribution, or a matched-parameter sparse-attention baseline. It also does not test whether pooling preserves low-probability but safety-critical views when the question does not explicitly name them. A robust driving interface should answer that latter question before treating average VQA efficiency as deployment readiness.

## High-Level Takeaways

- TS-VLM changes multi-view fusion from all-to-all attention to query-conditioned ranking and pooling.
- Its reported DriveLM scores and small-model compute result support efficient driving reasoning, not direct closed-loop action quality.
- The relevant falsification is a matched-latency test with unexpected hazards in a visually secondary view; the approach fails if semantic pooling systematically discards evidence before the question reveals its importance.
