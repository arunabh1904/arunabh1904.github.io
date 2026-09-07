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

### The question determines how camera views are pooled

The design asks a useful question before fusion: which camera views should matter for this query? Instead of paying attention cost across every token and view, TGSSP uses text semantics to order and aggregate features. The output is a query-adaptive view summary, so the model can favor a rear or side camera when the language task requires it without learning a full dense attention map.

In the module, each view feature and the projected question are compared by cosine similarity. SoftSort turns the resulting scores into a differentiable ordering and normalized weights, which pool the visual views before T5 generates the answer. The comparison table explains the design choice: hard top-one pooling loses secondary-view context, uniform pooling ignores relevance, and SinkhornSort spends roughly 180 times the FLOPs of SoftSort without improving the language scores. The architecture figure is therefore a budgeted information bottleneck, not a generic attention replacement; its failure mode is discarding a view before the question exposes why it matters.

![TS-VLM: Text-Guided SoftSort Pooling for Vision-Language Models in Multi-View Driving Reasoning source figure: The overall architecture of TS-VLM.](/assets/images/ts-vlm-text-guided-softsort-pooling-for-vision-language-models-in-multi-view-driving-reasoning-paper-figure.webp)
*Fig 1: TS-VLM uses the question text to softly sort and pool multi-view image tokens, then feeds the selected visual representation into a text-to-text answer model. | source: [TS-VLM, Figure 3](https://arxiv.org/abs/2505.12670)*

The performance-size plot compares answer quality with parameter count, not with driving safety. Its useful question is whether a much smaller text-conditioned pooling model can retain the language scores of larger systems. The measured latency and FLOPs below add the computational evidence that parameter count alone leaves out.

![Figure 1 from TS-VLM: Text-Guided SoftSort Pooling for Vision-Language Models in Multi-View Driving Reasoning](/assets/images/ts-vlm-text-guided-softsort-pooling-for-vision-language-models-in-multi-view-driving-reasoning-source-figure-1.webp)
*Fig 2: Model performance vs. model size on the DriveLM benchmark across four metrics: BLEU-4, METEOR, ROUGE-L, and CIDEr. Each circle represents a model, where the x-axis indicates model size (in millions of parameters, log scale), and the y-axis shows the performance score (↑ = better). | source: [TS-VLM, Figure 1](https://arxiv.org/abs/2505.12670)*


The paper's full comparison is on the DriveLM-NuScenes benchmark under one shared test set and an NVIDIA P100 environment. The 65.4M-parameter TS-VLM-Small reaches BLEU-4 56.82, METEOR 41.91, ROUGE-L 74.64, and CIDEr 3.39; the 20.1M Tiny model still reaches 53.59, 39.17, 73.72, and 3.33. Tiny uses 13.81B FLOPs and 0.11 GB, while the Small model uses 25.47B FLOPs and 0.31 GB; their measured per-frame inference times are 49.59 ms and 56.30 ms. On the Tiny ablation, SoftSort reaches BLEU-4 53.59 and CIDEr 3.331, versus 52.50 and 3.275 for hard top-one pooling and 50.80 and 3.249 for uniform pooling. SinkhornSort is more than 180× more expensive without improving the language metrics.

This figure-to-table path also states the boundary clearly. The evaluation is multi-view visual question answering and planning-language generation, not a closed-loop control policy; the reported latency is per frame on a P100, not an end-to-end vehicle control loop. The method can still discard a safety-critical view when its text query does not make that view salient, so a view-dropout or unexpected-hazard test under matched latency is the right deployment check.

## High-Level Takeaways

- TS-VLM changes multi-view fusion from all-to-all attention to query-conditioned ranking and pooling.
- Its reported DriveLM scores and small-model compute result support efficient driving reasoning, not direct closed-loop action quality.
- SoftSort retains secondary-view context that hard top-one pooling drops, while avoiding the much higher sorting cost of SinkhornSort in the reported ablation. Its remaining weakness is a relevant view receiving a low score from the question.
