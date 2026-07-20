---
title: 'Scaling Laws of Motion Forecasting and Planning'
date: '2025-06-09T04:00:00.000Z'
section: paper-shorts
postSlug: scaling-laws-of-motion-forecasting-and-planning
legacyPath: /paper shorts/2025/06/09/scaling-laws-of-motion-forecasting-and-planning.html
tags:
  - Other
field: 'Motion Forecasting & Planning'
summary: "2025 – Scaling Laws of Motion Forecasting and Planning"
---
## 2025 – Scaling Laws of Motion Forecasting and Planning

**arXiv:** [2506.08228](https://arxiv.org/abs/2506.08228)

**Project:** [Waymo research page](https://waymo.com/research/scaling-laws-of-motion-forecasting-and-planning/)

**Summary:** Waymo's scaling study asks whether motion forecasting and planning improve predictably like language models. It trains encoder-decoder autoregressive Transformers and measures how performance changes with compute, data, model size, and inference-time sampling.

The result is encouraging for foundation-model-style autonomy: training loss, open-loop metrics, and even closed-loop metrics improve with scale. The paper also makes the resource tradeoff explicit instead of treating model size as the only knob.

## Paper Insights

The problem is scaling strategy for joint motion forecasting and planning. The study uses a 500 thousand hour driving dataset and fits empirical scaling laws over total training compute. It also studies compute-optimal allocation between model parameters and training data, inference-time compute through sampling and clustering, and the value of training on logged behavior from other agents.

The report finds power-law improvement with training compute and a strong correlation between training loss and evaluation metrics. One concrete scaling result is that as training compute grows, the compute-optimal recipe increases model size 1.5x as fast as dataset size. For inference, sampling and clustering can make smaller models competitive until a crossover point where a larger model becomes more compute-efficient. The caveat is that this is a large internal Waymo study, so the exact curves may not transfer to smaller public datasets.

![Figure 2 from Scaling Laws of Motion Forecasting and Planning showing training loss as a function of FLOPs](/assets/images/scaling-laws-of-motion-forecasting-and-planning-paper-figure.png)
_Figure 2 is the core scaling-law evidence: training loss improves predictably as total compute increases across the model family. From the [Scaling Laws of Motion Forecasting and Planning paper](https://arxiv.org/abs/2506.08228), via arXiv HTML._

**What to look at:**
- Closed-loop metrics improve with scale, not only open-loop forecasting metrics.
- Data/model allocation matters; bigger models are not the whole story.
- Inference-time sampling creates a separate scaling axis.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Data scale | 500 thousand hours of driving data | Large enough to expose scaling behavior. |
| Model family | Encoder-decoder autoregressive Transformers | Connects motion forecasting to sequence-model scaling. |
| Key result | Power-law improvement with compute | Supports systematic scaling rather than ad hoc model growth. |

**Compact result slice:**

| Scaling ingredient | Reported value | Why it matters |
| ------------------ | -------------- | -------------- |
| Driving data | 447 thousand hours and 5.6 million miles | Shows the curves come from fleet-scale data. |
| Training examples | 541 million | Gives the scaling study enough examples to vary data budgets. |
| Compute-optimal trend | Model size grows 1.5x as fast as dataset size | Bigger compute budgets should not be spent on data and parameters equally. |

## Decision Lens

This study informs how an autonomy program should split additional compute among model parameters, logged driving data, and inference-time trajectory sampling. Its atomic example is an autoregressive scene sequence drawn from fleet-scale logs; the reported curves connect training loss to open-loop and closed-loop forecasting/planning metrics.

The evidence supports predictable improvement within the measured model, data, and sampling ranges, including a compute-optimal trend in which model size grows faster than dataset size. It does not prove that the same exponents survive a new architecture, geography, or safety distribution. A held-out large run with fixed evaluation and confidence intervals is the decisive test. The law would fail operationally if closed-loop ranking reversed even while training loss followed the fitted curve.

**Context:** The paper moved motion forecasting and planning into the scaling-laws conversation with evidence that closed-loop behavior can improve predictably.

**Takeaway:** For motion and planning models, scaling is a three-way budget problem: training compute, data/model allocation, and inference-time sampling all matter.
