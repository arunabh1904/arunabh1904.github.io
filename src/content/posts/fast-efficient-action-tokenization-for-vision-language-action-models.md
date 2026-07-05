---
title: 'FAST: Efficient Action Tokenization for Vision-Language-Action Models'
date: '2025-01-01T05:00:00.000Z'
section: paper-shorts
postSlug: fast-efficient-action-tokenization-for-vision-language-action-models
legacyPath: /paper shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html
tags:
  - Other
field: Robotics
summary: FAST compressed robot action trajectories into tokens so autoregressive VLA models could learn dexterous control more efficiently.
---
## 2025 - FAST

**arXiv:** [2501.09747](https://arxiv.org/abs/2501.09747)

**Plain-language summary:** FAST tackles a low-level but important bottleneck in vision-language-action models: continuous robot actions are dense time series, while Transformers prefer discrete tokens. The paper uses time-series compression to tokenize action trajectories efficiently.

That enables autoregressive VLA models to train on complex manipulation trajectories without sequence lengths exploding.

![Figure 1: We propose FAST, a simple yet effective approach for tokenization of robot action trajectories via time-series compression from FAST: Efficient Action Tokenization for Vision-Language-Action Models](/assets/images/fast-efficient-action-tokenization-for-vision-language-action-models-paper-figure.jpg)
_Figure 1: We propose FAST, a simple yet effective approach for tokenization of robot action trajectories via time-series compression. From the [FAST: Efficient Action Tokenization for Vision-Language-Action Models paper](https://arxiv.org/abs/2501.09747), via arXiv HTML._

**What to look at:**
- FAST compresses dense action trajectories into discrete tokens.
- The goal is autoregressive VLA training without huge action sequences.
- This is an action-representation paper more than a new robot brain.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Problem | High-frequency continuous actions | Naive tokenization makes sequences too long. |
| Method | Time-series compression | Creates compact action tokens. |
| Signal | Faster VLA training | Makes autoregressive robot policies more practical. |

**Why it mattered:** Action representation is one of the hidden make-or-break details in robot foundation models. If actions are tokenized poorly, the model wastes capacity on formatting instead of behavior.

**Take-home message:** Robotics needs its own tokenization work. The action vocabulary matters as much as the text vocabulary.
