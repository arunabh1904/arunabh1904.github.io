---
title: 'FAST: Efficient Action Tokenization for Vision-Language-Action Models'
date: '2025-01-01T05:00:00.000Z'
section: paper-shorts
postSlug: fast-efficient-action-tokenization-for-vision-language-action-models
legacyPath: /paper shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html
tags:
  - Other
field: 'Vision-Language-Action & Robotics'
summary: FAST compressed robot action trajectories into tokens so autoregressive VLA models could learn dexterous control more efficiently.
---
## 2025 - FAST

**arXiv:** [2501.09747](https://arxiv.org/abs/2501.09747)

**Summary:** FAST tackles a low-level but important bottleneck in vision-language-action models: continuous robot actions are dense time series, while Transformers prefer discrete tokens. The paper uses time-series compression to tokenize action trajectories efficiently.

That enables autoregressive VLA models to train on complex manipulation trajectories without sequence lengths exploding.

## Paper Insights

FAST improves action tokenization for autoregressive VLA policies. Simple per-timestep, per-dimension bins create long token sequences and handle high-frequency dexterous actions poorly. FAST compresses action chunks in frequency space using a discrete cosine transform, then quantizes the compact coefficients into action tokens. This preserves smooth temporal structure while shortening the sequence the model must generate. The evidence is better training and task performance for high-frequency robot behaviors. The caveat is compression design: too much compression loses control detail, while too little gives up the efficiency benefit.

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

## Decision Lens

FAST informs a representation decision that quietly controls the economics of autoregressive robot policies: how much of a continuous action trajectory deserves its own token. Instead of quantizing every timestep and control dimension independently, it transforms action chunks into frequency coefficients and discretizes the compressed representation. The model predicts a shorter sequence while retaining the smooth temporal structure that dexterous control needs.

The paper shows that action-aware compression can improve training efficiency and behavior, but average task success does not reveal which high-frequency corrections are lost. A missing stress test should stratify tasks by control bandwidth and compare FAST with learned tokenizers at equal bitrate and decoding latency. At ten times the horizon, small reconstruction errors can accumulate and rare abrupt motions may be suppressed by the frequency prior. The claim is falsified if compact tokens help offline likelihood but fail to improve—or reduce—closed-loop robustness under perturbations.

**Context:** Action representation is one of the hidden make-or-break details in robot foundation models. If actions are tokenized poorly, the model wastes capacity on formatting instead of behavior.

**Takeaway:** Robotics needs its own tokenization work. The action vocabulary matters as much as the text vocabulary.
