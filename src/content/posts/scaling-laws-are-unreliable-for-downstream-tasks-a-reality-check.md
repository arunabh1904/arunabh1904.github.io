---
title: "Scaling Laws Are Unreliable for Downstream Tasks: A Reality Check"
date: '2025-07-01T00:00:00.000Z'
section: paper-shorts
postSlug: scaling-laws-are-unreliable-for-downstream-tasks-a-reality-check
legacyPath: /paper shorts/2025/07/01/scaling-laws-are-unreliable-for-downstream-tasks-a-reality-check.html
tags:
  - Scaling Laws
  - Evaluation
  - Language Models
field: "Language Models"
summary: "2025 – Scaling Laws Are Unreliable for Downstream Tasks: A Reality Check"
---

## 2025 – Scaling Laws Are Unreliable for Downstream Tasks: A Reality Check

**Paper:** [Lourie, Hu, and Cho, arXiv:2507.00885v2](https://arxiv.org/abs/2507.00885v2)

## Summary

> Downstream scaling is a conditional empirical pattern, not a default property of a benchmark. Reanalyzing 46 tasks from Gadre et al., the paper classifies 18 (39%) as smoothly predictable in that setup; the remaining tasks are inverse, nonmonotonic, noisy, trendless, or breakthrough. Changing the validation corpus, task formatting, or evaluation harness can also change the qualitative trend, so the result is a warning about transportability rather than a universal success rate.

## Core Insights

### The proxy is part of the experiment

Pretraining loss is a useful coordinate for comparing models, but it is not a task-independent capability meter. The authors revisit checkpoints from Gadre et al. (2025), measuring cross-entropy on either C4 or Paloma's 100 Programming Languages and comparing it with few-shot task accuracy. A scaling claim therefore has three coupled objects: the pretraining corpus, the validation corpus used to measure loss, and the downstream task.

That coupling changes conclusions even before the model is enlarged. On HellaSwag, C4 and RedPajama checkpoints have nearly overlapping trends when loss is measured on C4; using 100 PLs makes C4 look better despite a worse validation loss. On CoQA, changing the validation corpus reverses which pretraining corpus appears to improve first. Better perplexity is not a portable ordering of data mixtures.

![The paper's taxonomy shows predictable, inverse, nonmonotonic, noisy, trendless, and breakthrough downstream curves against C4 cross-entropy.](/assets/images/downstream-scaling-reality-check-source-figure-2.png)
*Fig 1: Six qualitative behaviors appear across the evaluated tasks: the predictable panel follows the smooth improvement sought by the scaling forecast. The others exhibit declining, reversing, noisy, flat, or thresholded behavior over the observed range. | source: [Lourie et al., Figure 2](https://arxiv.org/html/2507.00885v2#S1.F2)*

[Open the source figure at full size](/assets/images/downstream-scaling-reality-check-source-figure-2.png)

### “39% predictable” is a classification inside one corpus of experiments

The 39% figure comes from an informal qualitative classification of the 46 tasks studied by Gadre et al., with fitted curves and $R^2$ reported separately. It is not the probability that a new benchmark will obey a law, nor a held-out forecast success rate. The paper deliberately avoids defining a single numerical threshold that would make the result depend on one chosen functional form. This makes the diagnosis useful, but it also means the class boundary is part of the authors' judgment.

The non-predictable cases are not a single failure mode. Inverse scaling means performance falls as C4 loss improves; nonmonotonic scaling switches direction; noisy scaling has a visible trend with large deviations; trendless scaling has no reliable direction over the observed loss range; breakthrough scaling is sigmoid-like and difficult to extrapolate before the transition. Treating all five as “noise” would hide different reasons a small-model proxy can fail.

### Cross-study differences are confounds, not causal ablations

The paper compares ten overlapping tasks from Gadre et al. and Magnusson et al. (2025), evaluating both sets on C4 where possible. MMLU is positive in both studies but differs in shape and noise. CommonsenseQA is nonmonotonic in one setup and cleanly increasing in the other; BoolQ looks trendless in the Magnusson checkpoints partly because their losses cover a narrower range. The studies also differ in architecture, prompts, number of shots, task formulation, and answer choices.

Those comparisons establish sensitivity, not which detail caused it. Appendix A records that LLM Foundry and OLMES ship different task versions, and that Foundry v0.4.0 contained an SIQA label error; the authors exclude SIQA from the main analysis and retain it only for the completeness figure. Appendix B adds C4 validation-loss measurements for 200 released DataDecide checkpoints: five sizes from 20M to 1B, five pretraining corpora, and eight training stages from 5,000 to 40,000 steps. These evaluations use A100 and H100 inference. These details matter because a change in harness can change the random baseline and the apparent slope.

The paper's result complements the earlier [loss-based overtraining note](/paper%20shorts/2024/03/13/language-models-scale-reliably-with-overtraining-and-on-downstream-tasks.html), which shows that aggregate downstream error can be forecast after adding a larger anchor run, while individual tasks remain noisy. This paper pushes the boundary earlier: before choosing a fitting law, verify that the task, validation corpus, and exact evaluation protocol produce a stable curve at all. The later [direct downstream-metrics scaling study](/paper%20shorts/2025/12/09/revisiting-the-scaling-properties-of-downstream-metrics-in-large-language-model-training.html) is a useful controlled counterpoint, because its positive result is explicitly tied to a fixed recipe, benchmark set, and compute range.

## High-Level Takeaways

- Treat downstream scaling as a hypothesis to test for each task, validation corpus, and harness, not as a consequence of lower pretraining loss.
- Preserve the qualitative distinction between inverse, nonmonotonic, noisy, trendless, and breakthrough curves; they imply different extrapolation failures.
- Read 18/46 = 39% as an in-study classification, not a general benchmark success probability or a preregistered prediction score.
- Cross-study comparisons reveal protocol sensitivity, but the paper does not isolate architecture, prompts, shots, or answer-choice count as causal factors.
- A useful next experiment is a matched protocol sweep that holds the checkpoints and validation corpus fixed while changing one evaluation choice at a time, then tests forecasts on withheld large checkpoints.
