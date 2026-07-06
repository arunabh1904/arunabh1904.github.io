---
title: 'AutoTrust: Benchmarking Trustworthiness in Large Vision Language Models for Autonomous Driving'
date: '2024-12-01T05:00:00.000Z'
section: paper-shorts
postSlug: autotrust-benchmarking-trustworthiness-in-large-vision-language-models-for-autonomous-driving
legacyPath: /paper shorts/2024/12/01/autotrust-benchmarking-trustworthiness-in-large-vision-language-models-for-autonomous-driving.html
tags:
  - Other
field: Autonomous Driving
summary: AutoTrust evaluated driving VLMs across hallucination, safety, robustness, privacy, and fairness.
---
## 2024 - AutoTrust

**arXiv:** [2412.15206](https://arxiv.org/abs/2412.15206)

**Plain-language summary:** AutoTrust asks whether driving VLMs can be trusted, not merely whether they answer benchmark questions correctly. It probes five dimensions: truthfulness, safety, robustness, privacy, and fairness.

That means questions can test whether a model hallucinates, gives unsafe advice, leaks sensitive information, breaks under perturbations, or behaves inconsistently across groups and regions.

## Paper Insights

AutoTrust reframes driving VLM evaluation around trustworthiness. It tests truthfulness, safety, robustness, privacy, and fairness over more than 10k scenes and 18k queries. The benchmark is designed to expose hallucination, unsafe advice, sensitive-information leakage, adversarial brittleness, and unfair or inconsistent behavior across driving contexts. One important finding is that driving specialization does not automatically improve trustworthiness; general VLMs can outperform specialist driving models on some axes. The limitation is that VQA-style trust tests still do not replace closed-loop validation, but they reveal failures ordinary accuracy benchmarks miss.

![Figure 1 from AutoTrust: benchmark overview for DriveVLM trustworthiness](/assets/images/autotrust-benchmarking-trustworthiness-in-large-vision-language-models-for-autonomous-driv-paper-figure.png)
_Figure 1 from the [AutoTrust paper](https://arxiv.org/abs/2412.15206), via arXiv HTML._

**What to look at:**
- Trustworthiness is split into truthfulness, safety, robustness, privacy, and fairness.
- The benchmark includes adversarial and sensitive-information probes.
- A driving-specialized model can still be less trustworthy than a general model.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Dataset | 10k scenes / 18k queries | Designed around trust dimensions. |
| Axes | Truthfulness, safety, robustness, privacy, fairness | Evaluates more than accuracy. |
| Main finding | Hidden unsafe behaviors | Capability can improve while trust still lags. |

**Why it mattered:** Trustworthiness is not one metric. A model can improve on standard driving QA while still becoming less safe or less private.

**Take-home message:** Driving VLM evaluation needs adversarial and ethical dimensions baked in from the start. Accuracy alone is too small a target.
