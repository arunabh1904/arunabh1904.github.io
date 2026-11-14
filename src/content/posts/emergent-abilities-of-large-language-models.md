---
title: 'Emergent Abilities of Large Language Models'
date: '2022-06-15T00:00:00.000Z'
section: paper-shorts
postSlug: emergent-abilities-of-large-language-models
legacyPath: /paper shorts/2022/06/15/emergent-abilities-of-large-language-models.html
tags: [Scaling Laws, Evaluation, Prompting]
field: 'Language Models'
summary: '2022 – Emergent Abilities of Large Language Models'
---

## 2022 – Emergent Abilities of Large Language Models

**Paper:** [Wei et al., arXiv:2206.07682v2](https://arxiv.org/abs/2206.07682v2)

## Summary

> Wei et al. collect tasks whose measured performance stays near chance in smaller language models and improves substantially at larger scales. They also include techniques, such as chain-of-thought prompting, whose benefit appears only for larger models in the reported setup. This is an empirical taxonomy of observed score curves, not a proof that a fixed parameter threshold creates an ability. The paper itself documents hidden improvements under alternative metrics and thresholds that move with the training recipe.

## Core Insights

### Emergence is defined through an evaluation curve

The definition extends the distinction in [Predictability and Surprise](/paper%20shorts/2022/02/15/predictability-and-surprise-in-large-generative-models.html): an ability is called emergent when it is absent at smaller measured scales and present at larger ones. The operational evidence is a task score near its random baseline followed by a substantial increase. The underlying model weights and generated outputs need not undergo a literal discontinuity for that pattern to appear.

The main figure collects eight examples rather than averaging them into one capability score.

![Original source Figure 2 shows eight task-score curves across LaMDA, GPT-3, Gopher, Chinchilla, and PaLM.](/assets/images/emergent-abilities-source-figure-2.png)
*Fig 1: Each point is a separately trained model, and each panel uses its task's evaluation metric and random baseline. The apparent threshold depends on the family, task, and sampled training scales. | source: [Paper, Figure 2](https://arxiv.org/html/2206.07682v2#S3.F2)*

The image is the unchanged reproduction of the original figure supplied in [Schaeffer et al., Figure 1](https://arxiv.org/html/2304.15004v2#S1.F1). [Open it at full size](/assets/images/emergent-abilities-source-figure-2.png).

The two-shot arithmetic example rises above the near-zero region around 13B parameters for GPT-3 and 68B for LaMDA. Those different locations already argue against a task-intrinsic parameter threshold. The paper explicitly treats compute, parameters, data quality, tokenization, architecture, and prompting as correlated variables; its cross-family comparisons do not ablate all of them.

### A technique can have a scale-dependent benefit

The second category compares a technique against its baseline at each scale. In the reported LaMDA experiments, chain-of-thought prompting helps mathematical reasoning only at larger sizes; instruction tuning also changes from harmful to beneficial as scale increases. The relevant quantity is the difference between two procedures on the same family, rather than whether either procedure passes a fixed score.

This does not imply that reasoning prompts or instruction tuning require roughly 100B parameters in general. The paper discusses smaller encoder-decoder models that learn instruction following, and lists 14 BIG-Bench tasks where PaLM 62B outperforms much larger earlier models that remain near chance. Better recipes can move an observed threshold.

### The appendix already exposes the metric problem

Appendix A evaluates six BIG-Bench tasks with cross-entropy as well as their headline exact-match, BLEU, or accuracy scores. All six show improvement in target likelihood while the headline scores remain near chance. Thus “no visible capability” under one metric does not mean no learning has occurred.

The authors also test partial-credit metrics on three generative tasks and still see sharp-looking curves. They therefore argue that exact match alone is an incomplete explanation. Their task taxonomy is partly judgment-based: two co-authors agree on labels for the 210-task analysis, while 40 tasks too noisy or otherwise unsuitable are excluded from the plotted categories. These are useful observations with explicit measurement choices, not an automatic detector of phase transitions.

The later [Mirage paper](/paper%20shorts/2023/04/28/are-emergent-abilities-of-large-language-models-a-mirage.html) makes metric dependence the central controlled intervention. Read together, the papers motivate evaluating both complete task success and graded progress, with enough examples and model sizes to distinguish a real change in behavior from the resolution of the measurement.

## High-Level Takeaways

- The paper identifies a pattern in observed task scores, not a universal critical model size.
- Technique benefit can change with scale even when the underlying model family is held fixed.
- Its own likelihood analysis finds progress hidden by flat headline metrics.
- The practical follow-up is to vary metrics and sampling density before interpreting a sharp curve as a new mechanism.
