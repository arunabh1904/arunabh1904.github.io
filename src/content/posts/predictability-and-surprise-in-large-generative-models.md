---
title: 'Predictability and Surprise in Large Generative Models'
date: '2022-02-15T00:00:00.000Z'
section: paper-shorts
postSlug: predictability-and-surprise-in-large-generative-models
legacyPath: /paper shorts/2022/02/15/predictability-and-surprise-in-large-generative-models.html
tags: [Scaling Laws, Evaluation]
field: 'Language Models'
summary: '2022 – Predictability and Surprise in Large Generative Models'
---

## 2022 – Predictability and Surprise in Large Generative Models

**Paper:** [Ganguli et al., arXiv:2202.07785v2](https://arxiv.org/abs/2202.07785v2)

## Summary

> A predictable training-loss curve can coexist with surprising task behavior. Ganguli et al. separate smooth aggregate scaling from abrupt task scores, unanticipated uses, and unpredictable individual outputs. Their argument explains why a scaling law can justify investment in a larger model without forecasting the full consequences of deploying it. The evidence combines earlier model results with targeted experiments; it does not establish a universal law of capability emergence.

## Core Insights

### A distributional average is not a task forecast

[Kaplan's scaling laws](/paper%20shorts/2020/01/23/scaling-laws-for-neural-language-models.html) predict loss over a broad text distribution. A specific task occupies only part of that distribution and uses its own scoring rule. Smooth improvement in the average therefore places a weak constraint on the shape of an individual task's score. The paper's distinction is between the quantity that can be forecast and the behavior a user actually cares about.

The historical examples make the gap visible.

![Source Figure 2 compares abrupt task-score changes for arithmetic, language understanding, and program synthesis.](/assets/images/predictability-and-surprise-source-figure-2.png)
*Fig 1: Task scores can remain low across several sampled model sizes before rising sharply. These curves summarize different published model families, not a single experiment that isolates parameter count from training data and compute. | source: [Paper, Figure 2](https://arxiv.org/html/2202.07785v2#S2.F2)*

[Open the figure at full size](/assets/images/predictability-and-surprise-source-figure-2.png)

For three-digit addition, the reported GPT-3 accuracy is below 1% at sizes under 6B, about 8% at 13B, and about 80% at 175B. That is a striking observed curve, but sparse model sizes and the scoring function limit what can be inferred about a discontinuity between them. The later [emergence survey](/paper%20shorts/2022/06/15/emergent-abilities-of-large-language-models.html) catalogues such cases; [the metric critique](/paper%20shorts/2023/04/28/are-emergent-abilities-of-large-language-models-a-mirage.html) tests whether their apparent sharpness survives rescoring fixed outputs.

### Unexpected applications are a separate uncertainty

The original recommendation experiment illustrates smooth scaling on a concrete task. The authors provide a user's past MovieLens ratings in a prompt, score the next-token probabilities of ratings one through five, and compute their probability-weighted mean. On a 10,000-rating test set, the largest model reaches 0.94 RMSE, versus 0.98 for a non-personalized mean-rating baseline and the paper's cited specialist result of about 0.82.

“Zero-shot” here means no task-specific weight training; the prompt still contains that user's observed ratings. The result does not show that a language model beats a specialized recommender. It shows that a general text model can support an application outside its original evaluation suite, and that measuring the application's own scaling curve can reveal an unattractive compute requirement.

The paper also prompts models on a recidivism dataset and measures toxicity on 1,000 shared prompts with 25 responses per prompt. These experiments expose unanticipated uses and undesirable behavior rather than recommend those uses. Their limitations matter: criminal-justice data contain measurement biases, fairness metrics disagree, and automated toxicity scores are imperfect proxies for human judgments. Improved aggregate accuracy cannot resolve those problems by itself.

### Predictability changes incentives before it removes uncertainty

The broader argument is that predictable loss makes a large training run easier to budget, while open-ended use makes its eventual behavior harder to characterize. The authors consequently advocate broader evaluation access, red teaming, and independent scrutiny. Those are proposals supported by the technical asymmetry, not interventions experimentally shown here to eliminate it.

For reading a scaling paper, the useful question is which forecast has actually been validated: held-out loss, a specified task score, or behavior under new prompts. Moving between those levels requires new evidence.

## High-Level Takeaways

- Smooth average loss does not determine the curve for every downstream task.
- A new prompted application can reveal a capability absent from the original benchmark suite.
- Scaling a task score does not guarantee desirable behavior or a competitive deployment cost.
- The paper motivates broader evaluation; its abrupt-score examples do not settle the later debate about metric-induced emergence.
