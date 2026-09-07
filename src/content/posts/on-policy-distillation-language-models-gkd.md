---
title: 'On-Policy Distillation of Language Models: GKD'
date: '2023-06-23T00:00:00.000Z'
section: paper-shorts
postSlug: on-policy-distillation-language-models-gkd
legacyPath: /paper shorts/2023/06/23/on-policy-distillation-language-models-gkd.html
tags:
  - Knowledge Distillation
  - Post-Training
field: 'Alignment & Post-Training'
topics:
  - language-systems
  - learning
summary: '2023 – On-Policy Distillation of Language Models: GKD'
---

## 2023 – On-Policy Distillation of Language Models: GKD

**arXiv:** [2306.13649](https://arxiv.org/abs/2306.13649)

**Conference:** ICLR 2024

## Summary

> Generalized Knowledge Distillation fixes a distribution mismatch in ordinary language-model distillation. A student trained only on teacher-written or human-written sequences learns under prefixes it may not visit at inference. GKD instead samples some sequences from the student, asks the teacher for a token distribution on those student-generated prefixes, and trains the student there. “On-policy” describes where the states come from; the method does not require a scalar reward or a policy-gradient estimator.

## Core Insights

![On-policy GKD improves student models across summarization, translation, and arithmetic compared with fixed-data distillation](/assets/images/on-policy-distillation-language-models-gkd-source-figure-1.webp)
*Fig 1: Across summarization, translation, and arithmetic, GKD on student-sampled outputs brings smaller T5 students closer to the T5-XL teacher than the fixed-data baselines. | source: [GKD, Figure 1](https://arxiv.org/abs/2306.13649)*

![Figure 3 from On-Policy Distillation of Language Models: GKD](/assets/images/on-policy-distillation-language-models-gkd-source-figure-3.webp)
*Fig 2: On XSum, the T5-Small student gains more from additional training data under on-policy GKD than the fixed-data baselines, showing the method's data-efficiency curve. | source: [GKD, Figure 3](https://arxiv.org/abs/2306.13649)*


Let $y_{<t}$ be a prefix sampled from either a fixed dataset or the current student. The teacher and student define next-token distributions $p_T(\cdot\mid x,y_{<t})$ and $p_S(\cdot\mid x,y_{<t})$. GKD minimizes a chosen divergence between those distributions and mixes the two prefix sources with a coefficient $\lambda$. At $\lambda=0$, training is conventional offline distillation; at $\lambda=1$, every target is evaluated on a student-generated trajectory.

This changes the feedback density. A terminal reward says whether the whole answer worked. GKD tells the student, at each visited prefix, how its full next-token distribution differs from the teacher. It can also choose the behavior of that correction: forward KL is more mode-covering, reverse KL is more mode-seeking, and Jensen–Shannon divergences interpolate. The best choice depends on whether diversity or concentrated generation matters.

| Setting | Evidence in the paper | Interpretation |
| --- | --- | --- |
| XSum summarization | On-policy GKD beats supervised KD and sequence KD | Student-prefix correction reduces exposure mismatch |
| WMT translation | Gains persist across student sizes | The mechanism is not tied to one output style |
| GSM8K arithmetic | On-policy variants give the largest relative gains | Correcting self-generated reasoning states is useful |
| FLAN task-agnostic distillation | Improves held-out BBH and MMLU | GKD can transfer across tasks, not only imitate one dataset |

The teacher is still the ceiling and the bottleneck. GKD assumes the starting student generates prefixes on which teacher probabilities are useful. If the student collapses into nonsense, teacher supervision on those states may spend compute far from the deployment boundary. Every on-policy sequence also requires both student generation and teacher scoring, so saved label collection becomes inference cost.

### Decision test and boundary

GKD is attractive when teacher logits are available and token-level correction is more informative than one sequence score. It has on-policy sampling but remains a differentiable supervised divergence: no reward, advantage, or credit assignment through future outcomes is required. When the teacher is much larger, its inference and logit transfer dominate the added cost. The decisive comparison refreshes an offline buffer at the same number of teacher tokens and measures exposure mismatch, diversity, and task quality. If replay matches on-policy performance, current-state coverage is not buying enough; if the teacher's preferred distribution encodes behavior the student must surpass, on-policy imitation becomes a ceiling. [DPO](/paper%20shorts/2023/05/01/direct-preference-optimization-dpo.html) uses fixed preference pairs, while [GRPO](/paper%20shorts/2024/02/05/deepseekmath-group-relative-policy-optimization-grpo.html) uses online states with a scalar reward. GKD occupies the dense-logit corner between them.
