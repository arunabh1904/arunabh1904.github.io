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

![On-policy GKD improves student models across summarization, translation, and arithmetic compared with fixed-data distillation](/assets/images/on-policy-distillation-language-models-paper-figure.png)
_Across three tasks and student sizes, the on-policy variants deliver the largest gains over the initial student. The figure reports task-specific performance rather than claiming one divergence is universally best. Source: [GKD](https://arxiv.org/abs/2306.13649)._

Let $y_{<t}$ be a prefix sampled from either a fixed dataset or the current student. The teacher and student define next-token distributions $p_T(\cdot\mid x,y_{<t})$ and $p_S(\cdot\mid x,y_{<t})$. GKD minimizes a chosen divergence between those distributions and mixes the two prefix sources with a coefficient $\lambda$. At $\lambda=0$, training is conventional offline distillation; at $\lambda=1$, every target is evaluated on a student-generated trajectory.

This changes the feedback density. A terminal reward says whether the whole answer worked. GKD tells the student, at each visited prefix, how its full next-token distribution differs from the teacher. It can also choose the behavior of that correction: forward KL is more mode-covering, reverse KL is more mode-seeking, and Jensen–Shannon divergences interpolate. The best choice depends on whether diversity or concentrated generation matters.

| Setting | Evidence in the paper | Interpretation |
| --- | --- | --- |
| XSum summarization | On-policy GKD beats supervised KD and sequence KD | Student-prefix correction reduces exposure mismatch |
| WMT translation | Gains persist across student sizes | The mechanism is not tied to one output style |
| GSM8K arithmetic | On-policy variants give the largest relative gains | Correcting self-generated reasoning states is useful |
| FLAN task-agnostic distillation | Improves held-out BBH and MMLU | GKD can transfer across tasks, not only imitate one dataset |

The teacher is still the ceiling and the bottleneck. GKD assumes the starting student generates prefixes on which teacher probabilities are useful. If the student collapses into nonsense, teacher supervision on those states may spend compute far from the deployment boundary. Every on-policy sequence also requires both student generation and teacher scoring, so saved label collection becomes inference cost.

## High-Level Takeaways

- GKD informs whether distillation data should follow a fixed corpus or the student's current state distribution. The atomic feedback unit is a teacher distribution at a student-visited prefix. This is attractive when the teacher is trusted, logits are accessible, and token-level correction is more informative than one sequence score.
- The method should not be conflated with reinforcement learning. It has on-policy sampling but a differentiable supervised divergence; no reward, advantage, or credit assignment through future outcomes is required. The paper also shows that GKD can be mixed with a reward objective, which makes the distinction operational: distillation supplies dense local imitation while RL supplies task-level preference.
- At ten times model size, teacher inference dominates cost and logit transfer becomes a systems problem. The claim would weaken if replaying a carefully refreshed offline buffer matched on-policy performance at the same number of teacher tokens. It would fail outright when the teacher's preferred distribution encodes the behavior the student is meant to surpass.
- [DPO](/paper%20shorts/2023/05/01/direct-preference-optimization-dpo.html) learns from fixed chosen/rejected responses. [GRPO](/paper%20shorts/2024/02/05/deepseekmath-group-relative-policy-optimization-grpo.html) samples online but reduces each completion to a relative scalar reward. GKD occupies the third corner: online states with dense teacher distributions.
- GKD makes distillation on-policy by training the student against teacher logits on prefixes generated by the current student.
- It needs teacher access and substantial inference, inherits teacher errors, and assumes the initial student can visit useful states.
- On-policy distillation is best understood as dense correction on the learner's own mistakes, not as another name for reinforcement learning.
