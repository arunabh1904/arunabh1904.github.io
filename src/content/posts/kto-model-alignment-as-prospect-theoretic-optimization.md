---
title: 'KTO: Model Alignment as Prospect Theoretic Optimization'
date: '2024-02-02T00:00:00.000Z'
section: paper-shorts
postSlug: kto-model-alignment-as-prospect-theoretic-optimization
legacyPath: /paper shorts/2024/02/02/kto-model-alignment-as-prospect-theoretic-optimization.html
tags:
  - Alignment
  - Preference Optimization
field: 'Alignment & Post-Training'
summary: "2024 – KTO: Model Alignment as Prospect Theoretic Optimization"
---

## 2024 – KTO: Model Alignment as Prospect Theoretic Optimization

**arXiv:** [2402.01306](https://arxiv.org/abs/2402.01306)

**GitHub:** [ContextualAI/HALOs](https://github.com/ContextualAI/HALOs)

**Conference:** ICML 2024

## Summary

> KTO asks whether alignment data must arrive as a chosen–rejected pair. It derives a human-aware loss from prospect theory and trains directly on binary judgments: an output was desirable or undesirable. That interface matters when feedback occurs naturally as approval, a safety flag, or a deployment failure rather than as two completions from the same prompt.

## Core Insights

![KTO implied human value curves showing loss aversion and a reference point for preferred and rejected outcomes](/assets/images/kto-model-alignment-as-prospect-theoretic-optimization-paper-figure.png)
*Figure 1 makes KTO's prospect-theory motivation concrete: desirable and undesirable outcomes are valued relative to a reference point, with asymmetric sensitivity to gains and losses. source: [KTO](https://arxiv.org/abs/2402.01306)*

![Figure 2 from KTO: Model Alignment as Prospect Theoretic Optimization](/assets/images/kto-model-alignment-as-prospect-theoretic-optimization-source-figure-2.webp)
*Figure 2 HALOs (DPO, offline PPO variant) outperform non-HALOs (SLiC, CSFT), as measured by the GPT-4-0613 -judged winrate of the aligned model’s generations against a hard-to-beat baseline: the outputs that would have been used as the targets for SFT. The -axis here plots the winrate above chance (i.e., the winrate – 50%). The difference between methods is only significant at 13B+ parameters, and only the HALO-aligned Llama-{13B, 30B} models are able to match the baseline and yield a winrate at or above chance. source: [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)*

![Figure 3 from KTO: Model Alignment as Prospect Theoretic Optimization](/assets/images/kto-model-alignment-as-prospect-theoretic-optimization-source-figure-3.webp)
*Figure 3 KTO is as good or better than DPO at all scales, as measured by the GPT-4-0613 -judged winrate of the aligned model’s generations against the outputs that would have been used for SFT. In fact, for the Llama models, KTO alone matches the performance of SFT+DPO and is significantly better than DPO alone. Error bars denote a 90% binomial confidence interval. source: [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)*


KTO measures each completion relative to a reference point estimated from the policy and reference model. Desirable examples receive a gain-shaped objective; undesirable examples receive a loss-shaped objective. The asymmetry encodes loss aversion, while a KL term keeps the policy from moving arbitrarily far from its reference.

The paper places KTO, DPO, and PPO-style objectives inside a broader family called human-aware losses. Across 1B–30B language models, KTO matches or exceeds paired-preference methods in the reported comparisons despite using unpaired binary feedback. The result does not mean pairs are useless. It shows that a loss with the right inductive bias can extract value from a cheaper feedback interface.

| Design choice | KTO's answer | Operational consequence |
| --- | --- | --- |
| Feedback unit | One prompt–response labeled desirable or undesirable | Logs and moderation outcomes can become training data without constructing pairs. |
| Reference | Policy-relative utility with KL control | The reference distribution remains part of the method even without pairwise labels. |
| Main comparison | Binary feedback versus preference pairs | Data interface and objective must be evaluated together. |

## High-Level Takeaways

- KTO informs whether a post-training program should pay to construct matched preference pairs or learn from independent positive and negative outcomes. Its atomic unit is a labeled completion, not a pair. For robotics, that maps naturally to successful and failed action chunks, but only if the label really reflects the action under the state in which it was taken.
- The experiments establish that binary feedback can be competitive in the studied language-model regime. They do not establish that KTO handles continuous actions, irreversible transitions, or highly imbalanced failure logs. A decisive robot-policy test would compare KTO-style binary optimization, correction SFT, and paired preferences under the same rollout and annotation budget. The claim would weaken if binary training improves logged desirability while closed-loop recovery and safety remain unchanged.
- KTO is the clean bridge from paired language preferences to deployment signals that arrive one event at a time.
- The reference-point estimate, class balance, and mapping from token likelihood to physical action quality become new sources of error in VLA training.
- Use KTO when binary feedback is genuinely abundant; do not pretend that an unmatched failure and success form a counterfactual pair.
