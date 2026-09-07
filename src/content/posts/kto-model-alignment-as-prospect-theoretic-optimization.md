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
*Fig 1: A schematic of KTO's prospect-theory motivation: desirable and undesirable outcomes are valued relative to a reference point, with asymmetric sensitivity to gains and losses. | source: [KTO, Figure 1](https://arxiv.org/abs/2402.01306)*

![Figure 2 from KTO: Model Alignment as Prospect Theoretic Optimization](/assets/images/kto-model-alignment-as-prospect-theoretic-optimization-source-figure-2.webp)
*Fig 2: Across model sizes, HALO objectives such as DPO and offline PPO score closer to or above the SFT target than SLiC and conditional SFT in GPT-4-judged comparisons. | source: [KTO, Figure 2](https://arxiv.org/abs/2402.01306)*

![Figure 3 from KTO: Model Alignment as Prospect Theoretic Optimization](/assets/images/kto-model-alignment-as-prospect-theoretic-optimization-source-figure-3.webp)
*Fig 3: KTO is competitive with DPO across the Pythia and Llama scales; for the Llama models, KTO alone matches the reported SFT-plus-DPO comparison more closely than DPO alone. | source: [KTO, Figure 3](https://arxiv.org/abs/2402.01306)*


KTO measures each completion relative to a reference point estimated from the policy and reference model. Desirable examples receive a gain-shaped objective; undesirable examples receive a loss-shaped objective. The asymmetry encodes loss aversion, while a KL term keeps the policy from moving arbitrarily far from its reference.

The paper places KTO, DPO, and PPO-style objectives inside a broader family called human-aware losses. Across 1B–30B language models, KTO matches or exceeds paired-preference methods in the reported comparisons despite using unpaired binary feedback. The result does not mean pairs are useless. It shows that a loss with the right inductive bias can extract value from a cheaper feedback interface.

| Design choice | KTO's answer | Operational consequence |
| --- | --- | --- |
| Feedback unit | One prompt–response labeled desirable or undesirable | Logs and moderation outcomes can become training data without constructing pairs. |
| Reference | Policy-relative utility with KL control | The reference distribution remains part of the method even without pairwise labels. |
| Main comparison | Binary feedback versus preference pairs | Data interface and objective must be evaluated together. |

### Decision test and boundary

KTO is the bridge from paired language preferences to deployment signals that arrive one event at a time. Its atomic unit is a labeled completion, so logs and moderation outcomes can be used without constructing a counterfactual pair. The experiments show binary feedback can compete in the studied language-model regime, but they do not establish behavior for continuous actions, irreversible transitions, or highly imbalanced failures. A robotics test should compare KTO, correction SFT, and paired preferences under the same rollout and annotation budget, then measure closed-loop recovery and safety. If binary optimization improves logged desirability while recovery is unchanged, the reference point is fitting the label rather than the task. Use KTO when binary feedback is genuinely abundant; an unmatched failure and success are not automatically a preference pair.
