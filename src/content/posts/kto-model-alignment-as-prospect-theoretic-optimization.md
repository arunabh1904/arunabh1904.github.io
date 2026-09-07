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

KTO changes the feedback required for preference optimization. DPO expects a chosen and rejected completion for the same prompt; KTO assigns each completion a desirable or undesirable label and estimates a reference point from the policy and reference model. Desirable examples receive a gain-shaped objective, undesirable examples receive a loss-shaped objective, while a detached KL estimate sets the utility reference point. This estimate is not a separate additive KL penalty backpropagated through the loss. The loss is therefore not “DPO with one item removed”: its reference point and asymmetric value function decide how an isolated label affects the update.

![KTO implied human value curves showing loss aversion and a reference point for preferred and rejected outcomes](/assets/images/kto-model-alignment-as-prospect-theoretic-optimization-paper-figure.png)
*Fig 1: The paper compares the implied value curves of prospect theory, PPO-Clip, and DPO. The reference point and asymmetric gain/loss response motivate KTO; these are conceptual value curves rather than measured KTO outcomes. | source: [KTO, Figure 1](https://arxiv.org/abs/2402.01306)*

In the default objective, the log probability ratio of a completion under the policy and reference model is compared with that reference point. Desirable examples use a sigmoid of the difference; undesirable examples reverse its sign. In practice, the reference point is estimated using mismatched prompt–response pairs within the microbatch, clamped nonnegative, and excluded from backpropagation. This is a convenient biased estimate, not an exact per-prompt KL calculation.

The paper places KTO, DPO, and PPO-style objectives inside a broader family called human-aware losses. Across 1B–30B language models, KTO matches or exceeds paired-preference methods in the reported comparisons despite using unpaired binary feedback. The result does not mean pairs are useless. It shows that a loss with the right inductive bias can extract value from a cheaper feedback interface.

The first comparison asks whether the broader human-aware loss family matters at all. Figure 2 compares HALO objectives such as DPO and offline PPO with conditional SFT and SLiC against the SFT target generations. The HALO methods are generally closer to or above chance, but the paper reports significant differences mainly at 13B and above; only the HALO-aligned Llama 13B and 30B models match or exceed the SFT target. The result is a scale-qualified comparison, not a universal win for every alignment loss.

![Figure 2 from KTO: Model Alignment as Prospect Theoretic Optimization](/assets/images/kto-model-alignment-as-prospect-theoretic-optimization-source-figure-2.webp)
*Fig 2: Across model sizes, HALO objectives such as DPO and offline PPO score closer to or above the SFT target than SLiC and conditional SFT in GPT-4-judged comparisons. | source: [KTO, Figure 2](https://arxiv.org/abs/2402.01306)*

| Design choice | KTO's answer | Operational consequence |
| --- | --- | --- |
| Feedback unit | One prompt–response labeled desirable or undesirable | Logs and moderation outcomes can become training data without constructing pairs. |
| Reference | Policy-relative utility with a detached KL reference point | The reference distribution remains part of the method even without pairwise labels. |
| Main comparison | Binary feedback versus preference pairs | Data interface and objective must be evaluated together. |

The KTO-specific comparison comes next. In Figure 3, SFT+KTO is competitive with SFT+DPO across Pythia and Llama scales. KTO alone is better than DPO alone for the Llama 7B, 13B, and 30B models, with the gap significant at 7B and 30B after the paper's multiple-comparison correction. The authors also note that KTO is more sensitive to learning rate than the other hyperparameters and recommend a larger practical learning rate than DPO; the exact setting is part of the result, not an implementation footnote.

![Figure 3 from KTO: Model Alignment as Prospect Theoretic Optimization](/assets/images/kto-model-alignment-as-prospect-theoretic-optimization-source-figure-3.webp)
*Fig 3: KTO is competitive with DPO across the Pythia and Llama scales; for the Llama models, KTO alone matches the reported SFT-plus-DPO comparison more closely than DPO alone. | source: [KTO, Figure 3](https://arxiv.org/abs/2402.01306)*

### Binary feedback changes the data contract

KTO is useful when feedback arrives one event at a time and pairing would be artificial. The experiments show that binary feedback can compete in the studied language-model regime, but they do not establish how the reference-point estimate behaves under extreme class imbalance or irreversible decisions. The unresolved comparison is simple: hold model, examples, and update budget fixed while comparing KTO with paired preferences and supervised correction. If KTO improves the logged label while an independent task metric stays flat, the reference point is fitting the feedback interface rather than the task.

## High-Level Takeaways

- KTO's atomic unit is a labeled completion, so approval, rejection, or moderation events can enter training without inventing a matched counterfactual pair.
- Prospect-theory shaping supplies the asymmetry between gains and losses; the KL reference still matters because the binary label alone does not define a target distribution.
- The reported advantage is scale- and protocol-dependent: KTO is competitive with DPO from 1B to 30B, while the strongest significance appears for the larger Llama comparisons.
- The open question is whether the binary data interface remains calibrated under class imbalance and task feedback that is delayed, continuous, or only weakly related to token likelihood.
