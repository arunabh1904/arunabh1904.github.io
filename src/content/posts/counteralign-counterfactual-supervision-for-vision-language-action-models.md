---
title: 'CounterAlign: Counterfactual Supervision for Vision-Language-Action Models'
date: '2026-08-22T09:00:00.000Z'
section: paper-shorts
postSlug: counteralign-counterfactual-supervision-for-vision-language-action-models
legacyPath: /paper shorts/2026/08/22/counteralign-counterfactual-supervision-for-vision-language-action-models.html
tags: [Vision-Language-Action, Offline RL, Robot Post-Training]
field: 'Robot Post-Training & Evaluation'
summary: '2026 – CounterAlign: Counterfactual Supervision for Vision-Language-Action Models'
---

## 2026 – CounterAlign: Counterfactual Supervision for Vision-Language-Action Models

**Paper:** [arXiv:2608.21740](https://arxiv.org/abs/2608.21740) · [Full text](https://arxiv.org/html/2608.21740v1)

## Summary

> CounterAlign improves a π0.5 policy's robustness by learning corrective rewards from the successful demonstrations already available. On LIBERO-PRO, its largest gains over behavior cloning appear when object positions or task requirements change. The cost moves from data collection to training: the policy architecture stays fixed, but the added reward and critic networks raise the reported trainable parameter count from approximately 3.4B to 12.7B. This is an offline post-training option for a data-constrained robot team with spare training memory, not evidence that additional interaction is unnecessary.

## Core Insights

### A successful action can be wrong for another instruction

Behavior cloning teaches the action recorded for an observation and instruction. It does not explicitly identify an action that is plausible in the same scene but inconsistent with the command. CounterAlign creates that contrast by replacing an instruction or action chunk with another from the dataset. Similarity filters avoid both obvious mismatches and pairs too similar to label reliably. The unit of supervision remains an observation, instruction, and action chunk; the new information is their compatibility.

Two discriminators evaluate that tuple. One distinguishes expert actions from policy-generated actions at logged observations. The other learns instruction–action compatibility from relabeled examples. The relabeling panel below shows how the observation stays fixed while instructions or actions are substituted.

![Panel b of CounterAlign Figure 2: the relabeling discriminator receives the same observation with original or substituted instructions and action chunks](/assets/images/counteralign-source-figure-2b.png)
*Fig 1: The relabeling discriminator compares instruction–action pairings while holding the observation fixed. Cropped to panel b of the source figure so the semantic supervision is readable. | source: [CounterAlign, Figure 2b](https://arxiv.org/html/2608.21740v1#S3.F2)*

Relabeling can also produce a valid alternative. CounterAlign therefore treats jointly changed instructions and actions as unlabeled examples under non-negative positive–unlabeled learning. It does not force every synthetic pair to be a failure. That distinction matters whenever several commands share a valid intermediate motion: a useful negative must expose a semantic error without penalizing an action that could satisfy either command.

### Relabeling changes the reward and critic before the actor

The reward combines the two log discriminator scores. An Implicit Q-Learning critic estimates advantage, and the actor uses that advantage to weight its flow-matching loss. Higher-valued demonstration chunks receive more weight. The main method keeps the actor's targets in the original expert data; actor-side relabeling is an ablation and does not consistently help.

Critic relabeling changes only the instruction. Substituting an action would leave the logged transition without the next observation required for its bootstrapped target. This is an important boundary: a synthetic instruction–action mismatch can train a compatibility score, but it does not supply the physical outcome of executing that action.

That separates CounterAlign from the experience-collection route discussed in [π*0.6](/paper%20shorts/2025/11/18/pi-star-0-6-vla-learns-from-experience.html). CounterAlign extracts more supervision from existing demonstrations; collecting rollouts and corrections can expose states absent from those demonstrations. These are different ways to improve a policy, and the paper does not compare their cost-adjusted returns directly.

### The gains depend on the perturbation

The following success rates come from Table 1. Each cell compares behavior-cloned π0.5 with CounterAlign, in that order; the paper reports 100 evaluation trials.

| LIBERO-PRO suite | Changed object position | Changed task |
| --- | --- | --- |
| Spatial | 53% → 60% | 55% → 63% |
| Object | 19% → 51% | 10% → 26% |
| Goal | 29% → 41% | 17% → 46% |
| LIBERO-10 | 6% → 11% | 17% → 29% |

This is not an across-the-board improvement. Under changed object appearance, success falls in all four suites; the Object suite drops from 89% to 80%. The reported CALVIN average task-chain length changes more modestly, from 3.93 to 4.03. On the TX-G2 robot, all four reported task scores improve, but evaluation uses only ten perturbed trials per task and averages success across primitive actions. Those scores should not be read as whole-task completion rates.

The ablations also limit attribution. The basic offline RL configuration already improves several position and task results before the full relabeling recipe is added. The comparison with behavior cloning therefore measures the complete training change, not the isolated effect of counterfactual examples.

### No new rollouts still leaves a substantial training bill

The appendix reports LIBERO training for 15,000 steps at batch size 256 on eight H200 GPUs, taking approximately 28 hours. It states that behavior cloning can fit on one H200, while CounterAlign needs at least five because of its auxiliary networks. TX-G2 training runs for 150,000 steps at batch size 64 on eight H200s, taking approximately 96 hours. The larger training system leaves the inference policy's architecture unchanged.

My decision test would compare this extra training against spending the same budget on more varied demonstrations, using repeated runs and the same perturbation suites. A matched-budget data-collection comparison and repeated-training-seed uncertainty are not reported. Without them, the paper supports testing CounterAlign where collecting robot data is difficult; it does not establish the cheapest route to robustness.

## High-Level Takeaways

- CounterAlign makes instruction–action compatibility an explicit reward while retaining the π0.5 policy architecture. Its new capacity sits in the training system.
- Relabeled tuples provide corrective supervision without supplying new physical transitions. Ambiguous joint relabelings remain unlabeled rather than becoming automatic negatives.
- Position and task perturbations show the clearest gains; appearance regressions, modest CALVIN improvement, and limited real-robot trials constrain the generalization claim.
- The practical trade-off is robot data versus auxiliary training memory. I would prefer the method only if repeated, matched-budget tests beat either stronger behavior cloning or collecting more varied demonstrations.
