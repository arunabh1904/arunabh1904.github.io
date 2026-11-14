---
title: "Self-Supervised On-Policy Distillation for Reasoning Language Models"
date: '2026-05-17T00:00:00.000Z'
section: paper-shorts
postSlug: self-supervised-on-policy-distillation-for-reasoning-language-models
legacyPath: /paper shorts/2026/05/17/self-supervised-on-policy-distillation-for-reasoning-language-models.html
tags:
  - Distillation
  - GRPO
field: "Alignment & Post-Training"
summary: "2026 – Self-Supervised On-Policy Distillation for Reasoning Language Models"
---

## 2026 – Self-Supervised On-Policy Distillation for Reasoning Language Models

**Paper:** [Tan and Hong, arXiv:2605.17497v1](https://arxiv.org/abs/2605.17497v1) · **Code link provided by the paper:** [SSOPD](https://github.com/tzq1999/SSOPD)

## Summary

> SSOPD reuses successful and failed attempts from the same GRPO group to create token-level guidance without external solution traces. A frozen teacher sees the shortest correct attempt as hindsight while scoring prefixes from the longest failed attempt; the student learns those distributions without seeing the successful solution. On Qwen3-8B, macro average correctness across three competition-math benchmarks rises from GRPO's 64.0 to 65.6, while smaller models still benefit more from a baseline using external solution information.

## Core Insights

A mixed GRPO group contains more information than its scalar rewards: one trajectory demonstrates a successful route, while another records states the current policy visits before failing. SSOPD adds a distillation objective at those failed states. The correct trajectory supplies context to a teacher; it is not copied as the student's supervised output sequence.

Follow one training pair through the source diagram. Select the shortest verified-correct completion $y_+$ and the longest wrong completion $y_-$. At a prefix of $y_-$, the student sees the question and that prefix. The teacher sees the same state plus $y_+$. Its next-token distribution becomes the student's local target.

![Source Figure 1 follows a mixed GRPO group through witness selection, failed-prefix distillation, and the weighted combined objective.](/assets/images/ssopd-source-figure-1.png)
*Fig 1: The successful completion is teacher context, while the failed completion supplies student-visited prefixes. The auxiliary loss acts only when a group contains both outcomes and leaves the inference-time student without privileged solution context. | source: [SSOPD, Figure 1](https://arxiv.org/html/2605.17497v1#S1.F1)*

[Open the figure at full size](/assets/images/ssopd-source-figure-1.png)

### The teacher has more context, not stronger weights

In the implementation, the teacher is the initial Qwen3 checkpoint with trainable LoRA adapters disabled. The student uses rank-64 LoRA adapters on attention and MLP projections. Teacher scoring runs with stop-gradient, and full-vocabulary distributions supervise at most the first 1,024 generated positions of the selected failed response.

The auxiliary objective uses a forward-KL-style sum with each action-level contribution capped at 0.05 before summation. This pointwise clipping is distinct from GRPO's policy-ratio clipping. Reusing the same rollout group avoids extra trajectory generation, but teacher scoring and the auxiliary student computation still add training work. An information-matched comparison is not automatically a compute-matched comparison.

The prompt weight is

$$
\lambda_x=4\lambda_0\hat p_x(1-\hat p_x),\qquad \lambda_0=0.5,
$$

where $\hat p_x$ is the group's fraction of correct responses. Balanced groups receive the largest auxiliary weight. All-correct or all-wrong groups receive zero: no usable success–failure pair exists. This differs from the later [AVSPO note](/paper%20shorts/2026/05/20/advantage-collapse-in-group-relative-policy-optimization.html), whose intervention specifically changes normalization in homogeneous groups.

### What the stopping-time argument establishes

The theory scores success under a random deadline, favoring successful continuations that finish sooner. Within an observed finite group, the shortest correct completion maximizes that criterion, while a long failed completion identifies a persistently unsuccessful route. This motivates the pairing rule; it does not prove that shorter reasoning is generally superior.

The local improvement result assumes access to an ideal distribution reweighted by future success. Replacing it with the actual hindsight-conditioned teacher introduces an approximation-error term. Improvement requires that error to be smaller than the useful variation among next actions. The theorem concerns a local distribution edit, not a guarantee that a shared neural-network update improves every prefix.

### Small consistent gains, with a clear information boundary

Training uses up to 30,000 verified-answer math prompts from OpenThoughts. GRPO and SSOPD receive no reference reasoning traces; SFT and the solution-conditioned OPSD baseline do. Runs use eight samples per prompt, rollout temperature 1.2, a 16,000-token rollout cap, learning rate $5\times10^{-6}$, and effective batch size 32.

Evaluation uses thinking mode, temperature 1.0, top-$p$ 0.95, and a maximum of 38,912 generated tokens. Avg@12 is average correctness across 12 attempts per problem, not best-of-12 or majority-vote accuracy.

| Qwen3 backbone | GRPO macro Avg@12 | SSOPD | External-solution OPSD |
|---|---:|---:|---:|
| 1.7B | 37.7 | 38.3 | 43.4 |
| 4B | 62.7 | 63.0 | 63.6 |
| 8B | 64.0 | 65.6 | 64.8 |

The macro average covers AIME 2024, AIME 2025, and HMMT 2025. All nine SSOPD benchmark entries improve over GRPO, but the 4B margin is small and the paper does not report multi-seed uncertainty. At smaller scales, external successful traces remain more useful than the model's own witnesses.

The frontier-weight ablation improves HMMT from 44.7 with a fixed coefficient to 45.0 with the default dynamic weight. Selector ablations also favor shortest-correct/longest-wrong, but report the best score across checkpoints on HMMT. They support the choice within that sweep rather than an untouched-test estimate of selector generalization.

My next comparison would match total training FLOPs, report multiple seeds, and measure how often selected correct answers contain sound reasoning. Final-answer verification supplies the success label; it does not independently validate every step in the witness used to construct dense supervision.

## High-Level Takeaways

- SSOPD turns a successful sampled trajectory into privileged teacher context at prefixes the student actually visits before failure.
- The teacher remains frozen while LoRA updates the student; extra information comes from context rather than stronger teacher weights.
- Homogeneous groups disable the auxiliary term, so the method depends on already sampling both success and failure.
- The strongest reported result is at 8B; smaller gains, checkpoint-selected ablations, and missing compute-matched uncertainty limit broader claims.
