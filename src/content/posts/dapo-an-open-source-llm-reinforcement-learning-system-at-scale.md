---
title: 'DAPO: An Open-Source LLM Reinforcement Learning System at Scale'
date: '2025-03-18T00:00:00.000Z'
section: paper-shorts
postSlug: dapo-an-open-source-llm-reinforcement-learning-system-at-scale
legacyPath: /paper shorts/2025/03/18/dapo-an-open-source-llm-reinforcement-learning-system-at-scale.html
tags: [Reinforcement Learning, Reasoning, Optimization]
field: 'Language Models'
summary: '2025 – DAPO: An Open-Source LLM Reinforcement Learning System at Scale'
---

## 2025 – DAPO: An Open-Source LLM Reinforcement Learning System at Scale

**Paper:** [Yu et al., arXiv:2503.14476v2](https://arxiv.org/abs/2503.14476v2) · [Project and released artifacts](https://dapo-sia.github.io/)

## Summary

> DAPO makes long-reasoning RL work through a coordinated set of training choices: asymmetric clipping, filtering uninformative answer groups, normalizing losses across tokens, and handling truncated responses carefully. On Qwen2.5-32B, the progressive recipe raises AIME 2024 accuracy from 30% to 50%. The practical contribution is identifying failure modes hidden by a short description such as “train with GRPO.”

## Core Insights

### Clipping and sampling determine whether useful gradients survive

[GRPO](/paper%20shorts/2024/02/05/deepseekmath-group-relative-policy-optimization-grpo.html) estimates advantages relative to other responses to the same prompt. DAPO keeps that group comparison but removes the KL penalty in its long-reasoning setting and separates the two clipping thresholds. Its lower threshold is 0.2 and upper threshold 0.28. The larger upper range leaves more incentive to increase probabilities of positively rewarded tokens before the clipped surrogate saturates. These are properties of the objective, not hard bounds on the updated policy's probabilities.

The second problem is uninformative groups. When every sampled answer is correct, or every answer is wrong, the group-relative advantage supplies no useful reward contrast. Dynamic sampling generates extra groups, discards these uniform-outcome groups, and fills the batch with groups that contain both outcomes. It spends additional rollout work to preserve a useful training signal and changes which prompts contribute gradients.

![DAPO dynamic sampling reaches comparable AIME accuracy in fewer training steps than its baseline.](/assets/images/dapo-source-figure-6.png)
*Fig 1: The purple dynamic-sampling run reaches the marked accuracy in fewer steps than the cyan baseline. The horizontal axis is training step; this plot alone does not establish equal-cost rollouts or a wall-clock speedup. | source: [Paper, Figure 6](https://arxiv.org/html/2503.14476v2#S4.F6)*

[Open the source figure at full size](/assets/images/dapo-source-figure-6.png).

### Loss normalization changes which responses carry weight

Averaging each response's token losses before averaging responses gives a long answer the same aggregate weight as a short one. DAPO instead sums token losses across the batch and divides by the total token count. Each token receives equal weight, so longer responses receive greater aggregate weight. This is a concrete optimization change with consequences for how rewarded and unrewarded long trajectories affect learning.

Truncation needs separate treatment. An answer cut off at the generation limit may be unfinished rather than incorrect. DAPO first masks the loss for overlong samples, then adds a soft length penalty that begins at 16,384 tokens and increases across a 4,096-token interval to the 20,480-token maximum. The aim is to avoid misleading correctness penalties while discouraging responses that consume the whole budget.

### The headline result belongs to the complete recipe

The experiment uses Qwen2.5-32B, approximately 17,000 competition-math prompts, and verifiable final answers. The data construction converts answers into integer targets where possible, reducing answer-parser ambiguity. That preprocessing is part of the method; its reliability does not automatically transfer to open-ended tasks.

The progressive ablation moves from 30% for naive GRPO to 36% with overlong filtering, 38% with the larger upper clipping threshold, 41% with soft length penalties, 42% with token-level normalization, and 50% with dynamic sampling. These are cumulative additions, not a factorial experiment that isolates every interaction. Evaluation repeats AIME 32 times and averages correctness at temperature 1.0 and top-p 0.7; avg@32 is not best-of-32 selection.

The comparison with [DeepSeek-R1-Zero-Qwen-32B](/paper%20shorts/2025/01/22/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning.html) reports 50% versus 47% with roughly half as many training steps. Step count is narrower than total compute because sampling volume, response lengths, and systems behavior can differ. The authors report favorable training-time behavior for their dynamic sampler, but the displayed curve directly establishes progress per step.

Training uses batches of 512 prompts with 16 responses each and a constant learning rate of $10^{-6}$ after a 20-rollout-step warmup. Reward alone is an incomplete diagnostic: the paper also tracks entropy, response length, and validation accuracy. Entropy collapse can remove exploration, while excessive entropy can produce incoherent output. Both failure modes can hide behind an improving training objective.

## High-Level Takeaways

- Group-relative RL depends on preserving reward variation within sampled groups.
- Token and response normalization optimize differently when answer lengths vary.
- A truncated answer should not silently receive the same interpretation as a completed wrong answer.
- DAPO's result supports a complete, reproducible recipe; step efficiency is not a matched-compute guarantee.
