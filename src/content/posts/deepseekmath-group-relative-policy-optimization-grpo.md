---
title: 'DeepSeekMath: Group Relative Policy Optimization (GRPO)'
date: '2024-02-05T00:00:00.000Z'
section: paper-shorts
postSlug: deepseekmath-group-relative-policy-optimization-grpo
legacyPath: /paper shorts/2024/02/05/deepseekmath-group-relative-policy-optimization-grpo.html
tags:
  - Reinforcement Learning
  - Reasoning
field: 'Reinforcement Learning'
topics:
  - language-systems
  - learning
summary: '2024 – DeepSeekMath: Group Relative Policy Optimization (GRPO)'
---

## 2024 – DeepSeekMath: Group Relative Policy Optimization (GRPO)

**arXiv:** [2402.03300](https://arxiv.org/abs/2402.03300)

**Code:** [deepseek-ai/DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math)

## Summary

> DeepSeekMath introduces Group Relative Policy Optimization as a cheaper alternative to PPO for language-model reasoning. For each prompt, the policy samples a group of completions and scores them with an outcome or process reward. Their within-group mean and standard deviation replace PPO's learned value baseline. Removing the critic saves memory and training compute, but it also makes learning depend on reward variation among samples of the same prompt.

## Core Insights

![DeepSeekMath contrasts PPO's actor, reference, reward, and value models with GRPO's critic-free group baseline](/assets/images/deepseekmath-grpo-paper-figure.png)
*PPO estimates advantages with a learned value model. GRPO samples several answers for one question and computes advantages relative to that group, eliminating the value network while retaining clipped policy ratios and a reference-policy KL penalty. source: [DeepSeekMath](https://arxiv.org/abs/2402.03300)*

![Figure 3 from DeepSeekMath: Group Relative Policy Optimization (GRPO)](/assets/images/deepseekmath-group-relative-policy-optimization-grpo-source-figure-3.webp)
*Figure 3 Benchmark curves of DeepSeek-LLM 1.3B trained on different mathematical corpora. source: [DeepSeekMath: Group Relative Policy Optimization (GRPO)](https://arxiv.org/abs/2402.03300)*

![Figure 7 from DeepSeekMath: Group Relative Policy Optimization (GRPO)](/assets/images/deepseekmath-group-relative-policy-optimization-grpo-source-figure-7.webp)
*Figure 7 The Maj@K and Pass@K of SFT and RL DeepSeekMath 7B on GSM8K and MATH (temperature ). It was noted that RL enhances Maj@K but not Pass@K. source: [DeepSeekMath: Group Relative Policy Optimization (GRPO)](https://arxiv.org/abs/2402.03300)*


For rewards $\{r_1,\ldots,r_G\}$ from $G$ completions of prompt $q$, outcome-supervised GRPO uses

$$
\hat A_i =
\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)}.
$$

The same completion-level advantage weights every token in completion $i$. The policy objective then applies PPO-style ratio clipping and a direct KL penalty to a frozen reference. The method is “group relative” because the question supplies its own local baseline: a reward of one is positive evidence only when other samples for that question score lower.

DeepSeekMath is a full training recipe, not an isolated optimizer ablation. The 7B base model is continued-pretrained on 120 billion math-related tokens, instruction-tuned, and then optimized with GRPO. On the reported chain-of-thought setting, RL raises GSM8K from 82.9% to 88.2% and MATH from 46.8% to 51.7%; multilingual MGSM-zh rises from 73.2% to 79.6%. These gains establish that the pipeline works, not that GRPO alone caused every point.

| Property | PPO | GRPO |
| --- | --- | --- |
| Baseline | Learned value function | Mean reward of same-prompt samples |
| Extra trainable model | Critic/value model | None |
| Rollouts | On-policy trajectories | On-policy groups per prompt |
| Credit under outcome reward | Per-token advantage can use value estimates | One normalized outcome copied to all tokens |
| Structural failure | Critic error and actor-critic instability | All-equal group rewards give zero advantage |

That last row is the central limitation. If every answer is correct or every answer is wrong, the group has no variance and supplies no policy gradient. Prompt scheduling, group size, verifier quality, and policy competence therefore become part of the optimizer. GRPO is inexpensive only when generating enough diverse completions and checking them is also inexpensive.

## High-Level Takeaways

- GRPO informs whether a learned critic is worth its cost when rewards can be verified at the completion level. It is a natural fit for math and code because many samples can be generated from one prompt and an exact checker can cheaply separate them. It is less natural when rewards are noisy, resets are costly, or intermediate actions need distinct credit.
- The missing comparison is PPO and GRPO with identical models, prompts, samples, rewards, KL budgets, and total inference compute. The DeepSeekMath system changes data and post-training stages together. At ten times rollout length, copying one terminal advantage to every token increases variance and invites length-dependent artifacts. Later reasoning systems repair sampling and clipping, but do not remove the fundamental dependence on informative within-prompt contrasts.
- [PPO](/paper%20shorts/2017/07/01/proximal-policy-optimization-ppo.html) supplies the clipped policy update that GRPO reuses. [On-policy distillation](/paper%20shorts/2023/06/23/on-policy-distillation-language-models-gkd.html) offers a different route: sample the student's own states, then obtain dense token-level targets from a teacher rather than a single outcome.
- GRPO replaces PPO's value model with a within-prompt reward baseline and became a core optimizer for verifiable-reward reasoning.
- Its advantage disappears on homogeneous groups, terminal rewards give coarse token credit, and the source paper does not isolate optimizer gains from the full DeepSeekMath recipe.
- GRPO makes online reasoning RL cheaper by trading a learned critic for more same-prompt samples—and makes sample diversity part of the learning algorithm.
