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

**Paper:** [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) · [Code](https://github.com/deepseek-ai/DeepSeek-Math)

## Summary

> DeepSeekMath replaces PPO's learned value baseline with rewards normalized across several answers to the same question. This removes a trainable critic, while retaining policy-ratio clipping, a reference model, and reward evaluation. Its 7B instruction model improves from 46.8% to 51.7% on MATH after reinforcement learning. The paper's more revealing result is that majority-vote accuracy improves while high-budget pass@K changes little: this run makes correct answers more likely without clearly expanding the set of questions for which a correct sampled answer can be found.

## Core Insights

### Let each question supply its own comparison group

A reward of one can mean different things for an easy arithmetic question and a difficult proof. GRPO samples several answers to the same question and compares their rewards locally. For outcome rewards $\{r_1,\ldots,r_G\}$, its completion-level advantage is

$$
\hat A_i=\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)}.
$$

Every token in completion $i$ receives that same advantage under outcome supervision. With illustrative binary rewards $[1,1,0,0]$, the successful answers lie above the mean and the unsuccessful ones below it. The question has created its own baseline without asking a value network to estimate expected future reward at every token.

The source diagram makes the resource trade-off visible. PPO's value model and generalized advantage estimator disappear in the lower GRPO path, replaced by several sampled outputs and group computation. The reward and reference models remain. Removing the critic saves its parameter, optimizer, and activation costs; it does not remove rollout generation or reward scoring.

![DeepSeekMath source Figure 4: PPO's learned value baseline versus GRPO's within-question group baseline](/assets/images/deepseekmath-source-figure-4-comparison.png)
*Fig 1: Follow the lower path from several answers to their rewards and group-relative advantages. The value model disappears, but the reward model, reference model, and sampling workload remain. | source: [DeepSeekMath, Figure 4](https://arxiv.org/abs/2402.03300)*

Equal rewards leave no outcome contrast to learn from. The written normalization is also undefined when the standard deviation is zero, so an implementation must handle that case explicitly. With a numerical safeguard, the reward-driven advantage becomes zero; the separate KL regularizer can still supply a gradient. “No useful reward contrast” is more precise than saying the entire training objective necessarily stops updating.

### The old policy and the reference policy have different jobs

The old policy supplies the probability denominator for PPO-style ratio clipping on recently sampled answers. The reference policy supplies a KL regularizer that discourages excessive drift. DeepSeekMath adds that regularizer directly to the objective rather than mixing it into rewards before group normalization.

Those two anchors operate on different time scales. The iterative algorithm refreshes the sampling policy within training, and resets the reference at the start of an outer RL iteration. Calling both a single permanently frozen baseline would hide that distinction. The group controls relative reward credit, clipping controls the sampled action-probability update, and the reference regularizes broader distributional movement.

The original experiment also uses learned reward models. Correctness judgments help construct their training data, but the main GRPO system is not simply a parameter-free exact-answer checker. In iterative RL, the reward model is updated on new policy outputs with 10% historical replay. Reward-model drift and exploitation remain relevant despite removing the critic.

### Outcome and process supervision assign different credit

Outcome supervision copies one normalized reward across the entire completion. An early useful step and a late arithmetic error therefore receive the same signed weight when they occur in the same answer. It is a coarse learning signal, even when the final-answer judgment is accurate.

The process-supervised variant scores reasoning-step endings. It normalizes those step rewards across the group's steps, then assigns each token the sum of normalized rewards at subsequent step endings. A token can consequently receive a different weight depending on what remains in its reasoning trajectory. Figure 5's experiments favor this finer process supervision over the outcome-only variant in the tested setting.

This is why “GRPO has one advantage per response” describes the outcome version, not the entire paper. The critic-free baseline and the granularity of reward supervision are separate choices.

### Pretraining data, RL data, and generated candidates are separate budgets

The mined DeepSeekMath corpus contains 120B math-related tokens. The 7B base model, initialized from DeepSeek-Coder, is trained for 500B tokens across a mixture: 56% that corpus, 4% AlgebraicStack, 10% arXiv, 20% GitHub code, and 10% natural-language Common Crawl. The harvested corpus size is not the total training-token count.

RL starts from the instruction-tuned model and uses roughly 144,000 GSM8K- and MATH-related questions. Section 4.2 samples 64 outputs per question, with maximum length 1,024 and one policy update after each exploration stage. That makes the trade concrete: a removed value network is exchanged for a substantial same-question sampling workload.

| Chain-of-thought evaluation | Instruction model | RL model |
| --- | ---: | ---: |
| GSM8K | 82.9% | 88.2% |
| MATH | 46.8% | 51.7% |
| MGSM-zh | 73.2% | 79.6% |
| CMATH | 84.6% | 88.8% |

These are before-and-after post-training results from the same instruction starting point. They support this RL recipe, including gains outside the two named RL benchmark families. They do not isolate the optimizer from reward modeling, generated data, and update choices, nor compare PPO and GRPO under a fully matched total compute budget.

### Better answer selection is different from a wider solution set

Figure 7 samples answers at temperature 0.7 and varies their number. Majority@K selects the most common answer; pass@K asks whether at least one sampled answer is correct. The latter is an oracle coverage measure unless a reliable selector can identify that correct candidate.

![DeepSeekMath source Figure 7: majority-vote and pass-at-K accuracy before and after reinforcement learning](/assets/images/deepseekmath-source-figure-7-sampling.png)
*Fig 2: RL raises majority-vote accuracy, while the high-budget pass-at-K curves change little or worsen. More probability reaches correct answers without a comparable expansion of sampled solution coverage. | source: [DeepSeekMath, Figure 7](https://arxiv.org/abs/2402.03300)*

Compare orange with purple for majority voting, and blue with green for pass@K. On GSM8K, RL improves the majority result while its large-K coverage is lower; on MATH, the large-K coverage curves are close. The authors interpret this as making existing correct responses more likely. It is evidence about these models, benchmarks, and sampling budgets, not a universal conclusion that reinforcement learning cannot learn new reasoning behavior.

## High-Level Takeaways

- GRPO trades a learned value baseline for same-question reward comparisons; reward scoring and multiple rollouts remain significant costs.
- Separate the old-policy clipping reference from the KL reference, and handle zero-variance reward groups explicitly.
- Outcome supervision assigns one weight per completion; process supervision supplies step-dependent credit within the same critic-free framework.
- The 120B-token mined corpus, 500B-token pretraining mixture, and 64-answer RL groups are distinct quantities.
- The sampling curves distinguish more reliable answer probability from broader solution coverage, which single-answer accuracy alone cannot show.
