---
title: 'Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study in Mathematical Reasoning'
date: '2025-09-29T00:00:00.000Z'
section: paper-shorts
postSlug: scaling-behaviors-of-llm-reinforcement-learning-post-training-an-empirical-study-in-mathematical-reasoning
legacyPath: /paper shorts/2025/09/29/scaling-behaviors-of-llm-reinforcement-learning-post-training-an-empirical-study-in-mathematical-reasoning.html
tags:
  - Language Models
  - Reinforcement Learning
  - Scaling Laws
  - Post-Training
field: 'Alignment & Post-Training'
summary: '2025 – Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study in Mathematical Reasoning'
---

## 2025 – Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study in Mathematical Reasoning

**Paper:** [Tan et al., arXiv:2509.25300v4](https://arxiv.org/abs/2509.25300v4)

## Summary

> Across 63 Qwen2.5 models from 0.5B to 72B, GRPO post-training on mathematical reasoning follows an empirical log-linear loss curve in both compute and unique data. Larger models learn more efficiently, but the fitted efficiency term saturates; at fixed compute, a 32B model can initially beat 72B because it can take more updates. Reusing a fixed dataset up to 25 times shows little degradation in the reported regime, while reuse at 100 times overfits. These are predictive fits for binary-reward math post-training, not universal scaling laws for capability, other domains, or other RL algorithms.

## Core Insights

The paper asks what changes when the scaling-law object is reinforcement learning rather than next-token pretraining. The resource axes are model size $N$, cumulative training compute $C$, and unique training data $D$. The measured outcome is test loss $L=1-\text{Pass@1}$ on a held-out set, so “loss” here is a transformed success rate, not cross-entropy. That choice makes the curves easy to interpret for verifiable math, but it also makes every coefficient dependent on the benchmark and answer checker.

### A fitted efficiency curve predicts held-out model size and late training

The authors fine-tune the Qwen2.5 dense family at 0.5B, 1.5B, 3B, 7B, 14B, 32B, and 72B parameters with GRPO. Training uses the mathematics subset of guru-RL-92k, ordered by difficulty using Qwen2.5-7B-Instruct. Rewards are binary answer-correctness outcomes. The common recipe uses learning rate $10^{-6}$, batch 512, reference-KL weight 0.001, clipping parameter 0.2, and input/output limits of 2,048/4,096 tokens. Training temperature is 1.0 and evaluation temperature is 0.7; configurations are repeated three times. The primary scaling fit is

$$
\log L(N,X)=-k(N)\log X+E(N), \qquad
k(N)=\frac{K_{\max}}{1+N_0/N},
$$

where $X$ is either compute or unique data. The first term describes the within-model learning slope; the second says that the slope itself increases with model size but approaches a ceiling.

The prediction protocol matters. Inter-model fits use 0.5B–32B models and forecast the held-out 72B model. Intra-model fits use early training points to extrapolate the rest of one trajectory. Across Qwen2.5 base and instruct runs, the reported fits have $R^2>0.99$; the paper also repeats the functional form on Llama 3 Instruct models from 1B to 70B. This is evidence that the chosen parametric form predicts the measured curves in the tested range. It is not evidence that the same coefficients transfer to a new reward, prompt distribution, or model family.

The plot below is deliberately simple: it is the experiment's data-allocation control, not a performance curve. Holding total training volume fixed while changing the reuse factor makes it possible to ask whether a consumed sample must be unique to remain useful.

![The source schematic keeps total post-training data fixed while varying how often the unique subset is reused.](/assets/images/rl-posttraining-data-reuse-source-figure-7.png)
*Fig 1: Each row preserves the same total data volume while changing the reuse factor $\tau$. The unique subset shrinks as repetition rises, so the experiment isolates reuse from total optimization steps while preserving curriculum order. | source: [Scaling Behaviors of LLM RL Post-Training, Figure 7](https://arxiv.org/html/2509.25300v4#S3.F7)*

[Open the figure at full size](/assets/images/rl-posttraining-data-reuse-source-figure-7.png)

### Compute optimality is a training-budget statement

At a fixed compute budget, larger models generally have lower fitted loss, but the frontier is not monotone in the model-size headline at every point. The paper reports a crossover where 32B initially beats 72B because the smaller model can complete more optimization steps with the same FLOPs. This is the post-training analogue of an allocation trade-off: a larger model buys a more efficient update, while a smaller model buys more updates.

The fitted $k(N)$ term makes the trade-off explicit. For Qwen2.5 base inter-model fits, the asymptotic efficiency is $K_{\max}=0.152$ with $N_0=17.4$B; for instruct models it is $0.144$ with $N_0=28.3$B. These are the compute-dimension inter-model coefficients; the data fits have separately estimated values. They are model-family and protocol parameters, not portable constants. The authors' Llama 3 validation has the same functional form but lower absolute held-out accuracy, with Llama-70B near 50% versus Qwen-72B near 59% in the reported comparison.

Appendix C estimates cumulative compute as $6NT$ over processed training tokens using non-embedding parameters. It does not separately itemize rollout generation, reference-policy scoring, or verification costs, and the larger starting models already embody different pretraining investments. The plotted frontier therefore does not establish minimum total lifecycle cost or wall-clock time.

### Data reuse changes the uniqueness bill, not the total update bill

The data-constrained experiment fixes $D_{\text{total}}$ and varies $\tau$ by sampling a subset of the math corpus, ordering it by the same difficulty curriculum, and cycling it until the total number of examples is unchanged. For $\tau\leq25$, final test loss is reported as insensitive to reuse. At $\tau=100$, the paper observes clear overfitting. The practical claim is therefore narrower than “duplicates are free”: moderate repetition can be a useful response to scarce high-quality reasoning data when the total optimization budget remains fixed.

The evaluation suite includes 500 held-out in-domain math problems and 2,453 additional examples across AIME2024, AMC2023, GSM8K, MATH-500, HumanEval, Zebra Puzzle, and SuperGPQA. RL improves unseen math tasks of varying difficulty, while HumanEval and SuperGPQA remain mostly flat and Zebra Puzzle can degrade for larger models. The result is a clean boundary on transfer: the learned update generalizes within the mathematical family more readily than it transfers to code, science, or a different logic format.

This paper extends the allocation question in [Training Compute-Optimal Large Language Models](/paper%20shorts/2022/03/29/training-compute-optimal-large-language-models.html) from pretraining loss to RL post-training outcomes. The comparison is useful only after fixing the metric: Chinchilla allocates pretraining FLOPs between parameters and tokens, while Tan et al. allocate GRPO updates over a verifier-defined math reward. [How to Read Scaling Laws for Language Models](/blog/2026/08/19/how-to-read-scaling-laws-for-language-models.html) gives the broader map and explains why a fit must be read together with its held-fixed resources.

The main unresolved object is capability transfer. The study uses one actor-only RL algorithm, dense Qwen2.5 and Llama 3 families, a mathematical curriculum, and a binary evaluator. Its $R^2$ values show that the chosen law describes these trajectories; they do not establish a domain-independent optimum. A stronger follow-up would fit the same surface on several reward types and domains, forecast held-out model sizes under matched token and rollout budgets, and report held-out forecast error and uncertainty on the target accuracy, alongside the goodness of fit.

## High-Level Takeaways

- RL post-training has a measurable compute-and-data response surface, but the paper's loss is one-minus-pass@1 on a specific math evaluation suite.
- Larger models have higher fitted learning efficiency, while $k(N)$ saturates; a smaller model can still win early at fixed FLOPs because it receives more updates.
- Reusing high-quality math data up to $\tau=25$ is nearly neutral in the reported regime, whereas $\tau=100$ overfits; uniqueness and total optimization volume are separate variables.
- Positive transfer is concentrated within mathematics. Flat code and science curves, plus Zebra Puzzle regressions, rule out a blanket claim that math RL creates general reasoning capability.
- The next falsification test is cross-domain, multi-algorithm forecasting under matched rollout, verifier, and compute costs—not another fit on the same binary reward.
