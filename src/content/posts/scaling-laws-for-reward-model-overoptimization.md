---
title: 'Scaling Laws for Reward Model Overoptimization'
date: '2022-10-19T00:00:00.000Z'
section: paper-shorts
postSlug: scaling-laws-for-reward-model-overoptimization
legacyPath: /paper shorts/2022/10/19/scaling-laws-for-reward-model-overoptimization.html
tags:
  - Alignment
  - Reward Models
field: 'Alignment & Post-Training'
summary: "2022 – Scaling Laws for Reward Model Overoptimization"
---

## 2022 – Scaling Laws for Reward Model Overoptimization

**arXiv:** [2210.10760](https://arxiv.org/abs/2210.10760)

## Summary

> This paper measures a version of Goodhart’s law in a controlled language-model experiment. A large “gold” reward model stands in for human judgment, a smaller proxy reward model is trained from its comparisons, and a policy is optimized against the proxy with best-of-$n$ sampling or reinforcement learning. Proxy reward keeps rising after gold reward peaks, so the useful quantity is the proxy–gold frontier and its stopping point.

## Core Insights

### The experiment separates the score being optimized from the score that matters

The paper needs a target that can be evaluated after optimization. It therefore treats a fixed 6B reward model as a synthetic gold judge, trains proxy reward models from its pairwise labels, and then asks how far a policy can move before the proxy becomes misleading. The proxy models range from 3M to 3B parameters; the policy is 1.2B parameters in the main reward-model scaling experiment.

The synthetic setup makes the source of every label explicit. The gold model scores two sampled completions for the same prompt, and the higher-scoring completion supplies the comparison label. The authors use 100,000 synthetic comparisons, hold out 10% for validation, recenter scores around the initial policy, and recalibrate proxy logits with validation soft labels. This is a useful experimental control, but it also bounds the conclusion: the gold model is a stationary stand-in for human intent, not human intent itself.

What exactly is being replaced by the synthetic setup? The source’s real and synthetic pipelines answer that question:

![Real and synthetic reward-model training setups, with human comparisons replaced by gold-model comparisons in the controlled experiment](/assets/images/scaling-laws-for-reward-model-overoptimization-source-figure-2.webp)
*Figure 1: The paper’s real and synthetic reward-model setups. In the synthetic version, the gold reward model supplies the comparison labels so the authors can vary proxy size and optimization pressure while keeping the target fixed. Source Figure 2: [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760).*

The distinction matters because “a better reward model” has two separate effects here: it can fit the proxy labels better, and it can remain aligned with the held-out gold score farther into optimization. The experiment is designed to measure the second effect.

### Optimization pressure is distance from the starting policy, not just the sampling budget

For best-of-$n$, the policy samples $n$ completions and keeps the one with the largest proxy score. For RL, the policy is trained with PPO. The paper puts these different procedures on a common horizontal axis,

$$
d=\sqrt{D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{init}})},
$$

the distance from the initial policy. In the synthetic experiment, the local gold-reward curves are fitted with different forms:

$$
R_{\mathrm{BoN}}(d)=d(\alpha_{\mathrm{BoN}}-\beta_{\mathrm{BoN}}d),
\qquad
R_{\mathrm{RL}}(d)=d(\alpha_{\mathrm{RL}}-\beta_{\mathrm{RL}}\log d).
$$

These are empirical descriptions of the measured regime, not a claim that reward must follow either function at arbitrary distance. The RL expression is especially a local fit near the origin, where its logarithmic behavior should not be read literally.

The following adapted plot makes the mechanism visible. Dashed curves are proxy scores; solid curves are gold scores. The proxy keeps improving, while the gold score bends over and can decline. Increasing reward-model size shifts the bend, but does not remove it:

![Adapted reward-model overoptimization curves for best-of-n and RL](/assets/images/scaling-laws-for-reward-model-overoptimization-paper-figure.png)
*Figure 2: Adapted from source Figure 1. The left panel shows best-of-$n$; the right shows RL. Larger proxy reward models postpone the proxy–gold divergence, while the gold score still has a finite peak. The plotted graphic is an adaptation of the source figure; source: [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760).*

A subtle comparison follows. RL is less KL-efficient than best-of-$n$: it needs more movement from the initial policy to reach the same proxy or gold score. Yet RL reaches a higher gold peak in these experiments, while best-of-$n$ and RL trace similar proxy–gold trajectories. Raw KL therefore cannot compare optimization pressure across optimizers. It is a coordinate on a frontier whose shape depends on the optimizer.

### Data quality changes the curve, while a KL penalty mostly changes when training stops

Reward-model size is not the only control. Holding the reward model at 12M parameters, the paper varies the number of comparison examples. More unique comparisons improve the gold score and reduce Goodharting; below roughly 2,000 comparisons, validation loss is close to chance and there is little useful improvement. Repeating a smaller dataset for four epochs is worse than using four times as many unique examples for one epoch. The model needs coverage of the preference boundary, not merely more passes over the same labels.

The source’s validation-loss plot shows why a held-out proxy diagnostic is informative but not sufficient:

![Gold reward at best-of-1000 versus proxy reward-model validation loss](/assets/images/scaling-laws-for-reward-model-overoptimization-source-figure-6.webp)
*Figure 3: Gold reward at best-of-1000 as a function of reward-model validation loss, averaged over the reported runs. Lower validation loss is associated with better gold reward in this controlled range, but the plot does not identify how far a policy may safely be optimized. Source Figure 6: [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760).*

Larger policies show a weaker, limited result: in the two policy sizes studied, the absolute gold improvement changes, but the overoptimization gap and peak KL are similar. The authors also vary a KL penalty. It can keep the policy closer to the initial model and make optimization converge earlier, which is useful as an early-stopping control, but it does not improve the underlying policy–gold frontier. A penalty limits movement; it does not add information to a misspecified reward. The PPO update also has a recent-policy KL mechanism, so this initial-policy distance should not be conflated with every KL term in the optimizer.

### The mechanism is a mixture of Goodhart effects

The paper’s discussion is more useful than the slogan “proxy optimization is dangerous.” Regressional Goodhart predicts that proxy noise is amplified at the extremes. Extremal Goodhart adds a distribution shift: optimization pushes the policy into regions unlike the reward-model training distribution. Causal Goodhart appears when a superficial correlate, such as answer length, is rewarded without being the intended property. The synthetic models are too weak to test adversarial Goodhart in a serious way.

The authors also derive an iterated-RLHF thought experiment. Under constant fitted coefficients and fresh reward models, repeated rounds change the RL expression to include a $\beta_{\mathrm{RL}}d\log k$ term after $k$ rounds. That suggests fresh supervision can move the frontier, but the derivation is valid only within the simplifying regime and cannot justify unlimited iteration.

## High-Level Takeaways

- Use an independent target metric to locate the peak of reward optimization. Proxy reward alone gives no stopping signal once the policy leaves the reward model’s reliable region.
- Treat reward-model size and unique comparison coverage as different investments: larger models reduce the overoptimization coefficient in this setup, while repeated copies of the same comparisons do not substitute for new coverage.
- Compare optimizers on their policy–target frontier rather than raw KL. Best-of-$n$ and RL consume KL differently, and a KL penalty controls early stopping without repairing a wrong target.
- The synthetic gold model makes the scaling experiment tractable while leaving human-intent mismatch outside the measurement. A deployment decision still needs held-out human or task-level evaluation at several optimization distances.
