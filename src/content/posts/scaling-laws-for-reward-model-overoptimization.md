---
title: 'Scaling Laws for Reward Model Overoptimization'
date: '2022-10-19T09:00:00.000Z'
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

This paper measures Goodhart's law instead of citing it. A large “gold” reward model stands in for human judgment, a smaller proxy reward model is trained from its labels, and a policy is pushed against the proxy using reinforcement learning or best-of-$n$ sampling. Proxy reward keeps rising after gold reward peaks.

## Paper Insights

The paper parameterizes optimization pressure by distance from the initial policy, using $d=\sqrt{D_{KL}(\pi\|\pi_{init})}$. The fitted gold-reward curves differ by optimizer: best-of-$n$ is well described by $d(\alpha-\beta d)$, while RL follows $d(\alpha-\beta\log d)$ in the synthetic setup. Larger reward models and more reward data change the coefficients smoothly, which makes the location of the peak somewhat predictable.

The practical result is a stopping rule, not permission to optimize harder. Policy size has weak influence on the proxy–gold gap, and a KL penalty does not repair a misspecified reward; it controls movement, not truth. In a robot loop, the analogue is a critic score that rises while real success, safety, or human intervention rate flattens.

| Signal | What it reveals | Robot analogue |
| --- | --- | --- |
| Proxy reward | What training directly optimizes | Learned progress or success score |
| Gold reward | Held-out target quality | Real success, safety, and human judgment |
| KL distance | Optimization pressure | Policy drift from the deployed SFT controller |

## Decision Lens

This paper informs how far to push against a learned critic before collecting better labels or changing the reward. The training unit is a preference-labeled completion in the experiment, but the reusable object is the proxy-versus-ground-truth frontier. Scaling the proxy can postpone overoptimization; it cannot eliminate the fact that the proxy omits something.

The fitted laws hold in a synthetic language setting with a model acting as gold truth. At ten times the deployment diversity, critic blind spots and policy-induced states will change the reward distribution itself. A robot post-training program should therefore sweep optimization pressure and pre-register a real-world stopping metric. The central safety claim fails if a curve fitted on offline critic judgments cannot predict the point where held-out robot success begins to decline.

**Context:** This is the quantitative foundation for treating reward optimization as a budgeted intervention rather than an unlimited objective.

**Limits:** A synthetic gold model is cheaper and more stationary than human or physical evaluation.

**Takeaway:** Rising reward is evidence of optimization; only an independent measure can tell whether it is evidence of progress.
