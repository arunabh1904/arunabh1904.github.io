---
title: 'GradNorm: Adaptive Loss Balancing'
date: '2017-11-07T05:00:00.000Z'
section: paper-shorts
postSlug: gradnorm-adaptive-loss-balancing
legacyPath: /paper shorts/2017/11/07/gradnorm-adaptive-loss-balancing.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2017 – GradNorm: balance task training rates through shared-layer gradient norms'
---
## 2017 – GradNorm

**arXiv:** [1711.02257](https://arxiv.org/abs/1711.02257)

**Code:** [lucidrains/gradnorm-pytorch](https://github.com/lucidrains/gradnorm-pytorch)

### Method and reported result

GradNorm adapts task weights so gradient norms at a shared layer track each task's relative training rate. A task learning too slowly receives a larger target gradient; one learning quickly is reduced. The asymmetry parameter alpha controls how strongly the algorithm equalizes rates.

## Summary

> Unlike loss-scale normalization, GradNorm observes the effect of each task on shared parameters. That makes it relevant to unified perception trunks whose heads learn at different speeds.

## Core Insights

The method computes per-task gradient norms, compares them with targets derived from normalized loss descent, and updates the task weights. On NYUv2 the paper reports roughly 5% training overhead. Across its sweep, most values between 0 and 3 improve over equal weighting, with alpha near 1.5 best in the reported setup.

![GradNorm: Adaptive Loss Balancing source figure: Gradient Normalization.](/assets/images/gradnorm-adaptive-loss-balancing-paper-figure.webp)
*Gradient Normalization. source: [GradNorm: Adaptive Loss Balancing](https://arxiv.org/abs/1711.02257)*

![Figure 1 from GradNorm: Adaptive Loss Balancing](/assets/images/gradnorm-adaptive-loss-balancing-source-figure-1.webp)
*Figure 1 Gradient Normalization. Imbalanced gradient norms across tasks (left) result in suboptimal training within a multitask network. We implement GradNorm through computing a novel gradient loss (right) which tunes the loss weights to fix such imbalances in gradient norms. We illustrate here a simplified case where such balancing results in equalized gradient norms, but in general there may be tasks that require relatively high or low gradient magnitudes for optimal training (discussed further in Section 3 ). source: [GradNorm: Adaptive Loss Balancing](https://arxiv.org/abs/1711.02257)*

![Figure 8 from GradNorm: Adaptive Loss Balancing](/assets/images/gradnorm-adaptive-loss-balancing-source-figure-8.webp)
*Figure 8 Examples from the Multi-Task Facial Landmark (MTFL) dataset. source: [GradNorm: Adaptive Loss Balancing](https://arxiv.org/abs/1711.02257)*


| Quantity | Meaning | Operational concern |
| --- | --- | --- |
| Loss ratio | Relative task training rate | Sensitive to noisy or plateaued losses. |
| Shared-layer norm | Task pressure on the trunk | Depends on which layer is measured. |
| Alpha | Strength of rate equalization | Large values can destabilize weights. |
| Learned weight | Updated task coefficient | Needs bounds and monitoring. |

## High-Level Takeaways

- GradNorm is useful when tasks have comparable value but visibly different convergence rates. It does not inspect gradient direction, so two equal-norm tasks can still conflict. In a driving stack, apply it only after defining task-specific metrics and minimum acceptable behavior; training speed is not safety priority.
- Measure per-task transfer against single-task controls and inspect scenario slices. A global weight can hide a task that learns normally overall but fails under rare sensor conditions.
- Homoscedastic uncertainty handles scale; GradNorm handles relative rate; PCGrad handles conflicting direction.
- Shared trunks need gradient instrumentation: adaptive weights should respond to how tasks train the shared representation, not only to the numerical size of their losses.
