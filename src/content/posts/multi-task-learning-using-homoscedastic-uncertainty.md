---
title: 'Multi-Task Learning Using Homoscedastic Uncertainty'
date: '2017-05-19T04:00:00.000Z'
section: paper-shorts
postSlug: multi-task-learning-using-homoscedastic-uncertainty
legacyPath: /paper shorts/2017/05/19/multi-task-learning-using-homoscedastic-uncertainty.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2017 – Learn global task-loss weights through homoscedastic uncertainty'
---
## 2017 – Homoscedastic-Uncertainty Weighting

**arXiv:** [1705.07115](https://arxiv.org/abs/1705.07115)

### Method and reported result

Kendall, Gal, and Cipolla derive trainable weights for classification and regression losses from task-dependent homoscedastic uncertainty. Each task loss is scaled by an inverse variance and paired with a log-variance regularizer, preventing the model from driving a weight to zero without cost.

## Summary

The method solves one narrow but common multi-task problem: losses with different units and scales should not be added with arbitrary equal coefficients.

## Core Insights

The learned uncertainty is global for each task, not input-dependent. On the paper's Tiny Cityscapes experiment, joint semantic segmentation, instance regression, and depth reach 63.4 semantic IoU, 3.50 instance error, and 0.522 depth error; the semantic-only result is 59.4 IoU. The final relative task weighting is reported near 43:1:0.16, showing how far raw equal weighting can be from the learned scale.

| Mechanism | What it handles | What it does not handle |
| --- | --- | --- |
| Inverse-variance scaling | Loss units and average task noise | Per-example sensor quality. |
| Log regularizer | Prevents zero-weight collapse | Conflicting gradient direction. |
| Joint likelihood view | Principled classification/regression mix | Task utility or safety priority. |
| Global parameter | Low overhead | Scenario-dependent affinity. |

## High-Level Takeaways

- Use homoscedastic weighting after each task loss is internally normalized and monitored. Inspect both weights and shared-layer gradients: a well-scaled loss can still point against another task. Safety-critical tasks may also need explicit minimum influence rather than unconstrained likelihood optimization.
- The relevant ablation compares equal normalized weights, tuned constants, learned uncertainty, and separated heads under matched compute.
- GradNorm controls relative training rates through gradient magnitude; PCGrad modifies direction when gradients conflict.
- Learned uncertainty is a principled unit converter for multi-task losses, not a complete solution to task conflict or changing sensor reliability.
