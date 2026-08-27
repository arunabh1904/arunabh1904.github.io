---
title: 'PCGrad: Gradient Surgery for Multi-Task Learning'
date: '2020-01-19T05:00:00.000Z'
section: paper-shorts
postSlug: pcgrad-gradient-surgery-for-multi-task-learning
legacyPath: /paper shorts/2020/01/19/pcgrad-gradient-surgery-for-multi-task-learning.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2020 – PCGrad: project away pairwise conflicting task gradients'
---
## 2020 – PCGrad

**arXiv:** [2001.06782](https://arxiv.org/abs/2001.06782)

### Method and reported result

Projected Conflicting Gradient modifies a task gradient when its dot product with another task gradient is negative. The conflicting component is projected off the other gradient's normal direction, and the adjusted gradients are summed for the update. Nonconflicting pairs are left unchanged.

## Summary

> The method targets the “tragic triad” identified by the paper: conflicting directions, unequal gradient magnitudes, and high local curvature can make a joint update worse than either task update alone.

## Core Insights

PCGrad is optimizer-adjacent and model-agnostic; it requires per-task gradients but no new inference module. Randomizing the task order matters because sequential projections are not commutative. In the paper's NYUv2 results, adding PCGrad to Cross-Stitch changes mIoU from 15.69 to 18.14 and depth error from 0.6277 to 0.5805; fixed projection order performs worse in its ablation.

![PCGrad projection of conflicting task gradients onto one another's normal planes while leaving aligned gradients unchanged](/assets/images/pcgrad-gradient-surgery-for-multi-task-learning-paper-figure.webp)
*Fig 1: When two task gradients have a negative cosine, PCGrad removes each conflicting component; non-conflicting gradients pass through unchanged. | source: [PCGrad](https://arxiv.org/abs/2001.06782)*

![Figure 3 from PCGrad: Gradient Surgery for Multi-Task Learning](/assets/images/pcgrad-gradient-surgery-for-multi-task-learning-source-figure-3.webp)
*Fig 2: For the two plots on the left, we show learning curves on MT10 and MT50 respectively. PCGrad significantly outperforms the other methods in terms of both success rates and data efficiency. | source: [PCGrad: Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)*


| Condition | PCGrad response | Limit |
| --- | --- | --- |
| Negative cosine | Remove conflicting component | Pairwise repair may not optimize all tasks globally. |
| Positive cosine | Keep gradient | Magnitude imbalance remains. |
| Many tasks | Random pair order | Per-task backward cost grows. |
| Supervised gains | Improves several reported metrics | May partly act as regularization. |

## High-Level Takeaways

- PCGrad is useful after measurement shows persistent negative cosine in shared layers and task quality suffers. It should not be enabled solely because a model has multiple heads. Log conflict rates by layer and scenario, compare adapters or partial separation, and measure training memory and wall time.
- Projection also encodes no task priority. Safety or planning ownership still requires an explicit policy for which degradation is acceptable.
- Uncertainty weighting and GradNorm change scalar weights; PCGrad changes the direction of the shared update.
- Gradient surgery is a targeted response to measured directional conflict, not a substitute for deciding which tasks should share parameters.
