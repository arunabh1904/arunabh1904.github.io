---
title: 'PCGrad: Gradient Surgery for Multi-Task Learning'
date: '2020-01-19T05:00:00.000Z'
section: paper-shorts
postSlug: pcgrad-gradient-surgery-for-multi-task-learning
legacyPath: /paper shorts/2020/01/19/pcgrad-gradient-surgery-for-multi-task-learning.html
tags: [Other]
field: 'Training Systems & Reliability'
summary: '2020 – PCGrad: project away pairwise conflicting task gradients'
---
## 2020 – PCGrad

**arXiv:** [2001.06782](https://arxiv.org/abs/2001.06782)<br>
**Code:** [tianheyu927/PCGrad](https://github.com/tianheyu927/PCGrad)

## Summary

> PCGrad changes the direction of conflicting task gradients before an optimizer combines them. It removes a component only when its dot product with another task's gradient is negative, preserving positive interactions. The paper reports gains in supervised multitask learning and faster learning of simulated robot skills, but the benefits differ across settings: supervised results suggest regularization, whereas the reinforcement-learning results show stronger training progress. Projection requires separate task gradients and additional training time; it neither chooses task priorities nor guarantees that every task improves.

## Core Insights

### A negative dot product is part of the problem

Two tasks can disagree about how shared parameters should change. If their gradients have a negative dot product, taking a descent step for one locally increases the other's loss. Yet conflict alone does not make the average gradient wrong: averaging still gives the gradient of the summed objective. The paper's proposed failure mechanism combines conflict with unequal gradient magnitudes and high local curvature.

The interaction matters. A large gradient can dominate the sum, while curvature makes its apparent benefit overoptimistic and the smaller task's damage worse than a first-order picture suggests. PCGrad tries to break the conflict component of that combination. It does not estimate a Hessian or test all three conditions at every update; the practical trigger is simply a negative dot product.

This changes a different quantity from [GradNorm](/paper%20shorts/2017/11/07/gradnorm-adaptive-loss-balancing.html), which uses loss weights to adjust gradient magnitudes according to training rates. Scaling a vector cannot turn it away from an opposing direction. PCGrad removes the opposing component itself, while retaining the opportunity for tasks with aligned gradients to reinforce each other.

### Projection changes a gradient without changing the network

The algorithm first computes each original task gradient $g_i$ and makes a working copy $h_i=g_i$. For each other task, visited in random order, it applies

$$
h_i\leftarrow h_i-\frac{h_i^\top g_j}{\|g_j\|_2^2}g_j
\qquad\text{if }h_i^\top g_j<0.
$$

The evolving working gradient is compared with the other task's original gradient. After these pairwise operations, the working gradients are summed and passed to an optimizer such as SGD or Adam. Task-specific parameters remain outside this shared-parameter correction. No inference-time module or extra prediction head is added.

For a concrete two-dimensional illustration, take $g_1=(1,1)$ and $g_2=(-1,0)$. Their dot product is negative. Removing the first gradient's component along the second gives $h_1=(0,1)$: its rightward component disappears, leaving motion perpendicular to $g_2$. Applying the symmetric operation to the second gradient gives $h_2=(-0.5,0.5)$. The resulting combined direction differs in both angle and length from the original sum.

The source figure traces the same operation. Panel (b) modifies the blue gradient while panel (c) modifies the red one. Panel (d) is just as important: an aligned pair is left alone. Orthogonalizing every pair would also remove useful shared learning signals.

![PCGrad source Figure 2: conflicting gradients, each projection, and an unchanged nonconflicting pair](/assets/images/pcgrad-gradient-surgery-for-multi-task-learning-paper-figure.webp)
*Fig 1: Panels (b) and (c) remove each gradient's conflicting component relative to the other. Panel (d) preserves a nonconflicting pair, allowing constructive transfer rather than forcing all task gradients to be orthogonal. | source: [PCGrad, Figure 2](https://arxiv.org/abs/2001.06782)*

With many tasks, sequential projections depend on order. A later projection can change an earlier relationship, so this is not a simultaneous solution to every pairwise constraint. Random shuffling makes the treatment symmetric in expectation. The MT50 appendix ablation performs worse with a fixed task order, supporting that implementation choice.

### Direction and magnitude both matter in the robot experiments

The reinforcement-learning experiments apply PCGrad to both actor and critic gradients in soft actor-critic. MT10 and MT50 contain ten and fifty simulated Meta-World manipulation tasks. The models receive task identifiers, and training samples equal amounts from per-task replay buffers. All compared methods use the paper's per-task entropy-temperature adjustment, so that addition should not be attributed solely to PCGrad.

The first two panels below report success against environment interactions. PCGrad learns substantially faster than independent agents and the shared SAC baselines. The paper reports approximately 70% success on MT50; independent agents eventually reach a similar level but require about 15 million more samples in its comparison. These are sample-efficiency results, not proof of lower wall-clock cost.

The rightmost panel asks what caused the gain. One ablation keeps only the corrected direction while restoring the original magnitude; another keeps only the corrected magnitude. Both underperform full PCGrad, and the magnitude-only variant is much weaker. The evidence supports the combined projection operation rather than treating PCGrad as merely a different scalar weighting rule.

![PCGrad source Figure 3: MT10 and MT50 learning curves and direction-versus-magnitude ablations](/assets/images/pcgrad-gradient-surgery-for-multi-task-learning-source-figure-3.webp)
*Fig 2: The left panels measure success against environment samples. The right panel separates direction and magnitude changes; retaining both performs better than either ablation in this MT10 experiment. | source: [PCGrad, Figure 3](https://arxiv.org/abs/2001.06782)*

### Better joint results can still include a regression

On NYUv2, adding PCGrad to Cross-Stitch changes semantic mIoU from 15.69 to 18.14 and absolute depth error from 0.6277 to 0.5805. The appendix also examines training and validation curves: training converges at a similar rate, while validation improves in two of three tasks. The authors interpret this supervised result as a possible regularization effect, distinct from the pronounced training improvement in reinforcement learning.

Cityscapes gives a useful counterexample to an all-task improvement claim. In the equal-weight MTAN comparison, PCGrad improves three of four reported metrics but worsens absolute depth error:

| Equal-weight model | Semantic mIoU, ↑ | Pixel accuracy, ↑ | Absolute depth error, ↓ | Relative depth error, ↓ |
| --- | --- | --- | --- | --- |
| MTAN | 53.04 | 91.11 | 0.0144 | 33.63 |
| MTAN + PCGrad | 53.59 | 91.45 | 0.0171 | 31.34 |

The two depth metrics move in different directions. Pairwise gradient surgery therefore does not supply a policy for deciding which metric matters most, even when the overall experiment is favorable. Such a policy has to come from the application and evaluation criteria.

### The guarantee and the cost are narrower than the intuition

Even the two-task convex convergence result allows a nonoptimal stopping case: exactly opposite gradients can both project to zero. Its assumptions include differentiability, a Lipschitz gradient for the summed loss, and a bounded step size. The separate one-step improvement analysis adds conditions on conflict, magnitude imbalance, curvature, and step size. Neither result is a blanket guarantee for a many-task nonconvex network trained with Adam.

Computing separate task gradients also has a cost. Appendix J reports supervised runs completing within twelve hours with PCGrad versus eight without it on a TITAN RTX, with maximum observed GPU memory of 10 GB versus 6 GB. On MT50, the reported training runs take five days with PCGrad versus three days for vanilla SAC. These are measurements from the paper's configurations, not universal overhead factors. They explain why improved sample efficiency and unchanged inference architecture do not imply cheaper training.

## High-Level Takeaways

- PCGrad removes negative gradient components while preserving aligned pairs. It changes update geometry rather than choosing new task losses or network branches.
- Each working gradient is projected against original task gradients in randomized order; sequential pairwise repair is not a global guarantee of compatibility.
- The MT10 ablation needs both direction and magnitude changes to recover the full gain. Scaling alone misses a substantial part of the reported benefit.
- Supervised validation gains can coexist with similar training convergence and individual metric regressions, as the Cityscapes depth comparison shows.
- Separate task gradients increase training cost, and opposite gradients can produce a zero update even in the simplified theory. Gradient surgery does not decide which task should take priority.
