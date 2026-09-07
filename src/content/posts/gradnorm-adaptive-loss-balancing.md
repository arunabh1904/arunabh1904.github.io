---
title: 'GradNorm: Adaptive Loss Balancing'
date: '2017-11-07T05:00:00.000Z'
section: paper-shorts
postSlug: gradnorm-adaptive-loss-balancing
legacyPath: /paper shorts/2017/11/07/gradnorm-adaptive-loss-balancing.html
tags: [Other]
field: 'Training Systems & Reliability'
summary: '2017 – GradNorm: balance task training rates through shared-layer gradient norms'
---
## 2017 – GradNorm

**arXiv:** [1711.02257](https://arxiv.org/abs/1711.02257)<br>
**Community implementation:** [lucidrains/gradnorm-pytorch](https://github.com/lucidrains/gradnorm-pytorch)

## Summary

> GradNorm adjusts task weights by measuring what each loss does to a shared layer, then giving slower-learning tasks a larger target gradient. On NYUv2, this improves joint depth, normals, and segmentation or keypoint prediction with about 5% training overhead in the reported setup. The surprising result is that reducing a task's weight can improve its test performance even while its training loss rises. The boundary appears in the appendix: a task that has stopped learning can attract ever more weight without becoming learnable.

## Core Insights

### The shared layer reveals which task dominates

A multitask network adds weighted losses, $L=\sum_i w_iL_i$, but equal coefficients do not give tasks equal influence. Depth regression and semantic classification use different units and have different derivatives. A numerically large loss can dominate the shared representation before another head has learned anything useful. GradNorm measures that influence where the branches meet: the norm of each weighted task gradient at a chosen shared layer.

The first figure shows the feedback loop. The orange arrows update loss weights; the ordinary task gradients still train the network. Equal-sized green circles illustrate one balanced case, but the algorithm does not generally demand equal gradients. A task whose loss is falling slowly receives a larger target. Balancing means allocating influence according to progress, not making every head identical.

![GradNorm source Figure 1: task gradients and the auxiliary loss-weight update](/assets/images/gradnorm-adaptive-loss-balancing-source-figure-1.webp)
*Fig 1: Task gradients meet in the shared layers. GradNorm adds an auxiliary update to the loss weights, changing each task's influence without replacing the shared network or its task objectives. | source: [GradNorm, Figure 1](https://arxiv.org/abs/1711.02257)*

### Relative progress determines the target norm

For task $i$, the algorithm first divides its current loss by its initial loss. It then divides that ratio by the mean ratio across tasks:

$$
r_i(t)=\frac{L_i(t)/L_i(0)}{\frac{1}{T}\sum_j L_j(t)/L_j(0)}.
$$

This is a relative inverse training rate. For an illustrative pair of tasks whose loss ratios are 0.5 and 0.9, the second task has improved less. Its $r_i$ exceeds one, so GradNorm asks it to contribute a larger gradient than the current task average. The comparison uses fractional progress rather than comparing a depth error directly with a classification loss.

At shared weights $W$, define $G_i=\|\nabla_W(w_iL_i)\|_2$ and let $\bar G$ be the mean task norm. The auxiliary objective is

$$
L_{\mathrm{grad}}=\sum_i\left|G_i-\operatorname{stopgrad}\big(\bar G\,r_i^\alpha\big)\right|.
$$

Only the task weights are updated through this auxiliary objective; network parameters are updated through the ordinary weighted task loss. Holding the target constant prevents the weight update from reducing both sides together. Afterward, weights are renormalized to sum to the number of tasks, separating their relative allocation from the global learning rate. The paper measures gradients at the last shared layer to keep the extra work small.

The exponent $\alpha$ controls how strongly unequal progress changes the targets. At zero, the targets are equal. Larger values give slow tasks more influence. NYUv2 results peak near 1.5, while the more symmetric synthetic tasks use 0.12. These are experiment-specific settings: the appendix reports instability when large exponents push some weights too close to zero.

### Less training pressure can improve generalization

The main experiments use two NYUv2 variants: depth, normals, and segmentation on the standard small dataset; and depth, normals, and room keypoints on an expanded 90,000-image dataset split by scene. The models use VGG-style SegNet and a thinner ResNet-50 architecture, sharing parameters until the final layer. That design lets the experiments isolate task weighting within a largely shared network.

The VGG results on the expanded dataset expose the important distinction between fitting and generalization:

| Weighting | Depth RMS error, m | Keypoint error, % | Normals error |
| --- | --- | --- | --- |
| Equal weights | 0.658 | 8.39 | 0.155 |
| Uncertainty weighting | 0.649 | 8.00 | 0.158 |
| Static weights averaged from GradNorm | 0.638 | 7.69 | 0.137 |
| GradNorm, $\alpha=1.5$ | 0.629 | 7.73 | 0.139 |

Lower is better in every column; normals use the paper's cosine-based error. Dynamic weighting improves all three tasks over equal weights, but the derived static weights slightly outperform it on keypoints and normals. The evidence supports finding a better allocation, not a claim that continuous adaptation must always be superior.

Now compare the depth panel's large test curve with its training inset. GradNorm leaves a higher training error while achieving a lower test error. The depth weight falls below 0.1 in the reported run. This is why reducing a task's gradient need not mean abandoning it: shared features learned through the other tasks can remain useful, while less direct pressure limits overfitting. The paper interprets this pattern as regularization.

![GradNorm source Figure 3: test curves with training-error insets for depth, keypoints, and normals](/assets/images/gradnorm-source-figure-3-training-test.png)
*Fig 2: Green GradNorm curves improve test error across the three tasks. In the depth panel, the training inset moves the other way, showing why a smaller training loss alone is an inadequate measure of multitask progress. | source: [GradNorm, Figure 3](https://arxiv.org/abs/1711.02257)*

A separate search trains 100 random static weight configurations and compares them with GradNorm at the same 15,000 training steps. GradNorm performs better than those sampled configurations, and configurations closer to its time-averaged weights tend to do better. This supports using an adaptive run to locate useful static weights. It does not establish a global optimum over every possible weighting or a full-duration search: normal training on this dataset lasts 80,000 steps.

### A stalled task can absorb weight without improving

The supplementary facial-landmark experiment makes the limitation concrete. Each face supplies five landmark locations plus four classification labels. GradNorm improves gender and smile classification while maintaining landmark performance; the authors caution that small landmark gains may fall within error bars. Glasses and pose classification instead settle on majority-class predictions and stop improving.

The figure shows why these are distinct tasks sharing the same input, rather than different scales of one regression loss. If a classifier remains stuck, its loss ratio stays high. GradNorm interprets that as insufficient progress and keeps trying to increase its influence. The signal cannot distinguish a task that needs more training from one whose current optimization has become unproductive.

![GradNorm source Figure 8: facial landmarks and classification labels in MTFL](/assets/images/gradnorm-source-figure-8-mtfl.png)
*Fig 3: The same face carries coordinate targets and classification labels. Their differing learning behavior tests rate balancing; the appendix's stalled glasses and pose classifiers expose a case where a larger target gradient does not solve the task. | source: [GradNorm, Figure 8](https://arxiv.org/abs/1711.02257)*

Compared with [homoscedastic uncertainty weighting](/paper%20shorts/2017/05/19/multi-task-learning-using-homoscedastic-uncertainty.html), GradNorm uses observed progress and gradient magnitude to choose the allocation. It still does not inspect gradient direction. Two tasks can have well-balanced norms and oppose each other in parameter space, leaving a separate problem for methods such as [PCGrad](/paper%20shorts/2020/01/19/pcgrad-gradient-surgery-for-multi-task-learning.html). Magnitude, rate, and direction describe different aspects of task interaction.

## High-Level Takeaways

- GradNorm measures task influence at a shared layer and adjusts it using relative loss progress; equal loss coefficients cannot provide that information.
- The stop-gradient target and weight-sum normalization keep the auxiliary update focused on relative allocation rather than shrinking its own target or silently changing the global learning rate.
- Lower depth test error despite higher training error is evidence that task weighting can regularize a shared representation. A smaller task weight need not imply worse task performance.
- Averaged GradNorm weights are competitive with dynamic weights in the reported NYUv2 experiment, making the adaptive run useful as a search procedure as well as a training rule.
- A stalled task can attract weight without improving, and equal norms can still point in conflicting directions. Training rate is useful feedback, but it is not a complete measure of task compatibility.
