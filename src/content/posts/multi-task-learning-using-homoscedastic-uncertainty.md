---
title: 'Multi-Task Learning Using Homoscedastic Uncertainty'
date: '2017-05-19T04:00:00.000Z'
section: paper-shorts
postSlug: multi-task-learning-using-homoscedastic-uncertainty
legacyPath: /paper shorts/2017/05/19/multi-task-learning-using-homoscedastic-uncertainty.html
tags: [Other]
field: 'Training Systems & Reliability'
summary: '2017 – Learn global task-loss weights through homoscedastic uncertainty'
---
## 2017 – Homoscedastic-Uncertainty Weighting

**arXiv:** [1705.07115](https://arxiv.org/abs/1705.07115)

## Summary

> Learning one uncertainty parameter per task makes a shared scene-understanding model less dependent on manually chosen loss coefficients. On Tiny Cityscapes, the three-task model reaches 63.4% semantic IoU versus 50.1% with equal weighting and 59.4% for semantic segmentation alone. The mechanism combines a loss scale with a penalty that makes ignoring a task costly. Its uncertainty is global for each task, not an estimate of which image or pixel is unreliable; shared features can still produce correlated errors across all three outputs.

## Core Insights

### Different loss units change the shared representation

A network predicting classes, object instances, and depth receives three different kinds of supervision. Cross-entropy measures classification error, instance regression measures pixel displacements, and the depth head predicts inverse distance. Adding these losses with equal coefficients makes the update depend on arbitrary numerical scales. Changing a regression target's units can change its influence even though the underlying task is the same.

The paper learns the coefficients through a likelihood model. A shared ResNet-101 encoder with atrous spatial pyramid pooling supplies features to small task-specific decoders. The uncertainty parameters act on the training losses, where they determine how strongly each task updates that shared representation. They are learned scalars, shared across inputs, rather than uncertainty maps produced by an extra decoder.

Read the first figure from the three outputs toward the summation. Each task still needs its own labels and loss. The learned scale changes their relative contributions; it does not convert semantic, instance, and depth supervision into interchangeable targets.

![Source Figure 1: shared encoder, three task decoders, and task uncertainty weighting](/assets/images/multi-task-learning-using-homoscedastic-uncertainty-source-figure-1.webp)
*Fig 1: A shared encoder supports semantic, instance, and depth outputs. Each task's learned uncertainty scales its contribution to the combined training objective; the uncertainty is a task-level parameter rather than an image-dependent prediction. | source: [Multi-Task Learning Using Uncertainty, Figure 1](https://arxiv.org/abs/1705.07115)*

### The regularizer prevents the trivial zero-weight solution

For the paper's Gaussian regression derivation, a squared residual loss $L_i$ contributes

$$
\frac{1}{2\sigma_i^2}L_i+\log\sigma_i.
$$

Increasing $\sigma_i$ reduces the residual's weight, but also increases the second term. Without that penalty, directly minimizing a weighted sum over freely learned nonnegative coefficients could simply set every coefficient to zero. The likelihood ties the reward for downweighting a noisy task to a cost for declaring it uncertain.

The implementation uses the log variance, $s_i=\log\sigma_i^2$. The same Gaussian expression becomes $\tfrac12 e^{-s_i}L_i+\tfrac12s_i$, allowing an unconstrained scalar parameter while keeping the variance positive. This form also reveals that the learned scale responds to the residual the model can achieve. It should not be read as a pure measurement of irreducible sensor noise, isolated from model fit or loss normalization.

Classification needs a separate step. The paper introduces a temperature-scaled softmax and then explicitly approximates its normalization term to obtain a weighted cross-entropy plus a log-scale penalty. That simplification is exact at unit scale, not an exact identity for arbitrary logits and temperatures. Likewise, the demonstrated depth and instance heads use absolute-error losses; the displayed squared-error derivation explains the Gaussian case rather than describing every task loss literally.

### Instance regression supplies geometry that class labels lack

The instance head predicts a two-dimensional vector from each object pixel to its object's centroid. Adding the pixel coordinate to that vector yields a vote for the center. OPTICS clusters those votes without requiring the number of objects in advance, and pixels are assigned to the resulting instances. Supervision applies only to pixels belonging to instance classes.

This representation handles a useful case: a tree or lamppost can split a car's visible mask into disconnected regions, but both regions can still vote for the same car center. Connectivity in the image is not required for agreement in the voting space. The semantic head identifies relevant object classes, while the geometric target separates individual objects within a class.

The four panels below trace that distinction. Semantic segmentation in panel (b) does not distinguish every car. Panel (c) encodes vector orientation as color and magnitude as intensity; clustering the projected votes produces the separate instance regions in panel (d).

![Source Figure 3: input, semantic segmentation, instance vectors, and instance segmentation](/assets/images/uncertainty-weighting-source-figure-3-instance-votes.png)
*Fig 2: The full four-panel figure separates class prediction from instance grouping. Pixels belonging to one object vote toward a common centroid, so disconnected visible regions can still receive the same instance identity. | source: [Multi-Task Learning Using Uncertainty, Figure 3](https://arxiv.org/abs/1705.07115)*

### The controlled result is about weighting, not universal task compatibility

The weighting comparison trains at 128×256 resolution for 50,000 iterations and reports Tiny Cityscapes validation results. Keeping that setting explicit matters: these are not the full-resolution test leaderboard numbers.

| Training objective | Semantic IoU, % | Instance error | Inverse-depth error |
| --- | --- | --- | --- |
| Separate single-task models | 59.4 | 4.61 | 0.640 |
| Equal three-task weights | 50.1 | 3.79 | 0.592 |
| Approximately tuned fixed weights | 62.8 | 3.61 | 0.549 |
| Learned three-task uncertainty | 63.4 | 3.50 | 0.522 |

Higher IoU and lower errors are better. Equal weighting improves the two regression metrics over separate models while harming segmentation. Learning the weights improves all three in this experiment. The fixed-weight comparison is an approximate search, so outperforming it does not prove that no better fixed coefficients exist. The authors attribute the advantage to both search resolution and the ability to change weights during training.

More tasks do not monotonically improve every result. Semantic-plus-instance training reports instance error 3.42, slightly better than the three-task model's 3.50. The third task improves the broader set of outcomes while introducing a trade-off in that metric. The paper's central benefit is a practical way to choose an allocation, not a guarantee that every added task helps every head.

### Global uncertainty cannot identify a bad frame

For the full-resolution model, the final relative weights are approximately 43:1:0.16 in semantic, depth, and instance order. These are learned training scales, not percentages of task importance. Appendix B also notes that changing uncertainty changes the total loss scale and therefore the effective learning rate; the experiments anneal the learning rate during training.

That global scaling differs from [GradNorm's](/paper%20shorts/2017/11/07/gradnorm-adaptive-loss-balancing.html) explicit normalization of the sum of task weights. The two methods also use different feedback: this likelihood formulation responds to task residuals, while GradNorm measures shared-layer gradient norms and relative progress. Neither global weighting rule, by itself, detects that one particular camera frame is degraded.

The failure examples make the shared-representation boundary visible. A reflection incorrectly classified as a person can also induce an incorrect human-shaped depth estimate. Agreement between outputs is therefore not independent confirmation: the same feature error can propagate into several heads. Task weighting helps the model learn jointly, but does not make its predictions independent or provide a guarantee against shared failures.

## High-Level Takeaways

- Inverse uncertainty gives heterogeneous losses an adaptive scale, while the log penalty prevents the trivial solution of ignoring every task.
- The Gaussian derivation, approximate classification objective, and actual absolute-error regression heads need to be distinguished when reproducing the method.
- The controlled Tiny Cityscapes experiment shows why equal weighting can help regression while damaging segmentation; learning the weights improves the joint trade-off in that setting.
- Instance centroid voting provides a geometric target that can group disconnected visible regions of the same object. It adds information beyond semantic class labels.
- One scalar per task cannot express frame-dependent reliability, and a shared encoder can produce correlated mistakes across heads. Learned loss scales are not a complete solution to task conflict or uncertainty estimation.
