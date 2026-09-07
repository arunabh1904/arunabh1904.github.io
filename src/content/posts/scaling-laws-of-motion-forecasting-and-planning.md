---
title: 'Scaling Laws of Motion Forecasting and Planning'
date: '2025-06-09T00:00:00.000Z'
section: paper-shorts
postSlug: scaling-laws-of-motion-forecasting-and-planning
legacyPath: /paper shorts/2025/06/09/scaling-laws-of-motion-forecasting-and-planning.html
tags:
  - Other
field: 'Motion Forecasting & Planning'
summary: "2025 – Scaling Laws of Motion Forecasting and Planning"
---

**arXiv:** [2506.08228](https://arxiv.org/abs/2506.08228)

**Project:** [Waymo research page](https://waymo.com/research/scaling-laws-of-motion-forecasting-and-planning/)

## Summary

This report asks whether joint motion forecasting and planning improve predictably as training compute grows, as language models do. It studies an encoder-decoder autoregressive Transformer on nearly 447 thousand hours of driving data, varying model size, training examples, and compute over 84 models. It then follows the same family into open-loop forecasting, closed-loop simulation, inference-time sampling, and a small cross-agent transfer study.

The useful result is a budget picture rather than a single “bigger is better” curve. Training loss, open-loop metrics, and the controlled closed-loop metric improve with scale in the measured range. At fixed training compute, the compute-optimal number of model parameters grows with exponent 0.63 while the number of training examples grows with exponent 0.44, so model size should grow about 1.5× as fast as data. Inference sampling creates a second scaling axis with its own crossover point.

## Core Insights

### Fit the budget allocation, not only the loss curve

The model predicts discrete motion tokens for eight agents with an encoder-decoder Transformer. The study uses 5 seconds of history to predict 11 seconds of future, creates overlapping examples with a 1.5-second sliding window, and trains on 59.8 million run segments, 5.6 million miles, and 541 million examples. The model series spans 900K to 118M parameters across seven compute budgets. Each iso-FLOP band varies parameters and training examples, then fits a parabola to find the minimum validation loss.

![The encoder-decoder motion model used in the scaling study](/assets/images/scaling-laws-of-motion-forecasting-and-planning-source-figure-1.webp)
*Fig 1: The study scales a shared scene encoder and autoregressive motion decoder, with the same interface across model and data budgets. | source: [Scaling Laws of Motion Forecasting and Planning, Figure 1](https://arxiv.org/abs/2506.08228)*

The fitted optima are `N_opt ∝ C^0.63` and `D_opt ∝ C^0.44`. The ratio is the practical insight: a new compute budget should increase parameters faster than examples, but still spend substantial capacity on data. The authors also note that at equal compute, their motion-optimal models are about 50× smaller than the language-model comparison they use, suggesting that the data distribution and task interface matter as much as the raw Transformer recipe.

### Training loss is a useful signal, with a boundary on the claim

The compute-optimal cross-entropy follows a power-law trend with an added constant that fits the curvature better than a pure power law. The constant could reflect irreducible entropy, repeated passes and overlapping examples, geographic mixture, or information lost in the perception interface; the study does not identify which explanation is correct.

![Training loss across the study's compute range](/assets/images/scaling-laws-of-motion-forecasting-and-planning-paper-figure.png)
*Fig 2: Each curve is a training run; moving right increases FLOPs while the colored family varies parameter count. The convergence pattern supports a scaling fit within the measured range, not an assumption that the same slope holds indefinitely. | source: [Scaling Laws of Motion Forecasting and Planning, Figure 2](https://arxiv.org/abs/2506.08228)*

The open-loop analysis aggregates 64 rollouts to 12 trajectories and reports that minADE and weighted ADE improve as compute-optimal training compute grows. The authors explicitly caution that a power law for distance metrics would require many more orders of magnitude. That caveat matters because the plotted fit is a compact description of the observed range, while the actual ablation found a parabolic form could fit the points better.

### The closed-loop result is controlled, not automatic

To test whether scale transfers to driving behavior, each forecasting model is fine-tuned for route-conditioned AV planning with the same `10^15` FLOPs. The simulator runs 30-second scenarios at 10 Hz; each action is selected from 128 trajectory rollouts, with a progress bias calibrated per model. A failure means excessive progress, insufficient progress, or a collision relative to manual driving. Failures decrease as pretraining compute grows, but the result is controlled by using one architecture, matched fine-tuning, and calibrated assertiveness. It supports open-loop loss as a proxy in this study; it does not remove the need for closed-loop evaluation.

### Sampling is another compute budget

At inference, the authors vary samples from 8 to 1024 and cluster them to six modes with NMS. Distance and coverage metrics improve over roughly three orders of magnitude, then saturate for each model. A larger model becomes more efficient after a crossover, so a small model plus many samples is useful only in part of the inference-FLOP range. A final transfer experiment finds that, within the same platform's correlated data, ten observed miles can be equivalent to roughly two to three demonstrated miles for AV loss; the authors flag the platform correlation as a limitation.

## High-Level Takeaways

- The paper turns scaling into a three-way decision among parameters, training examples, and inference samples. Its most concrete allocation signal is `C^0.63` for parameters versus `C^0.44` for examples.
- Closed-loop improvement is encouraging because the model family, fine-tuning budget, rollout policy, and assertiveness calibration were controlled. The result is evidence for this pipeline, not a universal guarantee across architectures or geographies.
- The power-law-plus-constant fit is more informative than a bare straight line: curvature may reflect data overlap, limited coverage, or a lossy perception interface rather than a solved task.
- Inference-time sampling extends the useful life of a smaller model, but only until a larger model becomes the better FLOP allocation.
