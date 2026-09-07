---
title: 'Small-Scale Experiments: Are We There Yet?'
date: '2026-08-12T09:47:01.000Z'
section: paper-shorts
postSlug: small-scale-experiments-are-we-there-yet
legacyPath: /paper shorts/2026/08/12/small-scale-experiments-are-we-there-yet.html
tags:
  - Language Models
  - Scaling Laws
  - Experiment Design
  - Hyperparameters
field: 'Training Systems & Reliability'
summary: '2026 – Small-Scale Experiments: Are We There Yet?'
---

## 2026 – Small-Scale Experiments: Are We There Yet?

**arXiv:** [2608.11859](https://arxiv.org/abs/2608.11859)

## Summary

> Scaling laws appear at four million parameters when each model–data budget is tuned to its attainable frontier. The obstacle is search: four or sixteen random configurations do not reveal a reliable law, sixty-four expose it weakly, and 256 produce accurate extrapolation to the held-out 268M scale. The paper explains this asymmetry through the hyperparameter loss surface, whose estimated intrinsic dimension falls as models grow. Small models can therefore support cheap architecture decisions, but only after unusually broad tuning and only near the measured range.

## Core Insights





### A scaling law describes the tuned frontier

The paper trains models from 4M to 268M effective parameters with a warmup–stable–decay schedule. Models from 4M to 34M fit the law, 67M and 134M validate choices, and 268M is the held-out test scale. Checkpoints at eight token budgets reuse the stable training phase and branch into separate decay phases, which makes the sweep cheaper than training every model–data point independently.

“Effective parameters” is a compute accounting convention: it excludes the embedding lookup, includes the output projection, and accounts for attention cost. It is not simply a count of stored weights. The sweep also evaluates losses before and after each decay branch, so one underlying stable trajectory supports several training-budget measurements.

The crucial ablation varies the number of random configurations available to the fitting search. With four configurations, held-out test mean squared error is $1.30\times10^{-2}$. With 64 it falls to $2.79\times10^{-3}$. With 256 it reaches $3.70\times10^{-6}$. Choices such as parameter accounting, tied scaling exponents, and per-budget learning-rate decay refine the estimate, but none compensates for missing the tuned frontier.

![Small-scale study source Figure 5 comparing scaling fits after different amounts of hyperparameter search](/assets/images/small-scale-hyperparameter-tuning-frontier.png)
*Fig 1: The diagonal means predicted and measured bits per character agree. As the search broadens, held-out pink points move toward it; good training fit alone does not ensure accurate extrapolation. | source: [Small-Scale Experiments, Figure 5](https://arxiv.org/abs/2608.11859)*

The plot compares predicted loss horizontally with actual loss vertically, not loss against parameter count. Read the pink test points separately from the blue training points. With four configurations, a seemingly reasonable fit to observed points still misses the held-out scale. Wider search changes the frontier being fitted, not just the regression's numerical precision.

This changes the economics of proxy studies. Tiny models are cheap per run, yet a credible comparison may need hundreds of runs. The relevant budget is the complete search, not the cost of one 4M model.

### Scale reduces hyperparameter sensitivity

The authors fit a noisy-quadratic distribution to the best tail of random-search outcomes. Its effective hyperparameter count estimates the local intrinsic dimension of the loss surface. That count trends down toward one as parameter scale increases. That estimate is not a literal count of knobs in the training script, nor the dimension of the neural network's parameter space. It summarizes how many local directions in the searched hyperparameters strongly affect near-optimal loss. A low value means many sampled settings can perform similarly well.

The sensitivity plots add an important condition: parameters and training data need to grow together. Enlarging a model while keeping training short, or training a tiny model longer, lowers loss without eliminating most of the gap between the best and merely good configurations.

![Source Figure 7 comparing best, tenth-percentile, and twenty-fifth-percentile losses as size and training duration vary](/assets/images/small-scale-source-figure-7-sensitivity.png)
*Fig 2: The gap above the best configuration measures sensitivity. It shrinks most when a large model also trains longer; increasing only one dimension while the other stays small leaves much of the gap. | source: [Small-Scale Experiments, Figure 7](https://arxiv.org/abs/2608.11859)*

A narrow gap to the 25th percentile means roughly a quarter of the sampled configurations are already near the best observed loss. That is a property of the specified search distribution. A different range of learning rates, batch sizes, or architectures could change it.

The result explains why a poorly tuned small proxy can rank ideas incorrectly even when large models are easy to tune. It also suggests a practical split: explore many configurations where runs are cheap, then carry a small set of simple hyperparameter rules upward as sensitivity falls.

The geometric account is empirical. It depends on the tested search space, parametrization, architecture family, optimizer, and fixed corpus. The paper notes that alternative parametrizations such as maximal update could change the trend.

### Diagnostics matter more than distant point estimates

The proposed workflow combines three checks:

| Diagnostic | Question it answers |
| --- | --- |
| Noisy-quadratic tail | Did the search reach the local tuning frontier? |
| Scaling law near the data | Which model family buys lower pretraining loss at equal compute? |
| Perplexity–capability correspondence | Does lower loss still track downstream capability under the fixed corpus? |

The pre-norm versus post-norm case study applies that sequence. The authors search 128–512 configurations at 4M and 34M parameters, validate at 134M, and find pre-norm easier to tune and better-scaling near the observations. Far beyond the data, the ranking depends on whether the two laws share an irreducible loss floor. The paper therefore chooses pre-norm from the near-data evidence and tuning profile, not from one distant extrapolated crossing.

The branching schedule also matters to a fair architecture comparison. An undecayed checkpoint can make a model look worse simply because it has not completed its optimization schedule. Reusing the stable phase saves compute while allowing each measured budget to receive a decay phase; it does not make all budget measurements independent training runs.

The fixed-corpus condition is the hard boundary. Equal pretraining loss can correspond to similar capabilities when the data composition stays fixed. Change the data and that proxy can break; this method does not solve data-centric experimentation.

## High-Level Takeaways

- A scaling law describes the tuned frontier; a few inexpensive runs may never reach it.
- More search and better regression solve different problems, so evaluate held-out scales explicitly.
- Hyperparameter sensitivity decreases most when model size and training data grow together in the tested regime.
- The estimated local dimension describes the search surface, not a universal number of training knobs.
- Compare architecture families near observations; distant floor estimates and changes in data composition can reverse the conclusion.
