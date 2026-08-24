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

![Scaling-law fits obtained after random searches of 4, 16, 64, and 256 hyperparameter configurations](/assets/images/small-scale-hyperparameter-tuning-frontier.png)
*The scaling law is fitted to the best run found at each scale. With a small search, poor configurations obscure the frontier; a 256-configuration search makes the held-out 268M prediction accurate.. source: [Small-Scale Experiments: Are We There Yet?](https://arxiv.org/abs/2608.11859)*

![Figure 1 from Small-Scale Experiments: Are We There Yet?](/assets/images/small-scale-experiments-are-we-there-yet-source-figure-1.webp)
*Figure 1 Scaling laws emerge at very small scales, smaller than widely believed; however, hyperparameter tuning becomes critical. Predictable scaling appears only at the fully tuned frontier. Smaller models are harder to tune, making their scaling laws harder to observe though no less present. source: [Small-Scale Experiments: Are We There Yet?](https://arxiv.org/abs/2608.11859)*

![Figure 2 from Small-Scale Experiments: Are We There Yet?](/assets/images/small-scale-experiments-are-we-there-yet-source-figure-2.webp)
*Figure 2 Given a fixed corpus, perplexity corresponds to downstream capability across a wide variety of tasks. The relationship between pretraining loss and capability is not always predictable or even monotonic; however, models achieving the same loss have a similar conditional distribution over their capabilities as we increase parameters and data, or even randomize the architecture, as long as the composition of pretraining data remains fixed. source: [Small-Scale Experiments: Are We There Yet?](https://arxiv.org/abs/2608.11859)*


### A scaling law describes the tuned frontier

The paper trains models from 4M to 268M effective parameters with a warmup–stable–decay schedule. Models from 4M to 34M fit the law, 67M and 134M validate choices, and 268M is the held-out test scale. Checkpoints at eight token budgets reuse the stable training phase and branch into separate decay phases, which makes the sweep cheaper than training every model–data point independently.

The crucial ablation varies how many random hyperparameter configurations are available at each scale. With four configurations, held-out test mean squared error is $1.30\times10^{-2}$. With 64 it falls to $2.79\times10^{-3}$. With 256 it reaches $3.70\times10^{-6}$. Choices such as parameter accounting, tied scaling exponents, and per-budget learning-rate decay refine the estimate, but none compensates for missing the tuned frontier.

This changes the economics of proxy studies. Tiny models are cheap per run, yet a credible comparison may need hundreds of runs. The relevant budget is the complete search, not the cost of one 4M model.

### Scale reduces hyperparameter sensitivity

The authors fit a noisy-quadratic distribution to the best tail of random-search outcomes. Its effective hyperparameter count estimates the local intrinsic dimension of the loss surface. That count trends down toward one as parameter scale increases. Random large-model configurations are therefore more likely to land near the optimum than random small-model configurations.

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

The fixed-corpus condition is the hard boundary. Equal pretraining loss can correspond to similar capabilities when the data composition stays fixed. Change the data and that proxy can break; this method does not solve data-centric experimentation.

## High-Level Takeaways

- Small-scale experiments trade model compute for search breadth. A cheap proxy run is not evidence unless the hyperparameter frontier has been found.
- Hyperparameter sensitivity falls with scale in the studied regime, so small models require the most exploration even though each run costs the least.
- Use scaling laws to compare families near observed compute. Long extrapolations become dominated by an uncertain irreducible loss floor.
- A decisive replication would repeat the protocol across optimizers, parametrizations, architectures, and fixed corpora while holding total search compute constant. Reject the small-scale ranking when it changes under reasonable search spaces or fails at the next measured scale.
