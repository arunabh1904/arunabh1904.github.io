---
title: 'TNT: Target-driveN Trajectory Prediction'
date: '2020-08-19T00:00:00.000Z'
section: paper-shorts
postSlug: tnt-target-driven-trajectory-prediction
legacyPath: /paper shorts/2020/08/19/tnt-target-driven-trajectory-prediction.html
tags:
  - Other
field: 'Motion Forecasting & Planning'
summary: "2020 – TNT: Target-driveN Trajectory Prediction"
---
## 2020 – TNT: Target-driveN Trajectory Prediction

**arXiv:** [2008.08294](https://arxiv.org/abs/2008.08294)<br>
**PMLR:** [CoRL 2020 proceedings](https://proceedings.mlr.press/v155/zhao21b.html)<br>
**Project:** [Waymo research page](https://waymo.com/research/tnt-target-driven-trajectory-prediction/)

## Summary

> TNT moves much of trajectory uncertainty into an explicit endpoint distribution, predicts a path toward each selected endpoint, then ranks and removes near-duplicate paths. On Argoverse validation, the six-trajectory result improves minFDE from 1.68 m for a MultiPath reimplementation using the same VectorNet encoder to 1.29 m. The decisive ablation is downstream of endpoint generation: scoring whole trajectories cuts six-mode miss rate from 0.216 to 0.093. Candidate coverage, path quality, and final selection remain separate bottlenecks, and the endpoint-conditioned unimodality assumption becomes less convincing as the horizon grows.

## Core Insights

### Choose where before predicting how

A car approaching an intersection can turn, continue, or slow down. Directly regressing one future risks averaging distinct possibilities into an implausible path. TNT instead represents multiple candidate locations at a fixed future time. Different endpoints can encode both route and speed choices, making the intermediate uncertainty visible in physical coordinates.

The factorization is

$$
p(s_{1:T}\mid x)\approx\sum_{\tau\in\mathcal T(x)}p(\tau\mid x)\,p(s_{1:T}\mid\tau,x).
$$

Here $x$ is the observed scene and $\tau$ is a candidate future endpoint. The modeling assumption is that conditioning on that endpoint leaves a trajectory distribution simple enough for one regression head. This is an approximation about the remaining uncertainty, not a claim that all paths to the same destination are identical.

TNT uses [VectorNet](/paper%20shorts/2020/05/08/vectornet-encoding-hd-maps-and-agent-dynamics-from-vectorized-representation.html) to encode maps and agent histories when vector maps exist. On Stanford Drone, it uses a ResNet-50 image encoder instead. The contribution therefore sits mainly after context encoding: endpoints organize the output space, while the scene encoder can change with the available input.

The pipeline below separates three selections that can otherwise blur together. Dense candidate locations enter the target predictor; a smaller set of refined targets enters motion estimation; an even smaller set of complete trajectories leaves the final scorer.

![TNT source Figure 2: context encoding and the target, motion, and scoring stages](/assets/images/tnt-target-driven-trajectory-prediction-paper-figure.png)
*Fig 1: Candidate endpoints are scored and refined before trajectories are generated. A separate scorer then evaluates the complete paths and selects a compact set, rather than treating endpoint probability as the final trajectory score. | source: [TNT, Figure 2](https://arxiv.org/abs/2008.08294)*

### Candidate design determines which futures are easy to represent

The first stage predicts a probability and a continuous offset for every sampled location. Its classification target is the candidate nearest the ground-truth endpoint; Huber regression refines that candidate's location. The offset matters: the Argoverse target-stage ablation improves fifty-candidate minFDE from 0.69 to 0.53 m when refinement is enabled. Discretization supplies coverage without forcing the answer to lie exactly on the grid.

Vehicle candidates come from lane centerlines on Argoverse and lane boundaries on INTERACTION. Pedestrian candidates use a surrounding grid because movement is less constrained by lane geometry. The paper gives a typical coarse-to-fine example of roughly 1,000 input locations reduced to fifty targets, but the actual candidate set depends on the scene and dataset.

Denser sampling helps until it stops resolving a meaningful ambiguity. On Argoverse, reducing target spacing from five meters to two and one improves six-mode minFDE from 1.55 to 1.31 and 1.29 m. Halving it again to 0.5 m leaves the result at 1.29 m. Continuous offsets and the later stages make ever finer candidate spacing unnecessary in that experiment.

### A likely endpoint does not guarantee a likely path

The second stage takes one target and the context feature and predicts all future coordinates with a two-layer MLP. During training, it receives the ground-truth endpoint through teacher forcing. At inference, it receives the target stage's predictions. Future timesteps are modeled as conditionally independent given the endpoint and context, enabling parallel decoding rather than a recurrent rollout.

The conditional-unimodality ablation compares the Huber regressor with a CVAE. With one trajectory per target, both give approximately 0.73 m minADE after final selection. Sampling ten CVAE trajectories per target improves that to 0.71 m. The modest gain supports the simpler decoder for the tested horizons; it does not rule out multiple routes or timing patterns sharing one endpoint on longer journeys.

The final scorer sees whole trajectories. It is trained against a soft distribution based on each proposed path's distance from the ground truth, using the maximum squared pointwise displacement as the distance measure. At inference, candidates are sorted by score and selected greedily, suppressing paths too similar to those already retained. This adds both path-level plausibility and diversity to the endpoint ranking.

### Selection recovers useful coverage within a six-trajectory budget

The stage ablation measures the distinction directly:

| Argoverse validation output | Candidate count | minFDE, m | minADE, m | Miss rate at 2 m |
| --- | --- | --- | --- | --- |
| Target stage | 50 | 0.533 | — | 0.027 |
| Motion stage, endpoint-ranked | 6 | 1.632 | 0.877 | 0.216 |
| Final trajectory scoring and selection | 6 | 1.292 | 0.728 | 0.093 |

The fifty targets cover the observed endpoint well, but retaining only six by endpoint ranking loses substantial coverage. Scoring the complete trajectories nearly halves that six-mode miss rate again. There is still a gap from the fifty-candidate oracle: a compact prediction set cannot inherit the larger set's coverage for free.

The retained example below shows the same progression visually. Many targets and trajectories cluster along plausible routes in the first two panels. The right panel keeps six representatives after scoring and suppression. It is a crop of one example from the source figure, not a summary of every behavior evaluated.

![TNT source Figure 4, example crop: fifty targets, fifty trajectories, and six selected trajectories](/assets/images/tnt-target-driven-trajectory-prediction-source-figure-4.webp)
*Fig 2: One Argoverse example traces candidate targets through motion estimation to the selected six paths. The reduction removes redundancy, but its usefulness depends on retaining distinct routes and speeds rather than merely the highest endpoint scores. | source: [TNT, Figure 4, example crop](https://arxiv.org/abs/2008.08294)*

### The benchmark result depends on both split and output budget

On Argoverse validation, TNT reports 1.29 m minFDE, 0.73 m minADE, and 0.09 miss rate with six trajectories. MultiPath, reimplemented with the same VectorNet context encoder, reports 1.68 m, 0.80 m, and 0.14. Sharing the encoder makes this a more informative test of the prediction design than comparing systems with unrelated scene representations.

The test leaderboard comparison is mixed. TNT has lower minADE than the cited challenge winner, 0.94 versus 0.97 m, and the same 0.13 miss rate, but worse minFDE, 1.54 versus 1.42 m. The paper's broad competitive-performance description should therefore be read metric by metric.

The pedestrian results also use different budgets and units. PAID reports 0.32 m minFDE and 0.18 m minADE with three trajectories. Stanford Drone reports 21.16 and 12.23 pixels with five trajectories and a longer 4.8-second prediction horizon. Those values cannot be directly ranked against the six-mode vehicle results. The portable idea is the endpoint-conditioned decomposition; candidate geometry, horizon, and evaluation budget still determine its usefulness.

## High-Level Takeaways

- TNT makes multimodality explicit through candidate endpoints, then models the path conditional on each endpoint. The endpoint is a useful intent proxy rather than a complete description of intent.
- Target offsets recover precision without arbitrarily dense sampling; the Argoverse spacing ablation saturates at about one meter in the tested configuration.
- Whole-trajectory scoring substantially improves the six-mode miss rate over endpoint ranking, demonstrating that target recall and final forecast quality are different problems.
- A richer conditional decoder adds little in the reported ablation, supporting simple regression for those horizons without proving it sufficient for long-term motion.
- Test gains are mixed across metrics, and pedestrian benchmarks change both units and candidate counts. The clean comparison holds the context encoder and output budget fixed.
