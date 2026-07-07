---
title: 'DenseTNT: End-to-End Trajectory Prediction from Dense Goal Sets'
date: '2021-08-22T04:00:00.000Z'
section: paper-shorts
postSlug: densetnt-end-to-end-trajectory-prediction-from-dense-goal-sets
legacyPath: /paper shorts/2021/08/22/densetnt-end-to-end-trajectory-prediction-from-dense-goal-sets.html
tags:
  - Other
field: BEV
summary: DenseTNT removes sparse goal anchors and heuristic goal selection by predicting dense goal probabilities and learning an online goal-set predictor.
---
## 2021 - DenseTNT

**arXiv:** [2108.09640](https://arxiv.org/abs/2108.09640)

**Project:** [DenseTNT project page](https://tsinghua-mars-lab.github.io/DenseTNT/)

**GitHub:** [Tsinghua-MARS-Lab/DenseTNT](https://github.com/Tsinghua-MARS-Lab/DenseTNT)

**CVF:** [ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/html/Gu_DenseTNT_End-to-End_Trajectory_Prediction_From_Dense_Goal_Sets_ICCV_2021_paper.html)

**Plain-language summary:** DenseTNT is the direct follow-up to goal-based methods like TNT. TNT showed that endpoints are a strong way to represent intent, but sparse anchors and NMS-style goal selection still leave a lot of hand-designed machinery in the loop.

DenseTNT replaces that machinery with dense goal probability estimation and a learned goal-set predictor. The model scores dense candidate positions on the road, predicts a set of goals from that heatmap, and completes trajectories conditioned on those goals.

## Paper Insights

DenseTNT starts from the observation that sparse goal anchors are too coarse. One anchor can only generate one goal, and two positions on the same lane can carry different local information. The paper therefore samples dense goal candidates on nearby lanes, uses a dense goal encoder to estimate a probability distribution over those candidates, and feeds the distribution into a multi-head goal-set predictor.

The main training problem is supervision. Each driving log shows only one realized future, while a trajectory predictor should output several plausible futures. DenseTNT solves this by using an offline optimization model to turn the dense goal heatmap into multi-future pseudo-labels. The online goal-set predictor then learns to imitate those pseudo-labels, so inference does not need the optimization loop.

![Figure 2 from DenseTNT showing sparse context encoding, dense goal probability estimation, goal-set prediction, and trajectory completion](/assets/images/densetnt-end-to-end-trajectory-prediction-from-dense-goal-sets-paper-figure.png)
_Figure 2 shows the dense-goal pipeline: vectorized context features become a goal heatmap, a learned goal-set predictor selects goals, and a trajectory decoder completes the futures. From the [DenseTNT paper](https://arxiv.org/abs/2108.09640), via the arXiv PDF._

**What to look at:**
- Dense goal scoring removes dependence on sparse predefined anchors.
- Goal-set prediction replaces NMS with a learned selection module.
- Offline optimization supplies multi-future pseudo-labels that are missing from ordinary logged data.
- The online model runs without the optimization loop, but its quality depends on the heatmap and pseudo-label objective.

**Evals / Benchmarks / Artifacts:**

| Component | Detail | Why it matters |
| --------- | ------ | -------------- |
| Context encoder | VectorNet-style sparse map and agent encoder | Preserves lane and agent structure. |
| Dense goal encoder | Scores dense candidate locations on nearby lanes | Captures fine-grained endpoint choices. |
| Offline model | Optimizes a goal set from the heatmap | Creates multi-future pseudo-labels from single-future logs. |
| Online model | Multi-head goal-set predictor plus trajectory completion | Avoids optimization at inference time. |

**Compact result slice:**

The Argoverse slice shows the tradeoff: DenseTNT improves miss-rate coverage, even when the online goal-set predictor does not win every raw distance metric.

| Setting | minADE | minFDE | Miss rate |
| ------- | ------ | ------ | --------- |
| TNT on Argoverse validation | 0.73 | 1.29 | 9.3% |
| DenseTNT with 100 ms optimization | 0.80 | 1.27 | 7.0% |
| DenseTNT with minFDE objective | 0.73 | 1.05 | 9.8% |
| DenseTNT online goal-set predictor | 0.82 | 1.37 | 7.0% |

**Waymo challenge slice:**

Waymo used mAP as the official ranking metric, so this table is about calibrated multimodal prediction rather than only the closest trajectory error.

| Method | mADE | mFDE | Miss rate | mAP |
| ------ | ---- | ---- | --------- | --- |
| TVN | 0.7558 | 1.5859 | 0.2032 | 0.3168 |
| SceneTransformer | 0.6117 | 1.2116 | 0.1564 | 0.2788 |
| DenseTNT | 1.0387 | 1.5514 | 0.1779 | 0.3281 |

**Why it mattered:** DenseTNT pushed goal-conditioned forecasting away from sparse anchor heuristics and toward dense probability maps plus learned set prediction.

**Take-home message:** TNT made endpoints the intent variable; DenseTNT made endpoint selection dense, learned, and closer to end-to-end.
