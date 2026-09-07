---
title: 'DenseTNT: End-to-End Trajectory Prediction from Dense Goal Sets'
date: '2021-08-22T00:00:00.000Z'
section: paper-shorts
postSlug: densetnt-end-to-end-trajectory-prediction-from-dense-goal-sets
legacyPath: /paper shorts/2021/08/22/densetnt-end-to-end-trajectory-prediction-from-dense-goal-sets.html
tags:
  - Other
field: 'Motion Forecasting & Planning'
summary: "2021 – DenseTNT: End-to-End Trajectory Prediction from Dense Goal Sets"
---

**arXiv:** [2108.09640](https://arxiv.org/abs/2108.09640)

**Project:** [DenseTNT project page](https://tsinghua-mars-lab.github.io/DenseTNT/)

**GitHub:** [Tsinghua-MARS-Lab/DenseTNT](https://github.com/Tsinghua-MARS-Lab/DenseTNT)

**CVF:** [ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/html/Gu_DenseTNT_End-to-End_Trajectory_Prediction_From_Dense_Goal_Sets_ICCV_2021_paper.html)

## Summary

DenseTNT treats the endpoint as the compact representation of trajectory intent, but removes the sparse hand-designed anchors used by earlier goal-based predictors. It samples a dense set of reachable positions around nearby lanes, estimates a probability for each position, predicts a small set of goals from that heatmap, and completes one trajectory per goal.

The interesting part is the supervision problem. A log contains one realized future, even though the predictor should cover several plausible futures. DenseTNT first uses an offline optimizer to turn the dense endpoint distribution into multi-goal pseudo-labels, then trains an online goal-set predictor to imitate them. The deployed model therefore keeps the dense coverage idea without running the optimizer at inference time.

## Core Insights

### Make the endpoint distribution dense before choosing modes

TNT-style anchors are sparse and often place one candidate per lane. That can miss nearby positions on the same lane that correspond to different speeds or maneuvers, and NMS needs a heatmap-dependent threshold. DenseTNT samples lane candidates within a 50 m Manhattan neighborhood, keeps points within 3 m of a lane centerline or inside a lane boundary, and uses 1 m spacing between adjacent candidates. A sparse VectorNet-style encoder represents lanes and agents; a dense goal encoder combines each candidate with that context to produce a categorical goal distribution.

![DenseTNT's dense-goal architecture, from context encoding through goal-set prediction](/assets/images/densetnt-end-to-end-trajectory-prediction-from-dense-goal-sets-paper-figure.png)
*Fig 1: DenseTNT scores dense endpoint candidates, predicts a compact goal set, and completes a trajectory for each selected goal. | source: [DenseTNT, Figure 2](https://arxiv.org/abs/2108.09640)*

### Use optimization only to create the missing multimodal labels

The offline model chooses a goal set by minimizing the expected endpoint error under the heatmap. Its hill-climbing search can find several high-probability, well-separated goals, but it is too expensive and too brittle to make the inference path. The online model replaces that search with multiple goal-set heads. During training, the authors perturb predicted goals 100 times, retain the best candidates, and train the set predictor against those pseudo-labels. The trajectory decoder is a two-layer MLP conditioned on a goal; it uses the ground-truth goal during training and a smooth L1 loss over the full future. Training is staged: 16 epochs train the lane, heatmap, and completion modules, then 6 epochs train the goal-set predictor from the offline labels.

![DenseTNT's two-stage training diagram](/assets/images/densetnt-end-to-end-trajectory-prediction-from-dense-goal-sets-source-figure-3.webp)
*Fig 2: The first stage learns the context, dense goal, and completion modules; the second stage learns to replace offline goal optimization with a predictor. | source: [DenseTNT, Figure 3](https://arxiv.org/abs/2108.09640)*

### The result is a coverage tradeoff, not a uniform distance win

On the Argoverse validation split, TNT reports minADE/minFDE/miss rate of 0.73/1.29/9.3%. DenseTNT with 100 ms of optimization reaches 0.80/1.27/7.0%, while optimizing the minFDE objective reaches 0.73/1.05/9.8%. The online goal-set predictor reaches 0.82/1.37/7.0%. The pattern is revealing: the learned selector preserves coverage at the same 7.0% miss rate as the default optimizer, while the objective that improves closest-endpoint error can sacrifice coverage.

| setting | minADE | minFDE | miss rate |
| --- | ---: | ---: | ---: |
| TNT | 0.73 | 1.29 | 9.3% |
| DenseTNT, 100 ms optimization | 0.80 | 1.27 | 7.0% |
| DenseTNT, minFDE objective | 0.73 | 1.05 | 9.8% |
| DenseTNT, online goal-set predictor | 0.82 | 1.37 | 7.0% |

The endpoint resolution is also a real compute knob. Moving from 3 m to 1 m sampling changes minFDE/MR from 1.42/12.5% to 1.27/7.0%; 0.5 m gives the same 1.27/7.0%. Increasing optimizer time from 20 ms to 100 ms changes 1.29/7.6% to 1.27/7.0%, with only small gains beyond that. On the Waymo challenge, the online system reports mADE 1.0387, mFDE 1.5514, miss rate 0.1779, and mAP 0.3281.

![DenseTNT's qualitative online predictions](/assets/images/densetnt-end-to-end-trajectory-prediction-from-dense-goal-sets-source-figure-5.webp)
*Fig 3: Dense heatmaps support several endpoint modes; the selected goals and completed trajectories cover those modes while the green path is the observed future. | source: [DenseTNT, Figure 5](https://arxiv.org/abs/2108.09640)*

## High-Level Takeaways

- DenseTNT is useful when endpoint uncertainty dominates trajectory uncertainty: spend capacity on a dense, map-aware endpoint distribution and decode paths conditionally.
- The offline optimizer is a label generator, not the deployed predictor. The online model matches its 7.0% Argoverse miss rate in the reported setting, but its minADE/minFDE are worse than the optimized selector.
- Candidate resolution has diminishing returns: 1 m sampling materially improves coverage over 3 m, while 0.5 m adds no reported gain. Map errors and out-of-lane behavior remain outside this endpoint construction.
- DenseTNT turns goal selection into a learned set-prediction problem, but its quality still depends on dense candidate coverage and the pseudo-label objective chosen for the optimizer.
