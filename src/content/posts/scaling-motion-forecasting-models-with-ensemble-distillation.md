---
title: 'Scaling Motion Forecasting Models with Ensemble Distillation'
date: '2024-04-05T00:00:00.000Z'
section: paper-shorts
postSlug: scaling-motion-forecasting-models-with-ensemble-distillation
legacyPath: /paper shorts/2024/04/05/scaling-motion-forecasting-models-with-ensemble-distillation.html
tags:
  - Other
field: 'Motion Forecasting & Planning'
summary: "2024 – Scaling Motion Forecasting Models with Ensemble Distillation"
---
## 2024 – Ensemble Distillation

**arXiv:** [2404.03843](https://arxiv.org/abs/2404.03843)

**Project:** [Waymo research page](https://waymo.com/research/scaling-motion-forecasting-models-with-ensemble-distillation/)

### Method and reported result

Motion forecasting gets better when you ensemble many strong models, but autonomous vehicles cannot usually afford to run a large ensemble in real time. This paper uses the ensemble as a teacher and trains smaller student models to keep much of the accuracy at lower compute.

## Summary

> The work is a deployment-minded scaling paper. It treats accuracy, latency, and onboard compute as coupled constraints instead of assuming the best leaderboard model can be used directly.

## Core Insights

The paper first builds large ensembles of optimized single motion-forecasting models and shows that the ensembles improve generalization. It then develops a generalized distillation framework that transfers those ensemble predictions into smaller student models. The task focus is motion forecasting on real-world autonomous-driving data.

The evidence includes strong Waymo Open Motion Dataset and Argoverse leaderboard performance for the ensembles, followed by student models that retain high performance at a fraction of the compute cost. The key caveat is that distillation quality depends on the teacher distribution: if the ensemble misses rare futures or encodes systematic bias, the student can inherit those limits.

![Figure 4 from Scaling Motion Forecasting Models with Ensemble Distillation showing metrics versus inference FLOPs](/assets/images/scaling-motion-forecasting-models-with-ensemble-distillation-paper-figure.png)
*Figure 4 shows the deployment tradeoff: larger ensembles improve metrics with more FLOPs, while distilled students aim to preserve much of that gain at lower inference cost. From the [ensemble distillation paper](https://arxiv.org/abs/2404.03843). source: [ensemble distillation paper](https://arxiv.org/abs/2404.03843)*

![Figure 3 from Scaling Motion Forecasting Models with Ensemble Distillation](/assets/images/scaling-motion-forecasting-models-with-ensemble-distillation-source-figure-3.webp)
*Figure 3 Fig. 3 : Illustration of the ensemble distillation pipeline. A set of teachers and a NMS that outputs trajectories form the ensemble. The student is trained with a groundtruth loss ( ) and a distillation loss ( ). source: [Scaling Motion Forecasting Models with Ensemble Distillation](https://arxiv.org/abs/2404.03843)*

![Figure 5 from Scaling Motion Forecasting Models with Ensemble Distillation](/assets/images/scaling-motion-forecasting-models-with-ensemble-distillation-source-figure-5.webp)
*Figure 5 Fig. 5 : Scaling ensemble result using the Argoverse dataset. The Ensemble models are represented in orange , with ensemble size linearly related to the FLOPs. Distilled student models are represented in green. source: [Scaling Motion Forecasting Models with Ensemble Distillation](https://arxiv.org/abs/2404.03843)*


_while distilled students aim to preserve much of that gain at lower inference cost. source: [ensemble distillation paper](https://arxiv.org/abs/2404.03843)


**What to look at:**
- Ensembles are used as a temporary training tool, not as the final deployed system.
- Distillation targets the practical gap between leaderboard accuracy and onboard compute.
- The paper is about scaling under constraints rather than inventing a new motion representation.

### Reported evidence

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Teacher | Large ensemble of optimized forecasters | Raises the accuracy ceiling. |
| Student | Distilled smaller model | Brings ensemble gains closer to deployment cost. |
| Benchmarks | WOMD and Argoverse leaderboards | Covers widely used motion-forecasting evaluations. |

## High-Level Takeaways

- This paper informs whether serving constraints should cap teacher quality or whether a costly forecasting ensemble can be used only during training and distilled into one onboard model. The unit is a scene with multiple teacher trajectory distributions; the student learns both logged futures and the ensemble's softened multimodal predictions.
- The result establishes a train-serve asymmetry: ensemble diversity can improve a student that cannot afford ensemble inference. The missing comparison matches total teacher-training compute against a single larger teacher and tests which teacher diversity actually transfers. At 10× ensemble size, teacher storage, inference for label generation, and correlated errors dominate. Distillation would fail as the allocation strategy if a directly trained student or single teacher matched leaderboard and calibration metrics at lower end-to-end cost.
- It showed a practical way to use scaling and ensembles even when the production model must stay small.
- For driving forecasts, the best model family may be a training-time ensemble plus a deployment-time student.
