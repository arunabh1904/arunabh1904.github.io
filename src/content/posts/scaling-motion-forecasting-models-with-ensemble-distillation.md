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

**arXiv:** [2404.03843](https://arxiv.org/abs/2404.03843)

**Project:** [Waymo research page](https://waymo.com/research/scaling-motion-forecasting-models-with-ensemble-distillation/)

## Summary

A motion-forecasting ensemble can improve both precision and coverage, but running many models onboard is expensive. This paper separates training compute from serving compute: independently trained Wayformer teachers form a multimodal ensemble, and a smaller student learns from the ensemble distribution as well as the logged ground truth.

The hard part is that trajectory modes have no natural correspondence across independently trained teachers. Averaging the first mode from one model with the first mode from another is meaningless. The paper therefore aggregates teacher mixtures with non-maximal suppression (NMS), samples from the resulting distribution, and trains the student with a negative log-likelihood distillation loss plus the ordinary ground-truth loss.

## Core Insights

### Preserve multimodality before distilling it

Each teacher emits a Gaussian mixture over a trajectory, with mixture weights, means, and per-timestep covariance. The ensemble is a weighted sum of the teacher distributions. A temperature `τ` flattens each teacher's mixture weights so lower-probability modes are not discarded too early, and a variance scale controls whether samples use the full Gaussian or just its means.

![Ensemble distillation's inference-compute tradeoff](/assets/images/scaling-motion-forecasting-models-with-ensemble-distillation-paper-figure.png)
*Fig 1: On WOMD, teacher ensembles improve metrics as inference FLOPs grow; distilled students occupy a much cheaper operating region. | source: [Scaling Motion Forecasting Models with Ensemble Distillation, Figure 4](https://arxiv.org/abs/2404.03843)*

Because teacher modes do not align, NMS first greedily selects modes that cover the most total likelihood and then refines them with a k-means-like update. This produces a smaller mixture that still represents distinct futures. The same aggregation is used to reduce the student's output when it emits more modes than the benchmark allows.

### Distillation transfers evidence, not only a softened label

The student minimizes
`L_total = L_distill + w_gt L_gt`.
The distillation term is the negative log likelihood of samples drawn from the aggregated teacher distribution. For the WOMD experiments, the authors use a 20-teacher ensemble, `τ=8`, `w_gt=0.4`, and variance scale `w_var=0.5`; to make label generation affordable, teacher sampling sets the variance scale to zero and uses the mixture means weighted by their probabilities. When teacher and student output counts match, a simpler bijective mode mapping can also supervise the mixture weights.

![The ensemble-to-student training pipeline](/assets/images/scaling-motion-forecasting-models-with-ensemble-distillation-source-figure-3.webp)
*Fig 2: NMS merges unmatched teacher modes into a compact teacher distribution; the student learns from its samples alongside the ground-truth loss. | source: [Scaling Motion Forecasting Models with Ensemble Distillation, Figure 3](https://arxiv.org/abs/2404.03843)*

The experimental protocol uses Wayformer early fusion with hidden size 256, two encoder layers, eight decoder layers, 64 teacher modes, AdamW, batch size 256, and one million steps. WOMD predicts up to eight agents from one second of history over an eight-second future; Argoverse 2 predicts one focal agent from five seconds of history over six seconds. All reported metrics use six final trajectories, and the WOMD training set duplicates classified U-turn and left/right-turn examples at 5%.

### The student keeps much of the ensemble gain at a different cost point

On the WOMD leaderboard table, the single Wayformer baseline reports minFDE 1.126, minADE 0.545, miss rate 0.123, mAP 0.412, and relative FLOPs 1×. The distilled student reports 1.122/0.546/0.117/0.438 with 1.36× relative FLOPs. The 20-model ensemble reports 1.137/0.549/0.118/0.446 at 20× relative FLOPs. The student therefore improves miss rate and mAP over the single model while approaching the ensemble's coverage at a much smaller serving budget.

![Ensemble scaling on Argoverse](/assets/images/scaling-motion-forecasting-models-with-ensemble-distillation-source-figure-5.webp)
*Fig 3: Argoverse ensembles and distilled students follow the same quality-versus-relative-FLOPs pattern, with students moving the curve toward deployable compute. | source: [Scaling Motion Forecasting Models with Ensemble Distillation, Figure 5](https://arxiv.org/abs/2404.03843)*

The temperature ablation makes the precision-coverage tradeoff explicit: increasing `τ` spreads probability mass toward lower-ranked trajectories, and `τ=8` is the reported optimum for the soft-mAP/minADE balance. Distillation can therefore improve the student even when it is smaller than the teacher, but it inherits the ensemble's blind spots and depends on expensive teacher inference during training.

## High-Level Takeaways

- Ensemble distillation creates a train-serve asymmetry: pay for diversity while generating labels, then deploy one student.
- NMS is part of the method, not a cosmetic post-process. It solves the mode correspondence problem and determines which teacher futures the student can see.
- In the reported WOMD table, the student moves from 1× to 1.36× relative FLOPs while improving mAP from 0.412 to 0.438; the 20-model ensemble reaches 0.446 at 20×. Those are matched six-mode metrics, so the compute comparison is interpretable.
- Temperature, variance scaling, and the ground-truth loss control whether the student preserves rare modes or concentrates on common trajectories. A student cannot recover futures absent from the teacher mixture.
