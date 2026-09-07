---
title: 'GeoWAM: Visual Geometry World Action Models for Autonomous Driving'
date: '2026-08-24T09:00:00.000Z'
section: paper-shorts
postSlug: geowam-visual-geometry-world-action-models-for-autonomous-driving
legacyPath: /paper shorts/2026/08/24/geowam-visual-geometry-world-action-models-for-autonomous-driving.html
tags: [Autonomous Driving]
field: 'Autonomous Driving: VLA & Planning'
summary: '2026 – GeoWAM: Visual Geometry World Action Models for Autonomous Driving'
---

## 2026 – GeoWAM: Visual Geometry World Action Models for Autonomous Driving

**Paper:** [arXiv:2608.23486](https://arxiv.org/abs/2608.23486) · [Full text](https://arxiv.org/html/2608.23486v1)

## Summary

> GeoWAM predicts future 3D point maps and uses their latent geometry to produce a driving trajectory. It reports 90.2 EPDMS on NAVSIM v2 navtest and 36.6 on the harder two-stage navhard protocol. The useful architectural choice is to forecast metric structure directly instead of generating RGB and reconstructing geometry afterward. The evidence supports that pipeline in the reported settings, but does not isolate geometry forecasting from its pretraining, supervision, and decoder choices.

## Core Insights

### Forecast the information the planner needs to preserve

A future image must describe appearance as well as physical layout. For planning, a painted texture may matter less than the position of a road edge or another vehicle. GeoWAM changes the prediction target: historical multiview images become geometric and ego-motion tokens, which a decoder forecasts into future geometry. A point-map head can turn those tokens into a 3D position for each future image pixel.

The encoder and point head initialize from DVGT-2. Three historical frames predict eight future frames at 2 Hz, covering four seconds. During geometry pretraining, the model samples two to eight camera views from sequences spanning seven driving datasets. Planning fine-tuning uses eight views on NAVSIM. This is a substantial geometric pretraining recipe, not simply a different output head attached to an otherwise matched video generator.

The cropped source diagram isolates the memory, future-geometry branch, and future-pose branch. Follow the geometry tokens first: they can produce a dense point map and also serve as context for the pose decoder. The trajectory head then reads predicted and historical ego tokens to regress one path. No image generation, trajectory anchors, or iterative action sampling sit on this path.

![GeoWAM source Figure 2: predicted future geometry conditions an ego-pose decoder and a deterministic trajectory head](/assets/images/geowam-source-figure-2.png)
*Fig 1: Future geometry has two uses: dense point-map prediction and context for ego-motion prediction. The action branch reads geometric features rather than reconstructing a future video first. Cropped to the prediction modules. | source: [GeoWAM, Figure 2](https://arxiv.org/abs/2608.23486)*

This is an inverse-dynamics-like arrangement: predicted scene evolution helps infer ego motion. It is not an action-conditioned simulator that can answer what would happen under any proposed maneuver. The distinction matters when deciding whether the model can support counterfactual planning or only generate a trajectory consistent with its forecast.

### Two stop-gradient boundaries have different jobs

During training, future images are encoded to create target features. Those targets are detached, and the forecasting branch only receives the historical images. This prevents the future observation from becoming an input shortcut. Feature alignment is combined with point regression, confidence-aware regression, and surface-normal consistency; current-frame geometry supervision also anchors the representation.

A second stop-gradient sits between predicted future geometry and the action branch. Trajectory loss cannot reshape the future-geometry tokens through that connection. The action decoder must use the forecast supplied to it rather than modify that forecast solely to make trajectory regression easier. This does not freeze the entire shared encoder: the action branch also reads historical memory, and geometry objectives remain active during joint fine-tuning.

That boundary trades some task-specific adaptation for preservation of a geometric training target. A useful follow-up would compare it with allowing action gradients through, measuring both point-map accuracy and planning. The reported paper does not supply a controlled removal of that stop-gradient connection.

### Direct geometry avoids an extra error-producing conversion

For the geometry comparison, video baselines first generate future RGB, then DVGT reconstructs point maps from those images. Their error can arise in either generation or reconstruction. GeoWAM predicts geometry directly. Table 1 reports the following averages across eight future steps on nuScenes validation.

| Pipeline | Absolute relative depth error, lower is better | Threshold accuracy, higher is better |
| --- | ---: | ---: |
| Epona video generation + DVGT | 0.274 | 0.655 |
| Cosmos 3 video generation + DVGT | 0.376 | 0.503 |
| VGGT-World | 0.325 | 0.544 |
| GeoWAM | 0.257 | 0.754 |

The direct geometry pipeline improves the aggregate measures, but the one-second threshold accuracy is higher for Epona plus DVGT: 0.732 versus 0.708. That exception is useful because it prevents an average from becoming “better at every future prediction.” These comparisons also differ in training and architecture; they establish pipeline performance, not a matched-cost proof that predicting pixels is intrinsically inferior.

### The harder evaluation changes the size of the apparent gain

On NAVSIM v2 navtest, GeoWAM scores 90.2 EPDMS against the listed DVGT-2 baseline's 89.6. On navhard, the corresponding values are 36.6 and 31.7. Navhard's two-stage protocol renders a new observation from the predicted ego pose, so the second planning decision sees a consequence of the first. It exposes a kind of recovery demand absent from a one-shot evaluation, while remaining a bounded approximation to extended interactive driving.

The per-component results make that difficulty visible. GeoWAM's no-at-fault-collision measure falls from 97.7 in the first stage to 80.4 in the second, and lane keeping from 96.0 to 49.9. A higher aggregate score than competing methods therefore coexists with substantial failures after the observation changes. The two benchmark aggregates should not be compared as if they measured the same scenario distribution and rollout protocol.

My read is that direct future geometry is a useful alternative when planning depends on metric structure and image synthesis is an expensive intermediate. The unresolved decision is how much of the benefit requires forecasting rather than stronger geometry pretraining alone. A matched-backbone, matched-data comparison against a current-geometry planner, with and without the future loss and gradient boundary, would isolate that question more cleanly.

## High-Level Takeaways

- GeoWAM predicts future geometric features and point maps, then regresses one trajectory without rendering future RGB or sampling candidate actions.
- Future-image targets train forecasting but are absent at inference. The separate action-to-geometry stop-gradient limits one path of task-driven representation change.
- Geometry evaluation favors the direct pipeline on average, but video-to-geometry baselines carry an additional reconstruction stage and are not matched-cost controls.
- The navhard result tests replanning after a synthesized observation change; the large second-stage failures remain relevant despite the aggregate improvement.
- Isolating future prediction from geometry pretraining is the next experiment needed before paying its full training and inference cost.
