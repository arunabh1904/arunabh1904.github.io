---
title: Generation of Vectorized Maps Beyond Vehicle View
date: '2026-09-07T09:00:00.000Z'
section: paper-shorts
postSlug: generation-of-vectorized-maps-beyond-vehicle-view
legacyPath: /paper shorts/2026/09/07/generation-of-vectorized-maps-beyond-vehicle-view.html
tags:
- Autonomous Driving
- Mapping
field: BEV Perception & Mapping
summary: 2026 – Generation of Vectorized Maps Beyond Vehicle View
---

## 2026 – Generation of Vectorized Maps Beyond Vehicle View

**Paper:** [arXiv:2609.07511](https://arxiv.org/abs/2609.07511) · [Full text](https://arxiv.org/html/2609.07511v1) · [Code](https://git-autopia.car.upm-csic.es/beyondformer) · [Processed dataset](https://zenodo.org/records/22256829)

## Summary

> BeyondFormer generates lane-centerline continuations beyond an observed map instead of reconstructing only visible road geometry. On 104 held-out, simplified road samples, it reduces Chamfer distance from 12.23 to 7.08 m against a local-curvature extrapolation baseline. The result establishes a map-forecasting experiment under accurate input maps; errors remain large, intersections and roundabouts are excluded, and lane splits are a failure case. Predicted continuation should therefore be treated as a hypothesis about unseen roads.

## Core Insights

### The input is a map, not the camera evidence behind it

Online vector mapping normally converts current sensor evidence into geometry. BeyondFormer starts after that step: it assumes an accurate in-view centerline map and predicts what comes next. The experimental input uses a 70 m radius, while the target follows the driving direction out to 150 m. The prediction can extend the planning horizon, but it cannot determine which of several unobserved road layouts is actually present.

The model represents each lane segment as a cubic Bézier curve with four two-dimensional control points. This supplies a fixed-size learning object even when the original polyline has a variable number of samples. A map Transformer encodes these curves, while a separate three-layer MLP encodes the forward-facing endpoints from which continuation should begin. The endpoints are useful geometric constraints rather than another observation of the unseen road.

The source architecture shows why both inputs matter. The decoder attends to the observed map and its endpoints, then sends its latent representation to geometry and connectivity heads.

![Source Figure 3: BeyondFormer map encoder, endpoint encoder, and geometry/topology decoder](/assets/images/october-2609.07511-s3-f3.webp)
*Fig 1: The map supplies global context and the endpoint encoder identifies where continuation begins. Geometry and topology heads jointly constrain the autoregressive output. | source: [BeyondFormer, Figure 3](https://arxiv.org/html/2609.07511v1#S3.F3)*

[Open figure at full resolution](/assets/images/october-2609.07511-s3-f3.webp)

Four autoregressive steps generate successive sets of curves. Scheduled sampling gradually replaces ground-truth preceding sets with generated ones during training, exposing the decoder to its own errors. The geometry head predicts control points; the topology head embeds lanes and predicts pairwise adjacency. A subsequent nearest-neighbor endpoint-matching step improves connectivity. The paper calls this last operation “fine tuning,” but it is a geometric correction of generated samples rather than a second learned language-model training stage.

### How the map forecasting dataset is created

The source is 3DHD CityScenes: geolocalized driving sequences and vector road annotations from Hamburg. Its original partitions contain 57,510 training, 8,087 validation, and 13,061 test locations, with disjoint road sections. The authors retain **3,608/171/104** samples respectively after selecting simplified road layouts without intersections or roundabouts. This is a constrained subset of one source dataset, not an unrestricted urban mapping benchmark.

For each vehicle location, centerlines within the input radius form the observed map. Subsequent positions along the recorded trajectory supply the future map up to the target distance. Preprocessing normalizes points to vehicle position and heading, standardizes segment lengths, and converts the result into Bézier curves. Newly exposed curves are grouped into the successive sets used for autoregressive supervision. Connectivity annotations derive from the connected source lanes.

This construction makes the split boundary and target provenance clear: the network sees map fragments, and the ground truth comes from already recorded geographic structure. It does not need to infer targets from an annotator's imagined continuation. Conversely, the accurate-map assumption removes errors from camera reconstruction, localization, and upstream vectorization. Robustness to those errors remains unmeasured.

### Geometry, adjacency, and their agreement have separate losses

The geometry objective combines pointwise Huber error, accumulated curve length, logarithmic curvature, and continuity between adjacent segments. The topology head uses binary cross-entropy on adjacency. A coupling term penalizes endpoint separation for lane pairs that should connect. The topology head therefore changes geometry training even though the main output remains curves.

The reported model has approximately 8M parameters, six encoder and six decoder blocks, eight attention heads, and width 128. Training uses one H200, batch size 64, AdamW, initial learning rate $10^{-4}$, 2.5% warmup, cosine scheduling, and dropout 0.1 for 10,000 epochs. The unusually long schedule is reported as epochs, not steps. Curves are sampled at 100 points for the geometric losses. Overall geometry, topology, and coupling weights are 1, 6, and 5; the geometry subweights are 1, 5, 350, and 50 for point, length, curvature, and continuity terms.

Those weights are part of a particular data and coordinate contract. Copying the architecture while changing map scale, curve sampling, or loss normalization would not reproduce the same optimization problem.

### A better long-range continuation can still be worse nearby

The baseline extends each curve using its endpoint tangent and local curvature. Evaluation compares discretized predictions with ground truth using pointwise RMSE, bidirectional Chamfer distance, and order-sensitive Fréchet distance, all in meters.

| Method | RMSE ↓ | Chamfer distance ↓ | Fréchet distance ↓ |
| --- | ---: | ---: | ---: |
| Curvature extrapolation | 8.78 ± 2.16 | 12.23 ± 4.97 | 10.80 ± 2.79 |
| BeyondFormer | 7.83 ± 3.66 | 7.08 ± 4.94 | 7.48 ± 2.95 |

The per-step comparison is more revealing than the average: first-step RMSE is **5.90 m** for BeyondFormer versus **3.22 m** for the baseline, while fourth-step RMSE is **9.06 m** versus **12.72 m**. Learning helps the more distant continuation, but the simpler geometric extension is better at the first step. Reported deviations across samples should not be read as uncertainty across independently trained models.

The qualitative figure exposes the unresolved topology problem. BeyondFormer follows some curvature changes the baseline misses, yet can generate cluttered, nonparallel lanes; both methods miss road splits.

![Source Figure 5: BeyondFormer and curvature extrapolation against ground-truth map continuations](/assets/images/october-2609.07511-s4-f5.webp)
*Fig 2: The learned continuation can follow longer-range curvature while retaining substantial geometric and topological errors. Lane splits remain a visible failure. | source: [BeyondFormer, Figure 5](https://arxiv.org/html/2609.07511v1#S4.F5)*

[Open figure at full resolution](/assets/images/october-2609.07511-s4-f5.webp)

Adding the endpoint encoder reduces the component-ablation RMSE from 17.62 to 14.63 m; adding the topology head reduces it to 8.99 m; endpoint correction reaches 7.83 m. The final correction improves RMSE and Chamfer distance but slightly worsens Fréchet distance from 7.25 to 7.48 m. On an RTX 4060 at batch size one, inference averages 15.3 ± 1.2 ms with 221 MB peak GPU memory. Fast inference does not resolve the uncertainty of unseen topology.

My deployment test would corrupt the input map and evaluate intersections, splits, and geographically distinct roads before allowing forecasts to constrain a vehicle plan. The paper does not report calibrated multiple hypotheses or downstream driving safety. Those are the missing pieces between useful map extrapolation and reliable long-range planning.

## High-Level Takeaways

- BeyondFormer changes the task from observing road geometry to forecasting unseen vector structure from an accurate existing map.
- Its benchmark is constructed from disjoint Hamburg road sections, then restricted to simple layouts; that selection is a central part of the result.
- Endpoint conditioning and topology supervision account for large ablation improvements, but topology itself remains unreliable in difficult continuations.
- The learned model wins at longer horizons while losing the first-step RMSE comparison, so aggregate accuracy should not replace horizon-specific inspection.
- The next decisive test is robust, uncertainty-aware prediction from imperfect input maps on complex and geographically held-out roads.
