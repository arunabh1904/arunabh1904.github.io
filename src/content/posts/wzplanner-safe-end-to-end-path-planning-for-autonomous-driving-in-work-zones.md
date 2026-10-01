---
title: 'WZPlanner: Safe End-to-End Path Planning for Autonomous Driving in Work Zones'
date: '2026-09-16T09:00:00.000Z'
section: paper-shorts
postSlug: wzplanner-safe-end-to-end-path-planning-for-autonomous-driving-in-work-zones
legacyPath: /paper shorts/2026/09/16/wzplanner-safe-end-to-end-path-planning-for-autonomous-driving-in-work-zones.html
tags: ["Autonomous Driving", "Research"]
field: Motion Forecasting & Planning
summary: '2026 – WZPlanner: Safe End-to-End Path Planning for Autonomous Driving in Work Zones'
---

## 2026 – WZPlanner: Safe End-to-End Path Planning for Autonomous Driving in Work Zones

**Paper:** [arXiv:2609.19393](https://arxiv.org/abs/2609.19393) · [Full text and supplement](https://arxiv.org/html/2609.19393v1) · [Code and dataset](https://github.com/Nishad-Sahu/WZPlanner)

## Summary

> WZPlanner couples a work-zone data-generation pipeline with compact models that predict temporary boundaries and feasible driving paths. WorkZonePlan contains 149,478 synthetic and 5,178 real frames, plus 228 weather-route evaluations from 76 CARLA layouts. BoundaryFormer++ improves average driving score over the tested external baselines, but its complete driving stack includes a separate LiDAR speed governor even in the “camera” variant. The paper's full-route audit, held-out towns, and annotation provenance are essential to interpreting its safety claims.

## Core Insights

### Construct the temporary corridor instead of assuming the map is current

Painted lanes and stored maps can disagree with cones, barriers, and detours. WZPlanner makes those temporary boundaries explicit supervision. Its **WAVE** pipeline begins with an operator placing work-zone assets in CARLA, selecting lane boundaries, and manually driving a representative path. These choices produce global-frame points for lane boundaries, work-zone boundaries, and valid trajectory options.

The generator replays the scenario with synchronized RGB, LiDAR, and calibration under different lighting, rain, and object choices. Annotation code transforms the geometry into the ego frame and checks projected boundary points against image depth. A depth disagreement beyond a threshold, illustrated as 25 cm, removes likely occluded points. Surviving geometry becomes third-order polynomial curves with valid ranges and objectness labels; rendered overlays support human inspection. Where two local routes are valid, the annotation retains both rather than forcing a single path.

![Source Figure 3: WAVE synthetic and real-data annotation pipeline](/assets/images/october-2609.19393-s2-f3.webp)
*Fig 1: Human scenario design and trajectory recording supply the geometry that WAVE transforms into calibrated training labels. Synthetic variation expands coverage without making those labels independently observed human driving decisions. | source: [WZPlanner, Figure 3](https://arxiv.org/html/2609.19393v1#S2.F3)*

[Open figure at full resolution](/assets/images/october-2609.19393-s2-f3.webp)

The real subset is **5,178 existing WorkZone3D frames**, not a new independently captured collection. WAVE retains its camera, six VLP-16 LiDAR units, GPS/vehicle state, calibration, and boxes, then adds lane/work-zone curves and trajectories through global-frame alignment and GUI correction. A 152-frame consistency audit reports 94.1% without flags, but it is a single-rater, AI-assisted visual screening. That supports apparent overlay consistency, not centimetre-accurate independent 3D ground truth.

### Frame counts, split units, and route counts answer different questions

The original Town01/Town10 corpus contains 137,749 frames. Another **11,729 frames from Town02–05** are reserved for out-of-distribution evaluation, producing the 149,478-frame synthetic release. BF++ uses a fixed 125,626-frame training set and 2,791-frame validation set; 9,332 remaining in-distribution frames are retained in the release but unused in that comparison. Earlier BF experiments use random 90/10 splits and should not be mistaken for held-out-town generalization.

The closed-loop benchmark has 12 layouts in each of Towns01–05 and 16 in Town10. Replaying these **76 base layouts under clear day, foggy dusk, and stormy night** yields 228 routes per method. These are correlated weather variants of the same layouts, not 228 independent scenarios. Training images span nine lighting/rain combinations and nine work-zone object types, while the closed-loop protocol uses the three named conditions.

Real-world BF++ results hold out one nighttime capture sequence of 773 frames and fine-tune on the other three sequences. This is more informative than a random frame split, but the main table reports that particular held-out sequence rather than an average across all four possible folds.

### A dedicated path decoder matters more than interchangeable slots

Original BoundaryFormer uses a pretrained Swin backbone and a two-layer token projection into a shared latent space. Learnable slots cross-attend to image tokens; heads predict cubic coefficients, valid range, type, and objectness. Hungarian matching assigns predicted entities to labels. The shared-slot version also treats trajectories as entities. The stronger distinct-planner version combines mean-pooled image context with objectness-weighted boundary features in a dedicated MLP that emits multiple trajectory hypotheses and probabilities.

![Source Figure 5: BoundaryFormer slots and separate trajectory decoder](/assets/images/october-2609.19393-s5-f5.webp)
*Fig 2: The original architecture tests whether trajectories should share boundary slots or receive a dedicated global decoder. This figure depicts BF; the later BF++ changes the backbone, geometry representation, and typed queries. | source: [WZPlanner, Figure 5](https://arxiv.org/html/2609.19393v1#S5.F5)*

[Open figure at full resolution](/assets/images/october-2609.19393-s5-f5.webp)

BF++ develops that distinction with a ConvNeXt-Tiny backbone. Images resize to **768 × 432**; stride-16/32 features pass through 1 × 1 convolution and GroupNorm to width 256 and combine on a 48 × 27 grid. Calibrated flat-ground coordinates provide positional information. A three-layer Transformer uses **10 lane, six work-zone, and one trajectory query**. The trajectory query emits two independently parameterized, confidence-scored centerlines. Curves use 38 anchors from 1.5 to 100 m, visibility and valid-range predictions, and two refinement passes sampling projected image evidence.

The camera+LiDAR model rasterizes a synchronized sweep into normalized depth, height, and hit-mask channels. Four strided convolution blocks produce a width-256 feature map, added through a **zero-initialized per-channel gate**. This keeps the initial fused model equivalent to its camera parent. These are geometric feature projections and sensor fusion, not a VLM visual-language projector; neither BF nor BF++ uses a language backbone.

### Training and deployment have separate geometric and control components

BF++ matches predictions within each entity type, preventing lane and work-zone identities from swapping. Its loss weights point regression by five, visibility by one, objectness by two, range by one-half, and adds refinement supervision; work-zone point errors receive another 2.5× weight.

For LiDAR adaptation, epochs 0–2 train the new branch and gate at $2\times10^{-4}$ with camera weights frozen. Epochs 3–11 unfreeze the decoder, heads, and refiner at one-quarter that rate while keeping the backbone frozen. Training uses AdamW, weight decay 0.05, batch size 12, gradient clipping 1, Gaussian image noise 0.02, and no geometric augmentation. Real-data adaptation fine-tunes the full network for eight epochs with backbone rate 0.1× the head rate. Training uses one RTX 5090; the supplement does not fully enumerate the initial camera model's optimization schedule.

The navigation command selects the highest-confidence compatible path **after both options have been decoded**. It does not expose the lowest-error option using ground truth. Pure pursuit tracks the selected path. Both BF++ variants additionally use a separate **32-channel utility LiDAR** for a stopping-distance speed governor, distinct from the learned model's optional 64-channel perception LiDAR. Thus “BF++-Camera” describes learned perception input, not a camera-only driving stack. External baselines keep their own controllers, so cross-family results do not isolate neural architecture from control design.

### Use the completed audit and retain the outcomes that do not improve

| Model | Full 228-route driving score | Successes / 228 | Collision events |
| --- | --- | --- | --- |
| BF++-Camera+LiDAR | 64.5 | 76 (33.3%) | 47 |
| BF++-Camera | 62.8 | 69 (30.3%) | 47 |
| SimLingo | 60.5 | 87 (38.2%) | 71 |
| TransFuser++ | 26.0 | 3 (1.3%) | 261 |

The main paper's 211-route table reflects the common completed subset at evaluation freeze. The supplement reruns interrupted simulator/harness failures to obtain all 912 method–route endpoints. Driving failures remain scored; only invalid harness runs are rerun. The completed table preserves BF++'s average driving-score lead while **SimLingo retains the highest full-route success rate**. A layout-blocked bootstrap on the separately identified 208-route diagnostic subset has confidence intervals crossing zero; it does not establish a statistically significant superiority claim.

Open-loop results expose another limit: camera work-zone F1 at 0.5 m falls from **0.715 in distribution to 0.174 on held-out towns**, and LiDAR does not improve the aggregate OOD result. The model is fast—9.6/11.3 ms for camera/fused variants on RTX 5070 Ti, including preprocessing and curve decoding—but generalization remains the harder problem. Matched-curve localization error and oracle best-of-two error must also remain distinct from missed detections and the route-selected path actually executed.

## High-Level Takeaways

- WAVE's labels originate in operator-designed geometry, replay, calibration, and curve fitting; synthetic scale does not replace label-provenance analysis.
- Temporary boundaries benefit from typed geometric queries and a dedicated trajectory pathway.
- The camera/fusion comparison shares a LiDAR speed governor; external-baseline comparisons also include controller and training differences.
- Completed 228-route results improve average driving score and collision counts, while SimLingo wins on perfect-route success.
- Held-out-town detection degradation and layout-level uncertainty qualify the paper's broad safety framing.
