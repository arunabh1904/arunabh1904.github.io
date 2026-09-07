---
title: 'S4-Driver: Scalable Self-Supervised Driving Multimodal Large Language Model with Spatio-Temporal Visual Representation'
date: '2025-05-30T02:20:14.000Z'
section: paper-shorts
postSlug: s4-driver-scalable-self-supervised-driving-multimodal-large-language-model-with-spatio-temporal-visual-representation
legacyPath: /paper shorts/2025/05/30/s4-driver-scalable-self-supervised-driving-multimodal-large-language-model-with-spatio-temporal-visual-representation.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – S4-Driver: Scalable Self-Supervised Driving Multimodal Large Language Model with Spatio-Temporal Visual Representation"
---
## 2025 – S4-Driver

**arXiv:** [2505.24139](https://arxiv.org/abs/2505.24139)

**Project:** [S4-Driver](https://s4-driver.github.io/)

## Summary

> S4-Driver turns multi-view, multi-frame features from a PaLI-3 5B multimodal model into a sparse 3D volume before predicting trajectories. It freezes the ViT visual encoder, learns the volume and planning interface from future trajectories, and avoids object, map, and motion annotations in the intermediate stack. On WOMD-Planning-ADE, its reported ADE@5 is 0.693 m and behavior-balanced ADE is 0.928; a WOMD-pretrained variant reaches 0.655 and 0.830. On nuScenes, the corresponding average ADE is 0.38 m, or 0.31 m with the paper's extra pretraining.

## Core Insights

### Make geometry the interface between vision and planning

S4-Driver's central choice is to stop treating image tokens as the planner's native space. For every voxel around the ego vehicle, the model projects its 3D location into each camera, bilinearly samples the corresponding feature maps, averages the valid views, and adds a positional embedding. A learned low-dimensional gate then scores the volume; only the top 6,000 locations are retained by default and the rest are represented by a shared vacant feature. This is sparse 3D reasoning without an occupancy label—the gate is learned because the future trajectory loss rewards useful locations.

The default temporal input adds one historical frame at -0.5 seconds. Ego-motion compensation aligns the historical sparse volume before its features are concatenated and projected. Inside the multimodal encoder, the model replaces a generic relative bias with 32 bin-wise distance bins for each spatial axis, so a local front/back relation is distinguishable from a merely similar token distance. The overview below is best read as an interface diagram: image features become a compact geometric memory, and language-model tokens then attend to that memory before producing actions.

![Figure 2 from S4-Driver: Scalable Self-Supervised Driving Multimodal Large Language Model with Spatio-Temporal Visual Representation](/assets/images/s4-driver-scalable-self-supervised-driving-multimodal-large-language-model-with-spatio-temporal-visual-representation-paper-figure.webp)
*Fig 1: Multi-view features are lifted into a sparse spatio-temporal volume, combined with ego context, and decoded into a meta-decision and future waypoints. | source: [S4-Driver: Scalable Self-Supervised Driving Multimodal Large Language Model with Spatio-Temporal Visual Representation, Figure 2](https://arxiv.org/abs/2505.24139)*

### Let a coarse maneuver condition the numerical plan

The planner first predicts one of four meta-decisions—keep stationary, keep speed, accelerate, or decelerate—then autoregressively emits floating-point waypoints. WOMD uses a five-second horizon divided into two 2.5-second stages; nuScenes uses a three-second horizon in one stage. The meta-decision is a heuristic derived from future motion rather than an independently observed label. For the behavior-balanced evaluation, the paper derives it from an 8-second ground-truth future and excludes “stop” from the input command to avoid future-information leakage; the waypoint target remains the future trajectory. The hierarchy is therefore useful as a planning scaffold, while “self-supervised” refers to avoiding intermediate object and map annotations rather than removing all target-derived supervision.

The second figure makes the input side of the training contract concrete: it shows the high-level behavior command, historical ego trajectory, velocity, and acceleration as text, followed by the request for a five-second trajectory. The source's paired target response contains the meta-decision and future waypoints, but the retained crop keeps the conditioning prompt legible rather than implying that the MLLM discovers a driving objective without labels. At test time, the authors also draw multiple nucleus samples and average their waypoints. Sixteen samples give 0.693/0.928 ADE/bADE on WOMD, better than greedy decoding (0.728/0.986) and four-sample beam search (0.739/0.997); weighted averaging is worse (0.747/1.005). The averaging is a practical way to temper the model's overconfidence on straight and stop cases.

![Figure 9 from S4-Driver: Scalable Self-Supervised Driving Multimodal Large Language Model with Spatio-Temporal Visual Representation](/assets/images/s4-driver-scalable-self-supervised-driving-multimodal-large-language-model-with-spatio-temporal-visual-representation-source-figure-9.webp)
*Fig 2: Retained crop of the source prompt: the behavior command and historical ego states condition the requested five-second trajectory; the full source figure pairs this input with the target response. | source: [S4-Driver: Scalable Self-Supervised Driving Multimodal Large Language Model with Spatio-Temporal Visual Representation, Figure 9](https://arxiv.org/abs/2505.24139)*

### The balanced metric exposes where the improvement lives

On nuScenes, S4-Driver reports ADE at 1/2/3 seconds of 0.16/0.34/0.63 m, averaging 0.38; the internally pretrained row reaches 0.13/0.28/0.51, averaging 0.31. On WOMD, vanilla PaLI-3 reaches 0.798 ADE@5 and 1.069 bADE, while S4-Driver reaches 0.693 and 0.928. The bADE change matters because more than 70% of the data are straight or stop, so an ordinary average can hide weak turns. The paper's all-eight-camera configuration also reaches 0.732/0.985, versus 0.765/1.036 with only the front camera, and bin-wise 3D relative bias improves over no bias (0.732/0.985 versus 0.750/1.005).

## High-Level Takeaways

- S4-Driver's main contribution is the sparse spatio-temporal volume: it gives a frozen visual backbone a geometric carrier for planning.
- The meta-decision is a useful coarse prior, while nucleus averaging turns repeated trajectory samples into a more stable action estimate.
- Behavior-balanced ADE shows that gains extend beyond the majority straight/stop cases, especially when the camera configuration and relative geometry are controlled.
- “Self-supervised” here means no intermediate object or map labels; the waypoint target and the evaluation command still depend on future trajectory data, and the paper does not report a closed-loop safety score.
