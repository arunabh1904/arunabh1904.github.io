---
title: 'DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving'
date: '2025-10-14T00:00:00.000Z'
section: paper-shorts
postSlug: drivevla-w0-world-models-amplify-data-scaling-law-in-autonomous-driving
legacyPath: /paper shorts/2025/10/14/drivevla-w0-world-models-amplify-data-scaling-law-in-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving"
---

## 2025 – DriveVLA-W0

**arXiv:** [2510.12796](https://arxiv.org/abs/2510.12796)

**Code:** [BraveGroup/DriveVLA-W0](https://github.com/BraveGroup/DriveVLA-W0)

## Summary

> DriveVLA-W0 argues that action labels are too sparse to make large driving VLAs use data efficiently. It adds dense future-scene supervision through either autoregressive visual-token prediction or latent diffusion, then attaches a lightweight action expert so the large representation model does not sit on the control loop. The experiments connect future-image fidelity, action decoding, and scaling behavior rather than treating them as separate tricks.

## Core Insights

DriveVLA-W0 starts from a mismatch between the input and target. A driving model sees images, language, and past actions, but action-only training compresses them into a few waypoints. The world model adds a dense self-supervised target: predict the visual scene that follows the current context, so the shared backbone also models traffic and ego motion.

### Two world-model objectives

The VQ variant uses an Emu3-style 8B backbone with discrete visual tokens. Its input interleaves language $L_t$, image tokens $V_t$, and previous action tokens $A_{t-1}$. In addition to action cross-entropy, the autoregressive objective predicts every visual token in the current frame from the preceding multimodal context, $L_{WM\text{-}AR}=-\sum_i\log P(v_i\mid S_{<V_t},v_{<i})$. The two losses are weighted together.

The ViT variant uses a Qwen2.5-VL 7B backbone with continuous visual features. It denoises the latent of the *next* image conditioned on the current visual and action features, using a noise-prediction MSE objective. Predicting the future is essential: reconstructing the current frame would reward appearance matching without requiring the representation to explain what an action will cause. The diffusion generator is bypassed during driving inference; its purpose is to shape the backbone during training, with future frames generated only for qualitative analysis.

![DriveVLA-W0 autoregressive and diffusion world-model objectives](/assets/images/drivevla-w0-world-models-amplify-data-scaling-law-in-autonomous-driving-paper-figure.png)
*Fig 1 (paper Figure 2): The VQ path predicts discrete visual tokens autoregressively, while the ViT path denoises a future-image latent from current visual/action features; both share multimodal context with the action objective. | source: [DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving](https://arxiv.org/abs/2510.12796)*

### A smaller expert keeps inference practical

The full VLA backbone is valuable for representation learning but expensive for every control update. DriveVLA-W0 pairs it with a 500M Action Expert in a Mixture-of-Experts design. The two experts form their own queries, keys, and values, concatenate them for one joint-attention operation, then route outputs back to their streams. The action expert reads the large model’s context without running the full VLA head for every generated waypoint.

The same interface compares three action decoders. A query-based expert updates learnable waypoint queries and regresses the trajectory with an L1 loss. An autoregressive expert predicts discrete action tokens, while a flow-matching expert learns a vector field from noise to continuous actions and follows it with a fixed-step ODE solver. All variants prefill the previous action features, so the comparison tests action-distribution modeling on top of a shared temporal prior.

![DriveVLA-W0 mixture-of-experts action decoders](/assets/images/drivevla-w0-world-models-amplify-data-scaling-law-in-autonomous-driving-source-figure-3.webp)
*Fig 2 (paper Figure 3): The large VLA Expert shares joint attention with a 500M Action Expert; the three panels then compare query-based, autoregressive, and flow-matching action interfaces. | source: [DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving](https://arxiv.org/abs/2510.12796)*

### Benchmark results and the scaling curve

On NAVSIM v1, Table 1 reports 88.4 PDMS for the query-based DriveVLA-W0 expert with one front camera. Multiple trajectory anchors raise the listed result to 90.2, and autoregressive best-of-six reaches 93.0. NAVSIM v2 adds direction, traffic-light, lane-keeping, and extended-comfort terms; Table 2 reports 86.1 EPDMS for the base configuration. These rows mix decoder strategies, so they show the framework’s range rather than one apples-to-apples comparison.

The main scaling test uses 70 million frames from more than one million clips, alongside 70k- and 700k-frame subsets. Table 3 shows why the extra target matters. At 70M frames, the VQ action-only baseline has 1.4829 m ADE and 4.88% collision, while VQ plus world modeling reaches 1.0563 m and 3.92%, a 28.8% ADE improvement. The ViT baseline changes from 1.1051 m and 3.59% to 1.0640 m and 3.02%, including a 15.9% collision reduction. Action-only training improves early and then saturates; future-scene supervision keeps extracting signal as the data grow.

![DriveVLA-W0 world modeling and data scaling](/assets/images/drivevla-w0-world-models-amplify-data-scaling-law-in-autonomous-driving-source-figure-1.webp)
*Fig 3 (paper Figure 1): The source plot contrasts sparse action supervision with joint visual/action supervision and shows the world-model curve continuing to improve as the in-house frame count grows. | source: [DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving](https://arxiv.org/abs/2510.12796)*

The transfer experiment in Figure 4 and Table 7 gives the proposed explanation a sharper test. A model is pretrained on NuPlan and fine-tuned on NAVSIM, where visuals are related but the action distribution shifts toward long-tail maneuvers. Action-only VQ pretraining drops PDMS from 68.7 when trained from scratch to 62.2 after pretraining. The W0-VQ model rises from 80.7 to 85.6 instead. Predicting future images encourages environment features that transfer across the action-distribution shift, rather than overfitting the source dataset’s controls.

### The best action interface changes with data scale

Table 4 reveals a useful reversal. On the 103k-frame NAVSIM training set, the query expert reaches 88.4 PDMS, ahead of flow matching at 87.2 and autoregressive decoding at 85.3. On the 70M-frame in-house set, the autoregressive expert has the lowest ADE and collision—1.0069 m and 2.95%—ahead of flow matching at 1.0362 m and 3.98% and the query expert at 1.1248 m and 4.53%. A query head is a good fit for a simple trajectory distribution, but a large and varied action manifold rewards the autoregressive decoder’s capacity and teacher-forced training. Flow matching is continuous, yet its sampling and optimization burden becomes a liability at this scale.

The representation ablations point to the same causal story. Table 5 raises PDMS from 84.1 for vision-only 6V pretraining to 85.6 for interleaved vision-action 6VA pretraining. Table 6 gives 83.3, 84.2, and 85.6 for VA, 2VA, and 6VA temporal context. In Appendix Table 8, the longer-context 6VA model also has better future-image FID (4.610 versus 9.847 for 2VA) and higher planning PDMS (85.6 versus 84.1). The action-conditioned future is doing more than adding a reconstruction head: it forces the visual representation to preserve the consequences of ego motion.

### Latency and evidence boundaries

The query-based expert reduces H200 inference latency from 117.8 ms for the full baseline to 74.3 ms and raises the paired PDMS from 85.6 to 88.4 (Section 4.5 and Appendix Figure 6). The flow expert is roughly 145 ms; autoregressive decoding is 95 ms for NAVSIM trajectories averaging 5.6 tokens and 170 ms for in-house trajectories averaging 17.8 tokens, below the 240 ms full-backbone baseline. This is the practical reason to separate representation learning from action decoding.

The evidence has two limits. Large-scale training uses a private 70M-frame dataset with 100 challenging test scenarios, and the future generator is skipped at inference. The scaling claim concerns shared representations under that distribution, not a deployed video simulator or a guarantee that more raw frames always help. Matched mixtures, held-out routes, and reactive closed-loop evaluation would test the objective beyond the paper’s action and sensor conventions.

## High-Level Takeaways

- DriveVLA-W0 adds action-conditioned future-scene prediction so a VLA receives dense supervision for the dynamics its sparse waypoint target omits.
- At 70M frames, world modeling cuts VQ ADE by 28.8% and ViT collision by 15.9% (Table 3), while the future generator is removed from the driving inference path.
- The best action decoder reverses with scale: query-based leads NAVSIM at 88.4 PDMS, while autoregressive decoding leads the 70M-frame action comparison (Table 4).
- Interleaved 6VA pretraining reaches 85.6 PDMS versus 84.1 for vision-only 6V (Table 5); private data and non-reactive evaluation still bound the scaling claim.
