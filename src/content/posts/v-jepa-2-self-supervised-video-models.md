---
title: 'V-JEPA 2: Self-Supervised Video Models for Understanding and Planning'
date: '2025-06-11T00:00:00.000Z'
section: paper-shorts
postSlug: v-jepa-2-self-supervised-video-models
legacyPath: /paper shorts/2025/06/11/v-jepa-2-self-supervised-video-models.html
tags: [Video Models, World Models]
field: 'Video & Interactive World Models'
summary: '2025 – V-JEPA 2: Self-Supervised Video Models for Understanding and Planning'
---

**arXiv:** [2506.09985](https://arxiv.org/abs/2506.09985)<br>
**Blog:** [V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks)

## Summary

> V-JEPA 2 separates broad visual learning from action supervision. It learns a video representation by predicting masked latents on more than one million hours of internet video and images, then adds a 300M action-conditioned predictor using under 62 hours of DROID robot video. The resulting system reaches 77.3 top-1 on Something-Something v2, 39.7 action Recall@5 on EK100, and zero-shot image-goal manipulation on new Franka arms.

## Core Insights

### The first stage learns predictable state, not pixels

![V-JEPA 2 overview from web-scale video pretraining to video QA, probes, and robot planning](/assets/images/v-jepa-2-paper-figure-1.png)
*Fig 1: V-JEPA 2 pretrains a video encoder on internet video and images, then branches into language alignment, attentive probes, and action-conditioned robot planning. | source: [V-JEPA 2, Figure 1](https://arxiv.org/abs/2506.09985)*

V-JEPA 2 uses a joint-embedding predictive objective: the encoder sees a masked view of a video, the predictor fills the missing representation, and an EMA encoder supplies the target. The loss is applied only to masked patches. That choice deliberately favors predictable structure over pixel-level detail. Grass texture or individual leaves may be unpredictable and irrelevant to planning, while a moving object’s trajectory is worth preserving.

The pretraining set contains more than one million hours of video and one million images. The encoder scales from 300M to 1B parameters, uses 3D rotary position embeddings, and is trained with a progressive resolution schedule so the model can move beyond short 16-frame clips. The resulting representation is not itself a language model and does not receive action labels; language alignment and action conditioning are downstream stages.

### Passive video and robot interaction have different jobs

The action-conditioned extension, V-JEPA 2-AC, freezes the pretrained encoder and learns a roughly 300M-parameter block-causal transformer. It receives per-frame visual features, seven-dimensional end-effector states, and seven-dimensional state differences as actions. Training uses teacher forcing plus a two-step rollout loss, which asks the predictor to tolerate feeding its own representation back into the next step.

![V-JEPA 2 scaling recipe across data, model size, training, and resolution](/assets/images/v-jepa-2-self-supervised-video-models-source-figure-3.webp)
*Fig 2: The report’s ViT-L ablation shows the average classification accuracy gained by data scaling, model scaling, longer training, and higher resolution. | source: [V-JEPA 2, Figure 3](https://arxiv.org/abs/2506.09985)*

The robot data are “unlabeled” only in a specific sense: the authors discard rewards, task names, and success flags, but use raw video plus end-effector state signals. They sample 4-second, 16-frame clips at 4 FPS and train on under 62 hours after dropping shorter videos. At deployment, a goal image is encoded, candidate action sequences are rolled forward in latent space, and the Cross-Entropy Method selects the sequence whose predicted representation is closest to the goal. Only the first action is executed before replanning.

This separation clarifies what the web data contribute. Passive video supplies a state space and a predictive prior; the DROID trajectories teach how control inputs move through that space. Temporal coherence alone would not tell the planner whether a commanded displacement moves the arm toward a goal.

### The evidence is strong, and the control loop remains bounded

With the common frozen-probe protocol, V-JEPA 2 ViT-g reaches 75.3 SSv2 accuracy at 256px and the higher-resolution ViT-g384 reaches 77.3. On EK100, the 1B ViT-g384 reaches 39.7 action Recall@5 using a 32-frame context ending one second before the action. The aligned 8B video-language model reports 84.0 PerceptionTest accuracy, 44.5 MVP paired accuracy, and 76.9 TempCompass multi-choice accuracy, but these are separate alignment experiments rather than direct outputs of the encoder.

In robot deployment, V-JEPA 2-AC runs zero-shot on two Franka arms in labs absent from DROID. The average table reports 100% reach, 65% cup grasp, 25% box grasp, 75% cup reach-with-object, 75% box reach-with-object, and 80%/65% cup/box pick-and-place across ten trials per skill and lab. Against Cosmos, V-JEPA 2-AC takes about 16 seconds per action with 800 CEM samples versus roughly four minutes with 80 samples, while retaining better object-interaction results.

The limits are part of the result. The predictor must infer a robot action axis from an uncalibrated monocular camera; the authors manually searched for camera placements that worked. Autoregressive error accumulates over longer horizons, and the manipulation tasks use visual subgoals rather than open-ended language goals. EK100 itself is a 100-hour kitchen dataset with a closed action vocabulary and a one-second anticipation setting, so the 39.7 score is not evidence of unconstrained future-action prediction.

## High-Level Takeaways

- V-JEPA 2’s contribution is the division of labor: web video learns a useful latent state, while a small interaction dataset teaches how actions move through it.
- Predicting latents instead of pixels makes large-scale pretraining practical and emphasizes structure that is useful for recognition and planning.
- The robot result is a real closed-loop test, with two unseen labs and image-goal control, but it still relies on 62 hours of robot state data, visual subgoals, and carefully chosen camera geometry.
- Strong frozen-probe and video-QA scores show transferable representations; they do not by themselves establish controllability, long-horizon stability, or generalization beyond kitchen actions and tabletop arms.
