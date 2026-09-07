---
title: 'Towards Zero-Shot Transfer Across Embodiments For Driving VLAs'
date: '2026-09-02T09:00:00.000Z'
section: paper-shorts
postSlug: towards-zero-shot-transfer-across-embodiments-for-driving-vlas
legacyPath: /paper shorts/2026/09/02/towards-zero-shot-transfer-across-embodiments-for-driving-vlas.html
tags: [Autonomous Driving]
field: 'Autonomous Driving: VLA & Planning'
summary: '2026 – Towards Zero-Shot Transfer Across Embodiments For Driving VLAs'
---

## 2026 – Towards Zero-Shot Transfer Across Embodiments For Driving VLAs

**Paper:** [arXiv:2609.02341](https://arxiv.org/abs/2609.02341) · [Full text](https://arxiv.org/html/2609.02341v1) · [Code](https://github.com/caiocj1/ad-vla)

## Summary

> BEV-Forcing teaches a driving VLA to reconstruct ground-plane vehicle occupancy from its image features, using a specialist model as the teacher. This helps transfer between camera rigs when training diversity is limited. The more revealing result is where the benefit disappears: after adding NAVSIM and nuScenes-QA to Waymo training, BEV-Forcing lowers held-out Waymo Rater Feedback Score from 7.939 to 7.902. An auxiliary geometry objective should be evaluated against a diverse-data baseline, not only against a single-dataset model.

## Core Insights

### Recognizing the same car is easier than recovering its position from a new rig

Changing camera height, orientation, or field of view changes how image coordinates relate to a future path. A VLA may retain the semantic fact that an image contains a car while losing the geometric relationship needed to steer around it. The paper studies this cross-embodiment transfer separately from recognizing unfamiliar objects with the same cameras.

The policy starts from Qwen3.5-2B and produces text describing two-dimensional waypoints at 1 Hz. Cubic splines resample those waypoints to the evaluation frequency. Training uses rank-16 LoRA for one epoch on four A100 GPUs. Keeping the action representation simple lets the experiments concentrate on which spatial information the backbone learns, though it also limits how broadly we should generalize the results to other trajectory decoders.

### A deliberately small head makes the backbone carry the geometry

BEV-Forcing adds learned queries corresponding to positions in a ground-plane grid. Those queries attend to projected image hidden states from the VLM's final layer. A linear prediction layer turns each query into a vehicle-occupancy logit. The target comes from SimpleBEV, a specialist camera-based model, so the auxiliary task can be used on datasets without annotated occupancy maps.

Follow the two routes into the loss in the source diagram. The VLM route must reconstruct occupancy from the features also used for planning. The teacher route supplies a fixed target. Language and trajectory supervision still use next-token prediction; weighted binary cross-entropy adds pressure to expose object layout inside those image features.

![BEV-Forcing source Figure 2: learned spatial queries read VLM image embeddings and predict a teacher occupancy grid](/assets/images/bev-forcing-source-figure-2.png)
*Fig 1: Spatial queries reconstruct vehicle occupancy from image hidden states. A small prediction head encourages the shared features to expose geometry instead of delegating reconstruction to a powerful decoder. | source: [BEV-Forcing, Figure 2](https://arxiv.org/abs/2609.02341)*

A large auxiliary decoder could solve the training task without making the representation equally useful to the planner. The head-capacity ablation supports this concern: in the paper's smaller comparison, replacing cross-attention with a full transformer block raises Physical AI average displacement error from 1.725 to 1.805 and final displacement error from 5.073 to 5.308. These values belong to that ablation and should not be mixed with the main benchmark runs. They support the chosen small head; they do not directly measure how spatial information is distributed across features.

Once training ends, the auxiliary head is removed. The additional cost lies in teacher target generation and training, rather than in an extra deployed perception module. Unlike [Qwen-Drive's explicit perception output](/paper%20shorts/2026/08/31/qwen-drive-1-0-an-initial-step-towards-a-vision-language-foundation-model-for-autonomous-driving.html), this head exists to shape a representation and need not remain available for inspection at inference.

### More data changes the answer to whether the auxiliary loss helps

The experiments progressively add camera rigs through Waymo, NAVSIM, and nuScenes-QA, then evaluate transfer to Physical AI and KITScenes LongTail. For Waymo-only training, BEV-Forcing reduces Physical AI average displacement error by 10.1%. On KITScenes test, the Waymo-plus-NAVSIM model improves its multi-maneuver score from 4.62 to 5.15. Those are useful gains under limited diversity.

The full mixture is the essential counterexample. Table 2 reports the following held-out Waymo results; larger Rater Feedback Score is better, while smaller average displacement error is better.

| Training mixture | BEV-Forcing | Rater Feedback Score | Average displacement error |
| --- | --- | ---: | ---: |
| Waymo | No | 7.873 | 2.898 |
| Waymo | Yes | 7.902 | 2.891 |
| Waymo + NAVSIM + nuScenes-QA | No | 7.939 | 2.833 |
| Waymo + NAVSIM + nuScenes-QA | Yes | 7.902 | 2.938 |

The same auxiliary loss that helps the narrow recipe regresses both measures in the broader one. This is stronger evidence than merely seeing a smaller improvement. The authors also report a reversal on KITScenes validation with the full mixture, although its internally reproduced score should remain separate from the official test results.

Training diversity and teacher geometry may supply partly overlapping information. That is a plausible interpretation, not a general law that more data eliminates geometric supervision. Dataset additions also change task mixture and sample volume; the experiment does not isolate camera-rig count at a fixed total budget.

### Language retention is part of the geometry experiment

Adding planning datasets alone can damage the backbone's language behavior. On nuScenes-QA, the unadapted model scores 29.2%, compared with 11.1% after Waymo-only training and 7.0% after Waymo-plus-NAVSIM training without BEV-Forcing. Adding nuScenes-QA, about 10% of the training samples, restores task accuracy to 60.2%. That last score reflects direct training on the question format and is not a broad measurement of preserved language ability.

The interaction matters because a failed multi-dataset run can look like a geometry problem while also suffering from lost instruction following. BEV supervision, linguistic co-training, image augmentation, and calibration inputs are different interventions. The paper finds that augmentation and calibration help zero-shot transfer more when paired with BEV-Forcing, but the gains remain modest in the richer mixture.

My takeaway is to test the cheap auxiliary head early when only one or two rigs are available, then retest it as the dataset grows. A matched-update comparison with equal sample counts, held-out rigs, and a stronger teacher would help separate regularization from additional supervision. The reported results concern trajectory prediction and human-rated candidate matching; they do not establish closed-loop recovery or safe transfer to an unseen vehicle.

## High-Level Takeaways

- BEV-Forcing transfers vehicle-layout supervision into VLM image features and removes its auxiliary head at inference.
- A smaller head works better in the reported capacity ablation, consistent with making the shared representation do more of the spatial work.
- The gain depends on the training mixture: the richest recipe reverses the held-out Waymo improvement. Recheck architectural additions as data diversity changes.
- Planning-only adaptation can erode language behavior, so geometry comparisons should control the linguistic co-training mixture too.
- Zero-shot camera-rig transfer under open-loop metrics is a useful test of representation quality. Interactive driving on the new embodiment remains a separate test.
