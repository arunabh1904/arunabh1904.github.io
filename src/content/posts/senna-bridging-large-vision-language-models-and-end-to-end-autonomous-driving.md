---
title: 'SENNA: Bridging Large Vision-Language Models and End-to-End Autonomous Driving'
date: '2024-10-29T00:00:00.000Z'
section: paper-shorts
postSlug: senna-bridging-large-vision-language-models-and-end-to-end-autonomous-driving
legacyPath: /paper shorts/2024/10/01/senna-bridging-large-vision-language-models-and-end-to-end-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2024 – SENNA: Bridging Large Vision-Language Models and End-to-End Autonomous Driving"
---
## 2024 – SENNA

**arXiv:** [2410.22313](https://arxiv.org/abs/2410.22313)

## Summary

> SENNA gives a large vision-language model a compact driving interface: it describes the scene, predicts a lateral/longitudinal meta-action, and hands that decision to a trajectory planner. Senna-VLM uses a CLIP ViT-L/14 encoder, a learned query adapter, and Vicuna-7B; Senna-E2E then combines the meta-action with scene features and predicts a continuous plan. On DriveX, SENNA reports 71.21% decision accuracy; on the DriveX-pretrained nuScenes row it reports average planning L2 of 0.43 m and average collision rate of 0.12%.

## Core Insights

### Let language choose the maneuver, not every control point

SENNA's VLM consumes surround-view images and produces structured answers about the scene, traffic signals, vulnerable road users, motion intent, and planning. Its most important output is a meta-action: one of Left, Straight, or Right crossed with Accelerate, Keep, Decelerate, or Stop. A Driving Vision Adapter uses learned image queries to compress each CLIP ViT-L/14 view before Vicuna-v1.5-7B reasons over the resulting tokens. Senna-E2E then maps the formatted decision into a learned embedding and lets a VADv2-style planner turn it into a trajectory.

The decomposition is visible in the first figure. Prior pipelines send perception features directly toward motion and planning; SENNA inserts an inspectable decision layer between scene interpretation and numeric control. That layer can make a driving intent legible, but its small vocabulary also creates a hard interface: a “turn left” embedding cannot carry every timing, gap, or recovery detail needed by the downstream planner.

![Figure 2 from SENNA: Bridging Large Vision-Language Models and End-to-End Autonomous Driving](/assets/images/senna-bridging-large-vision-language-models-and-end-to-end-autonomous-driving-paper-figure.png)
*Fig 1: Previous pipelines connect perception directly to planning, while SENNA routes visual reasoning through a meta-action before trajectory generation. | source: [SENNA: Bridging Large Vision-Language Models and End-to-End Autonomous Driving, Figure 2](https://arxiv.org/abs/2410.22313)*

### Train driving semantics from an automatic curriculum

The VLM training has three stages. Mix pretraining aligns the vision adapter with single-image instruction data; driving fine-tuning adds surround-view planning questions; planning fine-tuning teaches the meta-action questions. The labels are generated from 3D boxes, tracks, ego future, and planner state, then converted into questions and answers, so this is a planning-oriented automatic annotation pipeline rather than a collection of manually written explanations. DriveX contains one million three-second clips, with 800,000 for training and 200,000 for validation.

The planner is trained with ground-truth meta-actions but receives the VLM prediction at inference. This distinction matters: a reported trajectory error mixes planner quality with exposure to an incorrect language decision. On DriveX, SENNA reaches 71.21% decision accuracy; path F1 is 95.60 for straight, 89.37 for left, and 90.09 for right, while speed F1 is 80.18 keep, 58.83 accelerate, 61.99 decelerate, and 80.10 stop. The qualitative examples show why the intermediate text is useful: the model names the red light, nearby actors, and intended response rather than exposing only a curve.

![Figure 6 from SENNA: Bridging Large Vision-Language Models and End-to-End Autonomous Driving](/assets/images/senna-bridging-large-vision-language-models-and-end-to-end-autonomous-driving-source-figure-5.webp)
*Fig 2: Qualitative SENNA answers highlight scene details that support the predicted driving decision and explanation. | source: [SENNA: Bridging Large Vision-Language Models and End-to-End Autonomous Driving, Figure 6](https://arxiv.org/abs/2410.22313)*

### Token budget and camera coverage are part of the method

SENNA's ablations show that semantic compression has an operating point. With 32, 64, and 128 image tokens per view, DriveX accuracy rises from 69.36% to 70.42% to 71.21%; at 256 tokens it falls to 56.13%, and the 512/576-token settings collapse. Surround views also matter: front-only accuracy is 64.91%, versus 71.21% for six-view input. The resulting DriveX-pretrained nuScenes row reaches planning errors of 0.26, 0.42, and 0.61 m at 1, 2, and 3 seconds, averaging 0.43 m; collision rates are 0.05%, 0.11%, and 0.21%, averaging 0.12%.

The class distribution explains why the authors report more than overall accuracy. Their DriveX meta-action chart is dominated by common commands, so rare turns and longitudinal changes can be hidden by a single aggregate. A compact language interface is helpful only when its categories preserve the distinctions the planner needs.

![Figure 5 from SENNA: Bridging Large Vision-Language Models and End-to-End Autonomous Driving](/assets/images/senna-bridging-large-vision-language-models-and-end-to-end-autonomous-driving-source-figure-4.webp)
*Fig 3: The DriveX meta-action distribution shows the class imbalance that motivates path and speed breakdowns alongside overall accuracy. | source: [SENNA: Bridging Large Vision-Language Models and End-to-End Autonomous Driving, Figure 5](https://arxiv.org/abs/2410.22313)*

## High-Level Takeaways

- SENNA makes language a maneuver-level interface and leaves precise geometry to an end-to-end planner.
- The automatic QA curriculum supplies scene semantics at scale, while ground-truth meta-actions during planner training create a measurable inference-time exposure gap.
- The 128-token, surround-view operating point is a systems result as much as a modeling result: more visual tokens can hurt, and front-only context misses decisions that views around the ego vehicle disambiguate.
- Its reported nuScenes planning gains are meaningful under the paper's pretrained and ego-state settings, but they do not make the language decision a safety certificate.
