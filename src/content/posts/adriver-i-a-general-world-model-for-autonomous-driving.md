---
title: 'ADriver-I: A General World Model for Autonomous Driving'
date: '2023-11-22T17:44:29.000Z'
section: paper-shorts
postSlug: adriver-i-a-general-world-model-for-autonomous-driving
legacyPath: /paper shorts/2023/11/22/adriver-i-a-general-world-model-for-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2023 – ADriver-I: A General World Model for Autonomous Driving"
---
## 2023 – ADriver-I

**arXiv:** [2311.13549](https://arxiv.org/abs/2311.13549)

## Summary

> ADriver-I turns an interleaved sequence of visual features and control signals into a driving world model. The model predicts the current control, conditions future-frame generation on that control and the history, then feeds the imagined frame back into another control step. It is therefore a joint action-and-observation rollout rather than a planner that predicts a trajectory once. The paper evaluates on nuScenes and a private dataset, but its abstract does not report a closed-loop metric or the horizon over which the recursive rollout remains reliable.

## Core Insights

The crucial representation is the interleaved vision-action pair. It makes control part of the temporal context that an MLLM and diffusion model can process, then lets generated controls influence future visual predictions. In principle, that structure exposes a useful consistency question: do actions lead to plausible visual consequences? In practice, rollout quality can degrade through both control error and image-generation error.

![ADriver-I: A General World Model for Autonomous Driving source figure: Overview of of our ADriver-I framework.](/assets/images/adriver-i-a-general-world-model-for-autonomous-driving-paper-figure.webp)
*Overview of of our ADriver-I framework. source: [ADriver-I: A General World Model for Autonomous Driving](https://arxiv.org/abs/2311.13549)*

![Figure 3 from ADriver-I: A General World Model for Autonomous Driving](/assets/images/adriver-i-a-general-world-model-for-autonomous-driving-source-figure-3.webp)
*Figure 3 An example of conversation used for guiding GPT3.5 to generate the corresponding motion prompt. The control signals of the historical frame and current frame are used as inputs, while GPT3.5 outputs potential motion states. source: [ADriver-I: A General World Model for Autonomous Driving](https://arxiv.org/abs/2311.13549)*

![Figure 1 from ADriver-I: A General World Model for Autonomous Driving](/assets/images/adriver-i-a-general-world-model-for-autonomous-driving-source-figure-1.webp)
*Figure 1 Overview of of our ADriver-I framework. It takes the historical interleaved vision-action pairs { , } and current visual token as inputs. The multi-modal large language model (MLLM) reasons out the control signal of current frame. The predicted action is further used as the condition prior of video latent diffusion model (VDM) to generate the future four frames. The predicted next frame is selected and further input to the MLLM to produce the control signal. source: [ADriver-I: A General World Model for Autonomous Driving](https://arxiv.org/abs/2311.13549)*


The paper compares ADriver-I with constructed baselines and describes the result as favorable. The abstract gives neither the visual-token format, diffusion objective, action parameterization, data scale, nor a teacher-forced-versus-free-rollout ablation. A driving team would need those controls before treating a compelling imagined video as evidence that the world model is a safe planning model.

## High-Level Takeaways

- ADriver-I makes an interleaved vision-action transition the training unit, so action prediction and future observation prediction share a rollout interface.
- Its reported evaluation covers nuScenes and private driving data, but the abstract does not establish long-horizon closed-loop stability.
- The decisive experiment would hold the perception backbone and action head fixed while comparing direct planning, teacher-forced world-model planning, and free recursive rollout under the same safety budget.
