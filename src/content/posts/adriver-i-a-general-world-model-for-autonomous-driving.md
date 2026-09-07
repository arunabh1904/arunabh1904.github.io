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

> ADriver-I turns an interleaved sequence of images and low-level actions into a recurrent driving world model. A multimodal language model predicts the current speed and steering angle; a video latent-diffusion model uses that action and recent frames to synthesize the next four frames; the first generated frame is then fed back for the next control decision. The paper reports one-step action and short-horizon video metrics on nuScenes plus a private highway corpus, while showing that recursive “infinite driving” remains vulnerable to generation errors and has no route-level navigation input.

## Core Insights

### An action becomes the condition for the next observation

The paper’s key unit is an **interleaved vision-action pair**. Three historical image–action pairs and the current image go to a Vicuna-7B-1.5-based MLLM. The action is written as text-like fields for speed and steer angle; the model predicts the current action autoregressively. That representation gives the language model a direct control target while retaining a temporal history, instead of asking a separate planner to consume a frozen scene description.

![Figure 1 from ADriver-I: A General World Model for Autonomous Driving](/assets/images/adriver-i-a-general-world-model-for-autonomous-driving-source-figure-1.webp)
*Fig 1: The MLLM predicts the current action from the interleaved history, and the diffusion model predicts four future frames from that action and recent frames. The first generated frame closes the loop by becoming the next image input. | source: [ADriver-I, Figure 1](https://arxiv.org/abs/2311.13549)*

The diagram makes the causal claim concrete: the action is not merely predicted beside a video forecast; it is the condition that changes the forecast. If the model predicts a sharper turn, the imagined future should move accordingly, and that changed frame should influence the next action. The same loop also explains the central risk: a visually plausible but geometrically wrong frame becomes the next observation and can compound error.

![Figure 3 from ADriver-I: A General World Model for Autonomous Driving](/assets/images/adriver-i-a-general-world-model-for-autonomous-driving-source-figure-3.webp)
*Fig 2: GPT-3.5 converts consecutive speed and steering values into a motion description such as a gradual right shift or steady speed, which is used as text conditioning for the video diffusion model. | source: [ADriver-I, Figure 3](https://arxiv.org/abs/2311.13549)*

### Control numbers need different interfaces for action and video

This prompt is a practical repair for a modality mismatch. The authors found that the diffusion text encoder struggled to infer the meaning of signed steering numbers, so GPT-3.5 maps consecutive controls into common driving states. The MLLM and video model are trained separately: the MLLM uses a CLIP ViT-Large encoder, a two-layer visual adapter, and cross-entropy over three-decimal control numbers; the VDM starts from Stable Diffusion 2.1 with temporal modules and reference-video conditioning. The VDM is pretrained on about 1.4 million private samples and fine-tuned on roughly 23,000 nuScenes video samples, using eight-frame clips at $256\times512$ resolution and DDIM sampling with 50 steps.

The MLLM takes $336\times336$ images and is trained for two epochs with batch size 16, AdamW, and learning rate $2\times10^{-5}$ on eight A100 GPUs. The action encoding ablations are useful: direct absolute numbers reach 0.072 m/s speed L1 and 0.091 rad steering L1 on nuScenes, while translating numbers to English reaches 2.094 m/s and 0.536 rad. A separate precision ablation gives 0.212 m/s and 0.099 rad when numbers are rounded to zero decimal places. Predicting all intermediate actions in a multi-round conversation improves speed L1 from 0.078 to 0.072. The interface needs both numeric precision and temporal supervision; translating a control value into words can discard distinctions the action predictor needs.

### Short-horizon metrics leave recursive error growth unresolved

On nuScenes, ADriver-I reaches speed/steering L1 errors of 0.072 m/s and 0.091 rad, with threshold accuracies $A_{0.01}=0.237/0.411$, $A_{0.03}=0.398/0.575$, $A_{0.05}=0.535/0.664$, and $A_{0.07}=0.640/0.731$. On the private highway data, the corresponding L1 values are 0.035 m/s and 0.015 rad; the authors attribute the gap to the private set’s 1.4M training pairs and narrower highway distribution, versus about 23K nuScenes samples. For future generation, the four-frame condition $4F\rightarrow4F$ yields FID 5.5 and FVD 97, compared with 73.4/502.3 for DriveGAN and 52.6/452.0 for DriveDreamer under their reported input settings. These are frame-generation comparisons, not closed-loop driving safety metrics.

## High-Level Takeaways

- ADriver-I is an early demonstration of an action-conditioned visual simulator: it predicts a control, imagines its consequence, and reuses that imagined consequence.
- The one-step numbers are strong against the paper’s small constructed baselines, but the private-versus-nuScenes split shows how much the result depends on data scale and scene distribution.
- The paper’s own limitations are decisive: the MLLM and VDM are separately trained, fast control changes can produce low-quality frames, and the system lacks routing information for long-distance driving.
- A serious follow-up should report error growth under free-running rollouts, compare teacher-forced and generated-frame inputs, and measure action-conditioned geometry rather than FID/FVD alone.
