---
title: 'ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving'
date: '2025-06-09T03:14:04.000Z'
section: paper-shorts
postSlug: recogdrive-a-reinforced-cognitive-framework-for-end-to-end-autonomous-driving
legacyPath: /paper shorts/2025/06/09/recogdrive-a-reinforced-cognitive-framework-for-end-to-end-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving"
---
## 2025 – ReCogDrive

**arXiv:** [2506.08052](https://arxiv.org/abs/2506.08052)

## Summary

> ReCogDrive separates driving cognition from the action interface. InternVL3 is trained on a twelve-dataset, quality-controlled mixture of perception, dynamic-understanding, planning, and reasoning examples; its hidden states condition a diffusion planner that emits continuous trajectories. Diffusion Group Relative Policy Optimization (DiffGRPO) then scores complete denoising trajectories in NAVSIM and updates the planner with group-relative advantages. On NAVSIM navtest, the camera-only model reports 90.8 PDMS, compared with 88.1 for camera-and-LiDAR DiffusionDrive and 88.3 for WoTE; on Bench2Drive it reports a 71.36 driving score.

## Core Insights

### Put cognition and coordinates on different sides of the interface

The paper treats language as a useful place to represent driving concepts, but a poor place to serialize every waypoint. ReCogDrive uses InternVL3—InternViT-300M plus a Qwen2.5 language model with dynamic image resolution—to interpret the scene. A hierarchical data pipeline first generates examples at four cognitive levels, refines and augments open driving data, then applies automated quality control. The resulting VLM hidden states are passed to a diffusion transformer rather than decoded as a long string of floating-point numbers.

The diffusion planner uses self-attention for relations among waypoints and cross-attention to the VLM states. Historical trajectories are concatenated as context, ego state enters through adaptive layer normalization, and pooled semantic features give the planner a compact scene-level signal. The visual summary below is useful because it shows the handoff as two coupled paths: the language model supplies driving context, while the denoiser repeatedly turns that context into a physically continuous plan.

![Figure 1 from ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving](/assets/images/recogdrive-a-reinforced-cognitive-framework-for-end-to-end-autonomous-driving-source-figure-1.webp)
*Fig 1: Overview of ReCogDrive, with driving priors entering a diffusion denoising process that produces continuous trajectories. | source: [ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving, Figure 1](https://arxiv.org/abs/2506.08052)*

### Treat denoising as the policy, not just as a decoder

Training is staged around that interface. The VLM is frozen while the planner learns by DDPM imitation for 200 epochs; a ten-epoch DiffGRPO stage then samples trajectories and evaluates them in the NAVSIM simulator; the VLM receives three epochs of supervised fine-tuning. DiffGRPO regards the whole denoising chain as an internal Markov decision process. The reward is NAVSIM's PDMS, which combines collision, drivable-area compliance, time-to-collision, comfort, and progress terms. Group-standardized advantages compare sampled plans for the same scene, while a behavior-cloning term keeps the diffusion policy anchored to demonstrated trajectories. A discount of 0.6 weights later denoising decisions more heavily, since early steps still contain high noise.

![Figure 4 from ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving](/assets/images/recogdrive-a-reinforced-cognitive-framework-for-end-to-end-autonomous-driving-source-figure-4.webp)
*Fig 2: Imitation learning fits trajectories to demonstrations, whereas DiffGRPO samples multiple denoising trajectories, scores them in the simulator, and learns from their relative quality. | source: [ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving, Figure 4](https://arxiv.org/abs/2506.08052)*

That design explains why the paper's ablation is more informative than the headline comparison. Starting from a trajectory-only model, driving pretraining raises PDMS from 82.4 to 84.1, the diffusion planner raises it to 86.5, and DiffGRPO raises it to 90.8. The final camera-only NAVSIM row is NC 97.9, DAC 97.3, TTC 94.9, comfort 100, EP 87.3, and PDMS 90.8. The planner also generates a trajectory in about 0.075 seconds versus 0.5839 seconds for the text baseline, a reported 7.8× speedup.

### The result is tied to the evaluation contract

The gains are measured on NAVSIM's 1,192-scene navtrain/136-scene navtest setup and on 220 short CARLA routes in Bench2Drive. ReCogDrive reaches 45.45% scenario success and a 71.36 driving score on Bench2Drive, but the two benchmarks probe different things. DriveBench VQA is also strong (56.71 average GPT score), while adding chain-of-thought does not improve NAVSIM (90.7 versus 90.8 without it). That last result is a useful boundary: the paper's cognition is carried mainly by the trained representations and planner interface, not by requiring a visible verbal chain at inference.

## High-Level Takeaways

- ReCogDrive makes the VLM a source of driving priors and lets a diffusion model own the continuous action geometry.
- The strongest ablation jump comes from DiffGRPO on simulator-scored trajectories, after the data pipeline and diffusion interface are already in place.
- Its 0.075-second trajectory generation and camera-only 90.8 PDMS are reported under NAVSIM's closed-loop protocol; the CARLA result is a separate short-route test.
- The optional chain-of-thought result is revealing: richer textual reasoning is not automatically the missing ingredient when the latent driving representation and action interface already carry the useful information.
