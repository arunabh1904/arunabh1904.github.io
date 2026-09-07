---
title: 'Octo: An Open-Source Generalist Robot Policy'
date: '2024-05-20T00:00:00.000Z'
section: paper-shorts
postSlug: octo-an-open-source-generalist-robot-policy
legacyPath: /paper shorts/2024/05/20/octo-an-open-source-generalist-robot-policy.html
tags:
  - Robotics
  - Generalist Policies
field: 'Vision-Language-Action & Robotics'
summary: "2024 – Octo: An Open-Source Generalist Robot Policy"
---

## 2024 – An Open-Source Generalist Robot Policy

**arXiv:** [2405.12213](https://arxiv.org/abs/2405.12213)

**Project:** [octo-models.github.io](https://octo-models.github.io/)

## Summary

> Octo treats the input and output interface as part of a robot foundation model. A transformer policy pretrained on 800,000 Open X-Embodiment trajectories accepts language or goal-image tasks, predicts continuous action chunks with a diffusion head, and can be adapted to new cameras, proprioception, action spaces, and robots.

## Core Insights

### Flexibility lives in the token layout

A policy trained on one robot normally fixes the number and order of camera streams, task inputs, and action outputs. Octo instead turns each modality into a token block. Language is encoded with T5-base; images and goal images pass through a shallow convolutional patch encoder; position embeddings retain the sequence layout. The transformer produces learned readout tokens that summarize the task and observation prefix, and a small action head maps those readouts to a future action chunk.

The attention mask is blockwise. An observation block can attend to task tokens and observations from the same or earlier time, while a readout token can read the prefix without feeding information back into the observation blocks. Missing modalities are masked. A new camera stream, proprioceptive input, or action head can therefore be added while reusing the transformer initialization. This architectural flexibility does not mean the reported adaptation freezes the backbone: the experiments fine-tune the full model, which outperformed freezing subsets of its weights.

This is more than a convenient API. It gives the backbone a stable sequence interface while allowing the physical meaning of the final action to change. The price is that zero padding, sensor quality, and action conventions remain part of the learned distribution.

### Diffusion preserves continuous alternatives

Octo's action head samples Gaussian noise and denoises it into an action chunk over several steps using the transformer readout as conditioning. Only one transformer forward pass is needed for an action prediction; the denoising loop runs inside the lightweight head. The model thus combines the broad context of a transformer with the ability to represent multiple plausible continuous trajectories.

The training target is a DDPM-style denoising objective. The paper compares this choice with discretized actions and an MSE head on a controlled WidowX evaluation. The diffusion head reaches 83% mean success across two language-conditioned and two goal-conditioned tasks, versus 18% for discretized action prediction and 35% for MSE. The authors connect the gap to multimodality and precision: an MSE head averages incompatible actions, while a discretized head gives up continuous resolution.

### The source figures show what is being transferred

![Octo zero-shot and fine-tuning evaluation setups across nine robot systems](/assets/images/octo-an-open-source-generalist-robot-policy-source-figure-2.webp)
*Fig 1: The evaluation spans third-person and wrist-camera setups, single and dual arms, long-horizon tasks, and precise insertion, separating out-of-box control from adaptation to new interfaces. | source: [Octo: An Open-Source Generalist Robot Policy, Figure 4](https://arxiv.org/abs/2405.12213)*

The source figure is a useful warning against treating “nine robots” as one benchmark. Zero-shot tasks use setups represented in pretraining and delta end-effector actions. Fine-tuning tasks introduce force-torque observations, joint-position control, new embodiments, or bimanual coordination. A policy can perform well on the first test while still needing a new action head or sensor adapter for the second.

![Octo zero-shot success as model size increases on UR5 and WidowX tasks](/assets/images/octo-an-open-source-generalist-robot-policy-source-figure-4.webp)
*Fig 2: Zero-shot success rises from Octo-Tiny through Octo-Small to Octo-Base on both UR5 and WidowX, showing a capacity effect under the same task interfaces. | source: [Octo: An Open-Source Generalist Robot Policy, Figure 6](https://arxiv.org/abs/2405.12213)*

The scaling plot is small but informative. Octo-Tiny, Octo-Small, and Octo-Base contain about 10M, 27M, and 93M parameters. The larger model is more robust to initial scene configuration and less likely to make an early grasp attempt. Scaling helps perception and policy capacity, but the plot is measured on one language-conditioned task per robot with ten trials, so it is not a universal scaling law.

### Pretraining and adaptation are separate tests

Octo is pretrained on a curated mixture of 25 Open X-Embodiment datasets containing 800,000 trajectories. The selection removes datasets without image streams or delta end-effector actions, repetitive sources, low-resolution images, and excessively narrow tasks. More diverse datasets receive double weight, while repetitive datasets are down-weighted. Missing camera channels are zero-padded and gripper conventions are aligned so that +1 means open and 0 means closed. Only 27% of the mixture has wrist-camera information and 56% has language annotations, shortages the authors suggest are likely contributors to later modality-specific failures.

The released checkpoints are Octo-Small at 27M parameters and Octo-Base at 93M. The ViT-B-sized model was trained for 300,000 steps with batch size 2,048 on a TPU v4-128 pod in about 14 hours. Fine-tuning updates the full model with the same diffusion objective for 50,000 steps; a run with about 100 target trajectories takes under five hours on one 24 GB NVIDIA A5000. The authors found this full-model recipe stronger than freezing parts of the pretrained network.

For data-efficient fine-tuning, Octo averages 72% across six reported domains: 70% Berkeley Peg Insertion, 75% Stanford Coffee, 50% CMU Baking, 60% Berkeley Pick-Up, 100% Berkeley Coke, and 80% Berkeley Bimanual. Each domain uses about 100 target demonstrations and 20 trials. These numbers are adaptation results, not zero-shot results. The zero-shot comparison across WidowX, UR5, and RT-1 Robot reports a 29% higher success rate than RT-1-X and similar performance to RT-2-X on the tested WidowX and RT-1 Robot tasks. It uses two language tasks per robot and ten trials per task with varied initial conditions. RT-1-X and RT-2-X use a more restricted 350K-episode OXE subset, so the comparison also changes the training mixture.

Goal images also matter. On the WidowX tasks, goal-image conditioning gives a reported 25% higher success rate than language conditioning, consistent with a goal frame carrying spatial information that a short instruction leaves implicit. The gain is a property of the conditioning signal and training distribution, not proof that goal images solve planning.

### What the ablations reveal

The data-mixture ablation compares the full 25-dataset mixture with the 11-dataset RT-X mixture and with Bridge data from one robot. On the controlled WidowX suite, the aggregate means are 83% for Octo-Small, 60% for the RT-X mix, and 43% for the single-robot mixture. The architecture ablation gives 70% for a ResNet-50 plus transformer, while the objective ablations give 18% for discretized actions and 35% for MSE. These comparisons keep the evaluation at two language and two goal tasks over 40 trials, making the direction of the effects clear even though the components are not a complete factorial study.

The zero-shot generalization table makes the boundary sharper. On WidowX, Octo-Small averages 85% for in-distribution tasks, 80% for novel objects, 40% for a novel environment, and 5% for novel skills. The two novel-skill tasks are “flip cup on its side” at 10% and “put block in slot” at 0%. Broad data and flexible inputs transfer object and scene variation more readily than a skill whose motion is absent from the embodiment's training data.

### Limits of a generalist interface

Octo's demonstrations are mostly optimal robot trajectories, so imitation does not teach the policy how to recover from arbitrary failures. The model often benefits from using only a third-person camera rather than combining it with a wrist camera, consistent with the small wrist-camera fraction in pretraining. Language-conditioned performance also trails goal-image conditioning, consistent with only 56% of the data carrying language labels.

The practical lesson is to measure adaptation cost beside zero-shot success. A modular token layout and a diffusion readout make new interfaces cheap to test, but they cannot replace missing sensor coverage, contact dynamics, or task-specific demonstrations. Octo is valuable precisely because the checkpoints, training code, data loaders, and evaluation recipe make those tradeoffs inspectable.

## High-Level Takeaways

- Blockwise attention and readout tokens let the backbone survive changes to cameras, proprioception, task specification, and action space.
- A diffusion action head retains continuous, multimodal trajectories and outperforms the tested MSE and discretized alternatives.
- The 800,000-trajectory mixture improves initialization, while the strongest reported six-domain numbers require about 100 target demonstrations per domain.
- Scaling from 10M to 93M parameters improves the tested zero-shot tasks, but unseen skills remain nearly unsolved.
- A flexible interface lowers adaptation cost; it does not supply the missing data or contact behavior.
