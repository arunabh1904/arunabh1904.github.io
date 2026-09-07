---
title: 'RLDG: Robotic Generalist Policy Distillation via Reinforcement Learning'
date: '2024-12-13T00:00:00.000Z'
section: paper-shorts
postSlug: rldg-robotic-generalist-policy-distillation-via-reinforcement-learning
legacyPath: /paper shorts/2024/12/13/rldg-robotic-generalist-policy-distillation-via-reinforcement-learning.html
tags:
  - Robotics
  - Distillation
field: 'Robot Post-Training & Evaluation'
summary: "2024 – RLDG: Robotic Generalist Policy Distillation via Reinforcement Learning"
---

**arXiv:** [2412.09858](https://arxiv.org/abs/2412.09858)

**Project:** [generalist-distillation.github.io](https://generalist-distillation.github.io/)

## Summary

> RLDG uses reinforcement learning where it is easiest to make precise and supervised fine-tuning where it is easiest to preserve generality. Task-specific RL specialists solve contact-rich subtasks, their successful rollouts become training data, and a generalist such as OpenVLA or Octo distills those actions. The resulting policy can improve the bottleneck skill without forcing the foundation model through an unstable direct-RL update.

## Core Insights

![RLDG workflow training specialist reinforcement-learning policies collecting their rollouts and distilling them into a generalist robot policy](/assets/images/rldg-robotic-generalist-policy-distillation-via-reinforcement-learning-paper-figure.png)
*Fig 1: The specialist and generalist have different jobs: RL optimizes a narrow reward, while supervised distillation transfers the resulting state-action distribution into one policy that can handle several tasks. | source: [RLDG, Figure 1](https://arxiv.org/abs/2412.09858)*

### The data boundary is the method

RLDG first trains separate vision-based policies with HIL-SERL, an intervention-enabled RL system built on RLPD. After convergence, each specialist is rolled out to construct a balanced dataset. Connector Insertion uses separate USB, Ethernet, and VGA specialists, with equal episodes per connector; the resulting generalist is tested zero-shot on Type-C, HDMI, DisplayPort, and 3-pin XLR. For FMB Assembly, RL is used only for the precision-critical insertion segment and human demonstrations provide grasping and transport data. Each generalist is fine-tuned on the combined state-action data with its native supervised objective: cross-entropy over discretized action bins for OpenVLA, and diffusion training for Octo.

This boundary lets the reward be task-specific without making the final policy task-specific. The setup is a Franka Panda with a wrist-mounted RealSense D405. A 1 kHz impedance controller executes 6D end-effector delta-pose commands; data collection, RL, and Octo run at 10 Hz, while OpenVLA runs at 4 Hz. OpenVLA is the 7B model pretrained on 970k Open X-Embodiment demonstrations and discretizes each action dimension into 256 bins. Octo uses a continuous diffusion head. Both generalists are fine-tuned from pretrained checkpoints using only wrist-camera images, so the comparison tests the source of the training data under a matched observation interface.

### RL data is better at the contact boundary

![Figure 4 from RLDG: generalist success rates on seen and unseen manipulation tasks](/assets/images/rldg-robotic-generalist-policy-distillation-via-reinforcement-learning-source-figure-4.webp)
*Fig 2: RL-generated rollouts improve both OpenVLA and Octo over human-demonstration fine-tuning on seen tasks and on held-out connectors or scenes. | source: [RLDG, Figure 4](https://arxiv.org/abs/2412.09858)*

The paired comparison keeps the task, observation and action spaces, training configuration, and number of successful episodes fixed; only the data source changes. On precise FMB Insertion and Connector Insertion, OpenVLA gains 33 and 23 percentage points over its human-data counterpart, while Octo gains 10 and 37 points. For Pick and Place, OpenVLA moves from 16/20 to 19/20 successes and Octo from 1/20 to 4/20. The improvement transfers: OpenVLA is more than twice as successful on unseen Connector Insertion, and Octo rises from 0/20 to 4/20 on unseen Pick and Place.

The multi-stage result is the most useful deployment pattern. Combining RL data for FMB insertion with human data for the rest of FMB Assembly gives OpenVLA 20/20 successes versus 12/20 for the human-only version. RLDG therefore does not require replacing every demonstration; it can concentrate expensive RL on the phase where alignment or contact is the bottleneck.

![Figure 6 from RLDG: cycle time for specialist and distilled policies](/assets/images/rldg-robotic-generalist-policy-distillation-via-reinforcement-learning-source-figure-6.webp)
*Fig 3: Cycle time compares RL specialists, human-data generalists, and RLDG generalists; missing bars indicate a policy was not trained for that whole task or recorded no successes. | source: [RLDG, Figure 6](https://arxiv.org/abs/2412.09858)*

### The gain is mostly action quality, with a real trade-off

The scaling experiment separates two protocols. On the seen VGA connector, OpenVLA with RLDG reaches 100% with 45 RL episodes, compared with 300 human demonstrations. On unseen Type-C, it reaches 100% with 150 RL rollouts from the VGA, USB-A, and Ethernet specialists; human demonstrations plateau around 90% even at 900 episodes. The authors’ mixed-data experiment separates state coverage from action quality: relabeling human states with RL actions improves fine-tuning by more than 50% over fully human data at 25, 50, and 75 trajectories, but pure RL data remains best. The action visualization explains why: near the insertion point, RL actions cluster toward the correct lower-left motion, while human actions cluster near the center and only weakly point toward it.

That precision is not free. The RL specialist optimizes success and speed, and its Pick and Place behavior sometimes releases the object too early after clearing the bowl edge. A distilled generalist inherits the better alignment distribution but can also inherit this timing error. RLDG assumes well-designed rewards and enough specialist coverage; it does not remove the need to validate the data before distilling it.

## High-Level Takeaways

- RLDG moves RL from the final model update into a data-generation role, preserving a generalist’s multi-task interface while concentrating optimization on a narrow contact problem.
- The controlled evidence is strong on precise tasks: OpenVLA improves by 33 points on FMB Insertion, by 23 on Connector Insertion, and reaches 20/20 on the mixed-data FMB Assembly task.
- RL actions, rather than only RL states, account for much of the gain: they are concentrated toward the corrective motion needed for insertion instead of the diffuse human action distribution.
- Specialists can optimize the wrong operational preference. Faster releases improved the RL objective but caused premature drops, so reward design and trajectory filtering remain part of the method.
- The approach scales in engineering cost with the number of specialist rewards and tasks. Its evidence covers a small real-robot suite and does not show whether one specialist dataset can safely update a generalist across unrelated embodiments.
