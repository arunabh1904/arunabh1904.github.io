---
title: 'Xiaomi-Robotics-1: Scaling VLA Models with Real-World Trajectories'
date: '2026-07-16T00:00:00.000Z'
section: paper-shorts
postSlug: xiaomi-robotics-1-scaling-vla-with-real-world-trajectories
legacyPath: /paper shorts/2026/07/16/xiaomi-robotics-1-scaling-vla-with-real-world-trajectories.html
tags: [Vision-Language-Action, Robotics]
field: 'Vision-Language-Action & Robotics'
summary: '2026 – Xiaomi-Robotics-1: Scaling VLA Models with Real-World Trajectories'
---

**arXiv:** [2607.15330](https://arxiv.org/abs/2607.15330)

**Project:** [Xiaomi-Robotics-1](https://robotics.xiaomi.com/xiaomi-robotics-1.html)

## Summary

> Xiaomi-Robotics-1 separates two scaling problems: learning broad manipulation priors from more than 100,000 hours of UMI trajectories, then aligning those priors to robot embodiments and imperative instructions with roughly 10,000 hours of cross-embodiment data. An automatic pipeline labels state transitions with language. The paper reports 57.4% on RoboCasa365, above the previous 46.6%, and a 20.07 RoboDojo score versus 13.07.

## Core Insights

### State transitions turn UMI video into action supervision

![Figure 1 from Xiaomi-Robotics-1: Scaling VLA Models with Real-World Trajectories](/assets/images/xiaomi-robotics-1-scaling-vla-with-real-world-trajectories-source-figure-1.webp)
*Fig 1: Overview. Xiaomi-Robotics-1 combines robot trajectories, 100K hours of UMI data, and vision-language data, then measures data and model scaling before adapting to unseen environments and tasks. | source: [Xiaomi-Robotics-1, Figure 1](https://arxiv.org/abs/2607.15330)*

The scale comes from UMI handheld grippers and egocentric cameras rather than robot-only teleoperation. The corpus spans homes, commercial premises, industrial sites, offices, and outdoor settings. The authors cut trajectories into equal-length segments and use Qwen3.5-27B to caption how the grippers and interacting objects changed. A producer–consumer labeling pipeline processes the 100K-hour corpus in roughly two weeks. The resulting prompt describes a target state transition, so the model learns to move the current observation toward a language-specified outcome instead of merely imitating a task name.

### Pretraining and post-training use different contracts

The model pairs a Qwen3-VL vision-language backbone with a 16-layer DiT action decoder. The released scaling variants are 2B, 5B, and 10B total parameters (2.6B, 5.1B, and 10.5B in the paper's configuration table). Pretraining minimizes flow-matching and regression losses while retaining a vision-language next-token objective, with the language term weighted by $\lambda=0.1$ and vision-language data sampled against UMI trajectories at a 1:9 ratio.

Post-training changes the contract from state-transition descriptions to imperative robot instructions. Its roughly 10,000-hour mixture includes more than 7,200 hours of in-house mobile-manipulator and dual-arm data, more than 1,000 hours of instruction-labeled UMI data, and open datasets such as Bridge V2, RT-1, and DROID. Arm actions are expressed as relative end-effector poses in a common orientation frame; mobile bases and waists use their own velocity or relative-position channels, and masks prevent absent dimensions from contributing to the loss. The post-training sampling ratio is 0.5:0.5:0.5:8.5 across vision-language, open-source robot, instruction-labeled UMI, and in-house robot data.

### Data scale is the stronger axis in the reported sweep

The scaling experiment uses 12.5%, 25%, 50%, and 100% of a 20K-hour UMI subset so that the checkpoints can be compared at a manageable training budget. After post-training, out-of-box success across four tasks in unseen environments and object instances rises from 26% with no action pretraining to 53%, 56%, 69%, and 75% as the pretraining fraction grows. The full 100K-hour corpus is the data resource described by the paper; the 20K-hour subset is the controlled scaling study, and the two should not be conflated.

Model scaling is positive but flatter: the 2B, 5B, and 10B variants reach 61%, 75%, and 79% in the corresponding post-training evaluation. The gap between 5B and 10B is smaller than the gap created by adding UMI pretraining data, suggesting that representation coverage is the current bottleneck at these model sizes.

The downstream low-data test gives the scaling claim practical meaning. On four held-out tasks, fine-tuning with less than 10 hours per task yields 75% average success and 90% average progress for Xiaomi-Robotics-1, versus 40% and 66% for $\pi_{0.5}$. Each task is evaluated for 10 trials; the progress score matters because long-horizon failures can occur after several correct subtasks.

### Transfer is broad, but the interfaces still carry assumptions

Across benchmark families, Xiaomi-Robotics-1 reports 74.5% on RoboCasa, 57.4% on RoboCasa365 versus the previous 46.6%, and 20.07 on RoboDojo versus 13.07. These results combine large-scale pretraining, cross-embodiment alignment, and benchmark-specific evaluation, so they support a data-and-interface recipe rather than a claim that raw hours are interchangeable. UMI action normalization, state-transition caption quality, and the mask for missing embodiment channels are part of what makes the scaling curve work.

## High-Level Takeaways

- Xiaomi-Robotics-1 makes the useful scaling unit a labeled state transition: current observation, target change, and the action trajectory connecting them. This is richer than counting hours of unlabeled video, but it makes caption quality part of the data scale.
- The paper separates representation scale from embodiment alignment. Over 100K hours build the prior; about 10K cross-embodiment hours teach relative end-effector and platform-specific action conventions.
- Data scaling is stronger than model scaling in the controlled sweep: 26% without action pretraining rises to 75% at full 20K-subset pretraining, while 2B→10B moves 61%→79% after alignment.
- The low-data result (75% success, 90% progress with under 10 hours per task) is the most practical evidence that the broad prior transfers to new tasks, but it remains a four-task, 10-trial evaluation.
- The open question is whether the same gains survive a genuinely new embodiment, action convention, and instruction style when the prompt and masking scheme cannot be hand-validated.
