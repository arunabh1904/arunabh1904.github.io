---
title: 'RT-1: Robotics Transformer for Real-World Control at Scale'
date: '2022-12-13T00:00:00.000Z'
section: paper-shorts
postSlug: rt-1-robotics-transformer-for-real-world-control-at-scale
legacyPath: /paper shorts/2022/12/13/rt-1-robotics-transformer-for-real-world-control-at-scale.html
tags:
  - Robotics
  - Generalist Policies
field: 'Vision-Language-Action & Robotics'
summary: "2022 – RT-1: Robotics Transformer for Real-World Control at Scale"
---

## 2022 – RT-1: Robotics Transformer for Real-World Control at Scale

**arXiv:** [2212.06817](https://arxiv.org/abs/2212.06817)

**Project:** [robotics-transformer1.github.io](https://robotics-transformer1.github.io/)

## Summary

> RT-1 asks whether one real-time policy can absorb a broad robot dataset without giving up execution speed. It combines language-conditioned EfficientNet features, TokenLearner compression, and a decoder-only transformer that emits discretized arm and base actions at 3 Hz.

## Core Insights

### The scaling target is a policy, not a bigger model

RT-1 uses Everyday Robots mobile manipulators with a seven-degree-of-freedom arm, two-finger gripper, and mobile base. Human operators collected about 130,000 demonstrations over 17 months with 13 robots. The dataset contains 744 language instructions, grouped around picking, moving objects, placing objects upright, knocking them over, opening and closing drawers, and moving objects into or out of receptacles.

The paper's question is practical: can one model share capacity across these tasks and still command a robot quickly enough to be useful? It runs closed loop at 3 Hz, predicts until it emits a terminate mode or reaches a time limit, and keeps network inference under a 100 ms budget. The data and latency constraints shape the architecture as much as the transformer does.

![RT-1 overview linking language instructions and image history to an action policy on the mobile manipulator](/assets/images/rt-1-robotics-transformer-for-real-world-control-at-scale-source-figure-1.webp)
*Fig 1: The paper's overview places a short visual history and task instruction before RT-1, then shows the policy selecting arm, base, gripper, or termination outputs for closed-loop execution. | source: [RT-1: Robotics Transformer for Real-World Control at Scale, Figure 1](https://arxiv.org/abs/2212.06817)*

The overview is easy to read as a generic vision-to-action pipeline, but the key design choice is the narrow interface in the middle: many images and language descriptions must become a small token sequence before the control decoder can run at the required rate.

### Language enters perception early

![RT-1 architecture with identity-initialized FiLM, TokenLearner, and transformer action decoder](/assets/images/rt-1-robotics-transformer-for-real-world-control-at-scale-source-figure-3.png)
*Fig 2: A six-image history is encoded by an instruction-conditioned EfficientNet, compressed from 81 spatial tokens to eight, and passed to an eight-layer decoder-only transformer that emits tokenized actions. | source: [RT-1: Robotics Transformer for Real-World Control at Scale, Figure 3](https://arxiv.org/abs/2212.06817)*

Each input frame is a $300\times300$ RGB image. ImageNet-pretrained EfficientNet-B3 maps it to a $9\times9\times512$ feature map, which RT-1 flattens into 81 visual tokens. The natural-language instruction is embedded with Universal Sentence Encoder and injected into the EfficientNet through FiLM layers. Those FiLM projections are identity-initialized: the dense layers producing the scale and shift begin at zero, so inserting language does not destroy the pretrained image features at the start of training.

TokenLearner then soft-selects eight informative tokens from the 81. Six frames therefore become 48 tokens for the decoder-only transformer, whose eight self-attention layers contain 19M parameters. This is a useful compression boundary. The model can let the instruction influence which visual features survive, while the transformer spends its compute on a short history rather than a full feature map.

The output is also deliberately discrete. Seven arm variables describe position, orientation, and gripper opening; three more describe base translation and yaw; a categorical mode chooses arm control, base control, or termination. Continuous dimensions are uniformly binned into 256 values and trained with categorical cross-entropy under a causal mask. The discretization gives the model a multimodal categorical distribution instead of forcing every diverse demonstration into one Gaussian mean.

### Real-time constraints become measured ablations

RT-1 uses two inference shortcuts. TokenLearner reduces the visual token workload and gives a reported 2.4× speedup. For overlapping six-frame windows, previously computed visual tokens are reused, giving another 1.7× speedup. The full model is about 35M parameters and has a reported 15 ms network inference time in the ablation table.

The speed experiment also clarifies what the model is not doing. Autoregressively conditioning action tokens increases inference to 36 ms and does not improve the main task or robustness scores, so the released design predicts the action dimensions without feeding each predicted action back into the next step. A continuous Gaussian action head drops seen-task success from 97% to 68% and unseen-task success from 76% to 43%; the paper attributes the gap to the inability of a single Gaussian mode to represent the varied action distributions in the demonstrations.

### What the data experiments isolate

On the main evaluation, RT-1 reaches 97% on seen instructions, 76% on unseen instructions, 83% under distractors, and 59% under new backgrounds. These tests still vary object placement, time of day, and robot position. A more realistic three-level kitchen evaluation reports 70% overall: 88% for the new layout and lighting condition, 75% after adding unseen distractors, and 50% after adding more radical task, object, or location changes.

The dataset ablation separates quantity from diversity. The full set scores 97/76/83/59 on seen tasks, all unseen tasks, distractors, and backgrounds. Capping examples per task to 51% of the data gives 71/50/52/39; reducing to 22% gives 59/29/14/31. Keeping 97% of the data while removing the least represented tasks leaves 75% of task diversity and gives 86/67/42/53. Removing 25% of the task types hurts generalization about as much as discarding roughly half the trajectories. This is evidence for diversity in this collection, not a law that every rare trajectory is worth more than a repeated successful one.

The model also absorbs controlled heterogeneity. Adding 518,000 successful simulated trajectories leaves real-object performance at 90% instead of 92%, raises performance on simulation-only objects from 23% to 87%, and raises performance on simulated objects paired with an unseen skill from 7% to 33%. Mixing 209,000 Kuka bin-picking episodes with Everyday Robots data changes classroom success from 92% to 90% but raises the Everyday Robots bin-picking evaluation from 22% to 39%. Kuka-only training transfers at 0% to that target robot. The shared action schema helps, but target-robot data is still needed to anchor morphology and control conventions.

### Long horizons expose the remaining boundary

Within SayCan, planning succeeds 87% in both Kitchen1 and Kitchen2 for RT-1. Execution succeeds 67% in each, including tasks with as many as 50 low-level stages. The stable score across kitchens is a strong test of short-horizon skill robustness, but it does not mean RT-1 learned long-horizon planning itself: SayCan proposes the sequence and RT-1 executes each instruction.

RT-1 remains imitation learning. New instructions are combinations of verbs and objects represented in the collection, and the policy does not learn a motion absent from its demonstrations. A 3 Hz controller is also a poor fit for contact-rich manipulation that needs faster feedback. The architecture's lesson is therefore narrower and useful: compress vision and language until a shared policy can run continuously, then spend data on task and object diversity that matches the intended generalization.

## High-Level Takeaways

- FiLM lets the instruction guide visual feature extraction before the transformer, while TokenLearner keeps the six-frame context within a real-time budget.
- Per-dimension discretization handles multimodal demonstrations better than the tested Gaussian head, at the cost of quantized actions.
- The strongest data result is about breadth: removing task types hurts generalization more than removing many examples from already represented tasks.
- Simulation and another robot can add useful behaviors when target-robot data anchors the shared policy; morphology is not removed by a common schema.
- RT-1's 3 Hz closed loop supports broad manipulation, while fast contact recovery and genuinely novel motions remain outside the evidence.
