---
title: 'Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)'
date: '2023-04-23T00:00:00.000Z'
section: paper-shorts
postSlug: action-chunking-with-transformers-act
legacyPath: /paper shorts/2023/04/23/action-chunking-with-transformers-act.html
tags:
  - Robotics
  - Imitation Learning
field: 'Vision-Language-Action & Robotics'
summary: "2023 – Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)"
---

## 2023 – Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware

**arXiv:** [2304.13705](https://arxiv.org/abs/2304.13705)

**Project:** [ALOHA](https://tonyzhaozh.github.io/aloha/)

## Summary

> ACT pairs a low-cost bimanual teleoperation platform with a conditional variational autoencoder that predicts action sequences. The policy does not choose one motor command in isolation; it proposes a chunk, while temporal ensembling blends overlapping predictions during execution.

## Core Insights

![ACT conditional variational autoencoder with a transformer action decoder plus action chunking and temporal ensembling](/assets/images/action-chunking-with-transformers-act-paper-figure.png)
*Figure 4 shows the learned policy and deployment rule together: the CVAE predicts an action chunk from multi-view images and joints, while temporal ensembling reconciles overlapping chunk predictions during execution. source: [ACT](https://arxiv.org/abs/2304.13705)*

![Figure 1 from Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)](/assets/images/action-chunking-with-transformers-act-source-figure-1.webp)
*Figure 1 Fig. 1 : Left: Camera viewpoints of the front, top, and two wrist cameras, together with an illustration of the bimanual workspace of ALOHA . Middle: Detailed view of the “handle and scissor” mechanism and custom grippers. Right: Technical spec of the ViperX 6dof robot [ 1 ]. source: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)](https://arxiv.org/abs/2304.13705)*


Action chunking reduces the effective decision horizon and captures coordinated motion that per-step regression tends to fragment. A latent variable models variation across human demonstrations, while a transformer consumes multi-view images and joint state to generate the chunk. Temporal ensembling smooths successive predictions without committing to an entire open-loop plan.

The paper reports 80–90% success on six precise real-world tasks using roughly ten minutes of demonstrations per task. That result ties model design to a data-collection system: cheap, expressive teleoperation produces the demonstrations that make the sequence model useful.

| Lever | Benefit | Tradeoff |
| --- | --- | --- |
| Longer chunks | Shorter effective horizon and smoother behavior | Slower response to unexpected state changes |
| Temporal ensembling | Blends overlapping action predictions | Adds a weighting and latency choice |
| Latent action style | Represents multimodal demonstrations | Can absorb inconsistency rather than task structure |

## High-Level Takeaways

- ACT informs the chunk-length and execution-interface decision for imitation-trained robot policies. Its training unit is an observation paired with a future action trajectory. Visual and proprioceptive inputs share a transformer representation, while the CVAE compresses demonstration variation into a latent style variable.
- The results establish that chunks are effective for the studied precise bimanual tasks, not that longer is always better. A missing ablation would sweep chunk length under controlled perturbation frequency and inference latency. At ten times the task duration, open-loop commitment and latent-style ambiguity can compound. The method's core advantage disappears if a matched recurrent single-step policy recovers faster from disturbances while retaining the same smoothness and success.
- ACT made action chunks a default design axis for modern VLA adaptation recipes.
- The policy still learns from demonstrations and inherits their state coverage.
- Chunking buys temporal coherence by spending responsiveness; choose the horizon from the disturbance timescale, not convention.
