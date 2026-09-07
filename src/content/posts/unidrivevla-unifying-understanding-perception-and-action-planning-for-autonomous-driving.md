---
title: 'UniDriveVLA: Unifying Understanding, Perception, and Action Planning for Autonomous Driving'
date: '2026-04-02T00:00:00.000Z'
section: paper-shorts
postSlug: unidrivevla-unifying-understanding-perception-and-action-planning-for-autonomous-driving
legacyPath: /paper shorts/2026/04/02/unidrivevla-unifying-understanding-perception-and-action-planning-for-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2026 – UniDriveVLA: Unifying Understanding, Perception, and Action Planning for Autonomous Driving"
---

**arXiv:** [2604.02190](https://arxiv.org/abs/2604.02190)

**Project:** [UniDriveVLA](https://xiaomi-research.github.io/unidrivevla/)

**Code:** [xiaomi-research/UniDriveVLA](https://github.com/xiaomi-research/unidrivevla/)

## Summary

> UniDriveVLA treats the spatial-perception versus language-reasoning conflict as an optimization problem. Its Mixture-of-Transformers assigns understanding, perception, and action tokens to separate experts, then reconnects them with masked joint attention. Sparse detection, mapping, occupancy, ego, and motion queries supply spatial priors without replacing the language model's semantic path. The model reaches 78.37 driving score on Bench2Drive and reduces the shared-decoder planning error from 0.641 to 0.533 m in its ablation, but the evidence still mixes open-loop, closed-loop, and multi-task protocols.

## Core Insights

### Expert separation changes who is allowed to interfere with whom

A 2D VLA preserves the semantic behavior of its pretrained VLM but has weak spatial grounding. Adding 3D or occupancy tokens can improve geometry while forcing semantic and spatial objectives through the same parameters. UniDriveVLA addresses that tension with three token groups: understanding tokens from the vision-language backbone, sparse perception tokens for detection, mapping, occupancy, ego status, and motion, and action tokens for flow-matching trajectory generation.

![UniDriveVLA architecture with understanding, perception, and action experts](/assets/images/unidrivevla-unifying-understanding-perception-and-action-planning-for-autonomous-driving-paper-figure.png)
*Fig 1: The model separates understanding, perception, and action into expert-specific paths while retaining a masked joint-attention interface for coordinated driving. | source: [UniDriveVLA, Figure 3](https://arxiv.org/abs/2604.02190)*

Masked joint attention is the guardrail. Understanding tokens remain causal and cannot read later perception or action tokens. Perception tokens can read preceding semantic context, while action tokens can aggregate both semantic and spatial information. The model therefore keeps a shared decision process without asking the semantic expert to solve every spatial objective. The figure is best read as a visibility map: specialization is only useful if the attention mask preserves a direction of information flow.

![Masked joint attention across prefix, perception, and suffix token groups](/assets/images/unidrivevla-unifying-understanding-perception-and-action-planning-for-autonomous-driving-source-figure-4.webp)
*Fig 2: The mask lets perception tokens use semantic context and action tokens use both semantic and spatial context, while preserving the causal behavior of the language prefix. | source: [UniDriveVLA, Figure 4](https://arxiv.org/abs/2604.02190)*

### Sparse perception provides spatial evidence without a dense takeover

The perception branch is not a single detector bolted onto the VLM. It uses task-specific sparse queries initialized from dataset-level K-means instance banks, then updates them through temporal interaction, intra-task reasoning, cross-task communication, deformable feature aggregation, and task-wise refinement. Detection, online mapping, ego status, motion, and occupancy are trained as mutually supporting outputs. Their first-pass results are projected back into the perception expert's hidden space, where they can interact with understanding and action before a refined sparse decoder produces the final outputs.

Training is staged. The first stage anchors semantic reasoning with a mixture of driving VQA and general multimodal data. The second jointly adds language modeling, spatial tasks, and flow-matching planning with LoRA and half the base VLM learning rate. The third freezes the VLM and specializes the perception and action experts, adding a motion objective to give the action expert dynamic priors. This recipe is as important as the architecture: it limits catastrophic forgetting while gradually exposing the model to spatial and control losses.

The paper's representation analysis supports the motivation rather than proving it universally. In the shared-weight decoder, semantic and perception token cosine similarity rises toward one across layers; the MoT keeps it lower. That is consistent with feature collapse under shared optimization, but it does not establish that every dense shared decoder will fail or that expert decoupling is the only remedy.

### The controlled ablation is stronger than the leaderboard claim

On Bench2Drive, UniDriveVLA reports 78.37 driving score, 51.82% success, 198.86 efficiency, and 11.78 comfortness. Its fine-grained scores are 38.75% for merging, 80.00% for overtaking, 50.00% for emergency braking, 30.00% for giving way, and 58.95% for traffic signs, for a 51.53% mean. These are closed-loop CARLA results with six-view 900×1600 inputs and Think2Drive-generated demonstrations.

The shared-decoder ablation is more diagnostic. Replacing the shared-weight decoder with MoT changes general VQA from 31.1 to 45.5, DriveBench from 50.8 to 54.9, planning L2 from 0.641 to 0.533 m, and collision rate from 0.175 to 0.140. The action and perception additions also matter in the nuScenes ablation: adding ego state lowers L2 from 0.75 to 0.61, detection lowers collision rate to 0.10, and occupancy reaches the best L2 of 0.53 in that controlled sequence. The table does not show a matched compute sweep, and the model's motion prediction remains behind specialized baselines. The central claim is therefore “decoupling helps this unified recipe,” not “three experts always beat a shared model.”

![Vanilla 2D VLA, 3D-enhanced VLA, and UniDriveVLA](/assets/images/unidrivevla-unifying-understanding-perception-and-action-planning-for-autonomous-driving-source-figure-1.webp)
*Fig 3: The paper's motivating comparison frames UniDriveVLA as a compromise between semantic preservation and spatial perception, with continuous action prediction added as a third requirement. | source: [UniDriveVLA, Figure 1](https://arxiv.org/abs/2604.02190)*

## High-Level Takeaways

- Use expert decoupling when semantic, spatial, and action losses compete; use the mask to state exactly which information each branch may consume.
- Sparse queries are a bandwidth choice: they preserve task-specific geometry while avoiding a dense spatial representation taking over the VLM.
- The MoT versus shared-decoder ablation supports the mechanism more directly than cross-paper state-of-the-art comparisons.
- A decisive follow-up would match parameter count, active FLOPs, training data, and inference latency across shared, partially shared, and MoT models while measuring semantic retention and closed-loop safety separately.
