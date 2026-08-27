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

### Method and reported result

SENNA uses a hybrid architecture: a vision-language model produces a high-level textual plan, while an end-to-end driving module converts that plan and the sensor input into a precise trajectory. The design avoids asking the LVLM to output exact steering-level control directly.

## Summary

> This makes the language layer inspectable. A planner can say what it intends to do before the control module turns that intent into geometry.

## Core Insights

SENNA separates driving into a VLM-based decision layer and an end-to-end trajectory layer. Senna-VLM produces structured scene understanding, decisions, and explanations; Senna-E2E turns those semantics into planning outputs. Training uses staged pretraining and driving-specific instruction data so the language model learns traffic context rather than generic image chat. The paper's value is making the semantic decision step explicit. The caveat is that language plans are not safety guarantees: downstream control still has to handle geometry, timing, and uncertainty.

![Figure 1: Previous methods plan trajectories without a decision-making step, making model learning difficult from SENNA: Bridging Large Vision-Language Models and End-to-End Autonomous Driving](/assets/images/senna-bridging-large-vision-language-models-and-end-to-end-autonomous-driving-paper-figure.png)
*Fig 1: Previous methods plan trajectories without a decision-making step, making model learning difficult. | source: [SENNA: Bridging Large Vision-Language Models and End-to-End Autonomous Driving paper](https://arxiv.org/abs/2410.22313)*

![Figure 5 from SENNA: Bridging Large Vision-Language Models and End-to-End Autonomous Driving](/assets/images/senna-bridging-large-vision-language-models-and-end-to-end-autonomous-driving-source-figure-5.webp)
*Fig 2: Qualitative results of Senna. The red boxes and text highlights key information that is relevant to driving decisions. | source: [SENNA: Bridging Large Vision-Language Models and End-to-End Autonomous Driving](https://arxiv.org/abs/2410.22313)*

![Figure 4 from SENNA: Bridging Large Vision-Language Models and End-to-End Autonomous Driving](/assets/images/senna-bridging-large-vision-language-models-and-end-to-end-autonomous-driving-source-figure-4.webp)
*Fig 3: Visualization of Meta-action data distribution in the DriveX dataset. | source: [SENNA: Bridging Large Vision-Language Models and End-to-End Autonomous Driving](https://arxiv.org/abs/2410.22313)*


**What to look at:**
- Senna-VLM produces high-level textual plans.
- Senna-E2E turns those plans into precise trajectories.
- The interface is inspectable and potentially editable.

### Reported evidence

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Decomposition | Language plan plus control module | Separates semantic intent from numeric control. |
| Training | Planning-oriented QA and curriculum | Tunes the VLM for traffic decisions. |
| Caveat | Language plan is not a guarantee | Control still needs safety validation. |

## High-Level Takeaways

- SENNA informs whether a driving stack should separate high-level semantic commands from low-level continuous trajectory control. The VLM predicts a compact driving intention from images and language context; a fast planner conditions on that intention to generate the trajectory.
- The hierarchy reduces language-generation latency in the control loop, but the command vocabulary can become an information bottleneck. The missing test varies command granularity and compares oracle, learned, and absent high-level guidance under matched planner capacity. At 10× scenario complexity, ambiguous commands and recovery from a wrong high-level decision dominate. The separation would fail if direct end-to-end planning matched safety and generalization without the semantic intermediate.
- SENNA captures a useful decomposition for safety-critical systems: use language for semantic planning, but keep numeric control in a component designed for precision.
- The most useful VLM in a driving stack may be the one that thinks out loud at the right abstraction level.
