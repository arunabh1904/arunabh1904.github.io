---
title: 'OpenDriveVLA: Towards End-to-end Autonomous Driving with Large Vision Language Action Model'
date: '2025-03-30T00:00:00.000Z'
section: paper-shorts
postSlug: opendrivevla-towards-end-to-end-autonomous-driving-with-large-vision-language-action-model
legacyPath: /paper shorts/2025/03/30/opendrivevla-towards-end-to-end-autonomous-driving-with-large-vision-language-action-model.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – OpenDriveVLA: Towards End-to-end Autonomous Driving with Large Vision Language Action Model"
---
## 2025 – OpenDriveVLA

**arXiv:** [2503.23463](https://arxiv.org/abs/2503.23463)

**Project:** [DriveVLA](https://drivevla.github.io/)

**Code:** [DriveVLA/OpenDriveVLA](https://github.com/DriveVLA/OpenDriveVLA)

## Summary

> OpenDriveVLA keeps spatial structure visible to a language model instead of asking one visual token stream to carry every driving relation. It aligns 2D instance tokens, 3D instance tokens, map tokens, scene tokens, ego state, history, and command information through four staged training phases, then predicts discrete waypoints. On open-loop nuScenes, the 3B and 7B variants both reach 0.33 m average L2 under ST-P3 metrics; their average collision rates are 0.10% and 0.10%. The work reports no reactive closed-loop driving result, so its main claim is structured open-loop planning and instruction following.

## Core Insights

### Hierarchical tokens preserve geometry

OpenDriveVLA's basic design choice is to give the language model several spatially meaningful interfaces. Instance tokens describe detected agents in image space and 3D space; map tokens describe lane and drivable structure; scene tokens summarize global context. Ego state, temporal history, and a high-level command join those tokens before the autoregressive model predicts action. The model can therefore answer “what is around me?” and “what should I do?” using the same representation without discarding coordinates at the vision-language boundary.

![OpenDriveVLA's four-stage training pipeline](/assets/images/opendrivevla-towards-end-to-end-autonomous-driving-with-large-vision-language-action-model-paper-figure.png)
*Fig 1: The source pipeline moves from hierarchical feature alignment to driving instruction tuning, agent-environment-ego interaction modeling, and trajectory planning tuning. | source: [OpenDriveVLA, Figure 3](https://arxiv.org/abs/2503.23463)*

The figure should be read as a dependency chain. Stage 1 teaches the model what each structured token means. Stage 2 gives it language tasks over driving scenes. Stage 2.5 makes agent motion and its relation to the ego vehicle explicit. Stage 3 finally asks for the ego trajectory. This order reduces the burden on the planning loss: it does not have to discover perception, language grounding, and action geometry at the same time.

### Staged supervision turns one VLA into several specialists

The training data combines TOD3Cap, nuCaption, nuScenesQA, nuX, and GPT-Driver-style trajectory examples. Stage 1 aligns visual features while keeping the 2D encoder frozen. Stage 2 uses driving instruction and question-answer data. Stage 2.5 predicts agent motion conditioned on scene, map, and ego context, which gives the model a behavior-aware intermediate task. Stage 3 predicts a discrete waypoint sequence conditioned on the visual tokens, ego state, and driving command.

The command test makes the action interface concrete. In the source qualitative example, the same scene is evaluated with the original “keep forward” instruction and a modified “turn right” instruction. The planned path changes with the command, which is evidence for command conditioning. It is not proof that arbitrary natural-language commands are safe to execute or that every intermediate answer is grounded: the appendix reports a Figure 8 scene caption claiming that no pedestrians are visible in the right-front view even though one is present. The VLA can therefore follow a command while still hallucinating about a safety-relevant object.

![Instruction-following example with keep-forward and turn-right commands](/assets/images/opendrivevla-towards-end-to-end-autonomous-driving-with-large-vision-language-action-model-source-figure-3.webp)
*Fig 2: The qualitative source example compares the planned action under “keep forward” and “turn right” while showing the accompanying QA predictions. | source: [OpenDriveVLA, Figure 4](https://arxiv.org/abs/2503.23463)*

The important path through this figure is from changed instruction to changed trajectory. The middle answers are a trace of what the model says about the scene, but they are not independent perception evidence; the appendix counterexample shows why. A language-conditioned trajectory without reliable grounding would be a prompt sensitivity test, not a driving result.

### Open-loop gains plateau across model size

Under ST-P3 metrics, OpenDriveVLA-0.5B reports 0.35 m average L2 and 0.09% average collision rate, while the 3B and 7B variants both report 0.33 m and 0.10%. Under UniAD metrics, the corresponding average L2 values are 0.68, 0.67, and 0.66, with average collision rates 0.26%, 0.30%, and 0.25%. The larger models therefore do not deliver a monotonic safety gain, and the metric family changes the apparent ranking.

The staged ablation on the 0.5B model isolates where the structure helps. Adding instruction tuning to Stage 1 changes ST-P3 from 0.36 m/0.13% collision to 0.35/0.12%. Adding agent-environment-ego interaction changes it to 0.35/0.11%, and trajectory tuning reaches 0.35/0.09%. Under UniAD the sequence is 0.70/0.37%, 0.69/0.32%, 0.68/0.31%, and 0.68/0.26%. The small L2 changes alongside the larger collision changes suggest that structured intermediate supervision affects the dangerous tail more than average displacement.

![Structured instance, map, and scene tokens in a nuScenes example](/assets/images/opendrivevla-towards-end-to-end-autonomous-driving-with-large-vision-language-action-model-source-figure-5.webp)
*Fig 3: The source example labels instance, map, and scene tokens with their spatial context before the language model receives them. | source: [OpenDriveVLA, Figure 6](https://arxiv.org/abs/2503.23463)*

The token example explains why a language model can use these inputs without treating them as a flat caption. Object boxes, map boundaries, and scene context occupy different roles, and the tokenization keeps those roles inspectable. The remaining question is whether the same structure survives action feedback; this paper does not test that in a reactive simulator.

## High-Level Takeaways

- OpenDriveVLA acts through a discrete waypoint sequence conditioned on structured visual tokens, ego state, history, and a command.
- Staged alignment and agent-motion supervision improve collision metrics in the 0.5B ablation while average L2 barely moves.
- Scaling from 3B to 7B saturates on the reported open-loop averages, and ST-P3 and UniAD metrics can rank variants differently.
- The open-loop tables leave the main deployment question unanswered: whether spatial token grounding survives reactive feedback under the same controller and command coverage.
