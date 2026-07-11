---
title: 'OpenDriveVLA: Towards End-to-end Autonomous Driving with Large Vision Language Action Model'
date: '2025-03-30T04:00:00.000Z'
section: paper-shorts
postSlug: opendrivevla-towards-end-to-end-autonomous-driving-with-large-vision-language-action-model
legacyPath: /paper shorts/2025/03/30/opendrivevla-towards-end-to-end-autonomous-driving-with-large-vision-language-action-model.html
tags:
  - Other
field: Autonomous Driving
summary: OpenDriveVLA adapts open VLMs for end-to-end driving by aligning 2D and 3D structured visual tokens with language and autoregressively modeling agent-environment-ego interactions.
---
## 2025 - OpenDriveVLA

**arXiv:** [2503.23463](https://arxiv.org/abs/2503.23463)

**Project:** [DriveVLA](https://drivevla.github.io/)

**Code:** [DriveVLA/OpenDriveVLA](https://github.com/DriveVLA/OpenDriveVLA)

**Summary:** OpenDriveVLA is a driving VLA built on open-source language models. It conditions action generation on camera-derived visual tokens, 3D structured perception, ego state, and language commands.

The interesting part is spatial grounding. The paper does not treat a VLM as a generic captioner bolted onto the stack. It aligns 2D and 3D driving tokens with the language model and then trains the system for driving instruction following, agent-environment-ego interaction, and trajectory planning.

## Paper Insights

OpenDriveVLA uses hierarchical vision-language alignment to map 2D and 3D structured visual tokens into a shared semantic space. The training recipe includes vision-centric pretraining, driving instruction tuning, interaction modeling, and trajectory planning tuning. The paper reports state-of-the-art results on nuScenes planning and driving QA among its compared methods.

This is part of the shift from "VLM as explainer" to "VLA as spatial action model." The caveat is that open-loop planning and QA are still indirect evidence; the real question is whether the aligned tokens remain reliable under closed-loop distribution shift.

![Figure 2 from OpenDriveVLA showing hierarchical feature alignment, driving instruction tuning, agent-env-ego interaction modeling, and trajectory planning tuning](/assets/images/opendrivevla-towards-end-to-end-autonomous-driving-with-large-vision-language-action-model-paper-figure.png)
_Figure 2 shows OpenDriveVLA's staged training pipeline, from hierarchical feature alignment to trajectory planning tuning. From the [OpenDriveVLA paper](https://arxiv.org/abs/2503.23463), via arXiv HTML._

**What to look at:**
- 2D and 3D structured visual tokens are both aligned to language.
- Agent-environment-ego interaction is an explicit training stage.
- Planning is produced by a VLA rather than a separate handoff from text reasoning.

**Evals / Benchmarks / Artifacts:**

| Component | Detail | Why it matters |
| --------- | ------ | -------------- |
| Spatial grounding | 2D and 3D structured tokens | Pushes beyond image-only VLM semantics. |
| Training | Hierarchical alignment plus instruction and planning tuning | Bridges perception, language, and action. |
| Interaction | Agent-environment-ego autoregressive modeling | Makes other actors and ego state part of the action context. |
| Evaluation | nuScenes planning and driving QA | Tests both action quality and language-grounded understanding. |

**Context:** OpenDriveVLA made the spatial-token design problem explicit for autonomous-driving VLA systems.

**Takeaway:** A driving VLA needs language reasoning, but it also needs structured 3D scene tokens that actions can depend on.
