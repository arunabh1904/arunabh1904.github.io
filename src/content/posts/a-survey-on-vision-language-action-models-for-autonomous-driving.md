---
title: 'A Survey on Vision-Language-Action Models for Autonomous Driving'
date: '2025-06-30T04:00:00.000Z'
section: paper-shorts
postSlug: a-survey-on-vision-language-action-models-for-autonomous-driving
legacyPath: /paper shorts/2025/06/30/a-survey-on-vision-language-action-models-for-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: This survey organizes VLA-for-autonomous-driving work around architecture blocks, model evolution, datasets, benchmarks, and open deployment challenges.
---
## 2025 - A Survey on VLA Models for Autonomous Driving

**arXiv:** [2506.24044](https://arxiv.org/abs/2506.24044)

**Awesome list:** [Awesome-VLA4AD](https://github.com/JohnsonJiang1996/Awesome-VLA4AD)

**Summary:** This survey is a taxonomy and bibliography for VLA-for-autonomous-driving work. It treats VLA4AD as a distinct line: models that connect visual perception, language understanding, and driving actions rather than stopping at explanation or QA.

The survey is useful as a map. It shows how the field moved from end-to-end perception-to-control, to VLMs that explain scenes, to VLA systems that generate actions with language-aware reasoning.

## Paper Insights

The survey formalizes the building blocks of VLA4AD systems, compares more than 20 representative models, and reviews datasets and benchmarks. It separates architectures by how they connect perception, reasoning, and action, then closes with recurring deployment problems: robustness, real-time efficiency, safety, and formal verification.

This is not a method paper, so the value is coverage and vocabulary. It helps place papers like Talk2BEV, DriveLM, OpenDriveVLA, DiffVLA, UniDriveVLA, and DriveVLA-W0 on the same timeline without pretending they solve the same subproblem.

![Figure 2 from the VLA4AD survey showing the architecture blocks for vision-language-action autonomous driving systems](/assets/images/a-survey-on-vision-language-action-models-for-autonomous-driving-paper-figure.png)
_Figure 2 summarizes the VLA4AD architecture blocks, connecting visual inputs, language reasoning, and action generation. From the [survey paper](https://arxiv.org/abs/2506.24044), via arXiv HTML._

**What to look at:**
- The survey separates VLM-as-explainer, modular VLA, end-to-end VLA, and augmented VLA styles.
- Architecture blocks give a vocabulary for comparing otherwise incompatible systems.
- The bibliography is useful for tracking the fast-moving 2024-2026 driving VLA cluster.

**Taxonomy slice:**

| Category | Typical role | Examples to connect |
| -------- | ------------ | ------------------- |
| VLM as explainer | Describe or answer questions about a driving scene | Talk2BEV, DriveLM-style reasoning tasks |
| Modular VLA | Use language reasoning as an intermediate signal | Hybrid VLM plus planner systems |
| End-to-end VLA | Map scene inputs and instructions toward actions | OpenDriveVLA and related action models |
| Augmented VLA | Add tools, chains of thought, or world models | DiffVLA and DriveVLA-W0-style extensions |

## Decision Lens

This survey informs how to partition a driving-VLA research portfolio across perception-language alignment, world modeling, action generation, datasets, and closed-loop evaluation. Its comparison unit is not one token or trajectory but a system interface: visual representation, language/reasoning backbone, action head, and deployment loop.

The taxonomy is useful only if it predicts which interfaces transfer across papers. A controlled benchmark that fixes sensors, backbone, data, latency, and action space would test that causal value. As the field expands, inconsistent action definitions and mostly open-loop metrics will age the taxonomy faster than model names. The survey's organizing claim would fail if capability and safety differences were explained better by data quality or evaluation protocol than by the proposed architecture categories.

**Context:** The survey gives a shared vocabulary for a field where "VLA" can mean anything from QA to closed-loop trajectory generation.

**Takeaway:** Use this paper as the index card for the VLA-for-driving literature, then read the individual method papers for the actual design tradeoffs.
