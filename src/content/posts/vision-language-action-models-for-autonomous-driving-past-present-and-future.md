---
title: 'Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future'
date: '2025-12-18T00:00:00.000Z'
section: paper-shorts
postSlug: vision-language-action-models-for-autonomous-driving-past-present-and-future
legacyPath: /paper shorts/2025/12/18/vision-language-action-models-for-autonomous-driving-past-present-and-future.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future"
---

**arXiv:** [2512.16760](https://arxiv.org/abs/2512.16760)

**Project:** [VLA4AD](https://worldbench.github.io/vla4ad)

**Awesome list:** [awesome-vla-for-ad](https://github.com/worldbench/awesome-vla-for-ad)

## Summary

> This survey organizes driving VLAs around two decisions: whether perception, reasoning, and planning live in one end-to-end model or in a slow VLM plus a fast driving system, and whether the action interface is textual or numerical. That split is more useful than the label “VLA” alone because it predicts where precision, latency, interpretability, and failure recovery are traded.

## Core Insights

### Every driving VLA exposes three interfaces

The paper starts from a compact formulation, (a_t = H(F(x_t;	heta))): a VLM backbone $F$ turns multimodal input into a representation, and an action head $H$ turns that representation into an executable output. The input $x_t$ is not just an image. The survey distinguishes camera and LiDAR observations, BEV or occupancy features, language instructions, and vehicle state such as speed, acceleration, steering, and yaw rate. A paper's real architecture is therefore determined by what enters $F$, what the backbone preserves, and what $H$ is allowed to emit.

![Representative VA and VLA models organized by output family and system boundary](/assets/images/vision-language-action-models-for-autonomous-driving-past-present-and-future-paper-figure.png)
*Fig 1: The survey's model map separates vision-action systems, end-to-end VLA textual and numerical generators, and dual-system VLA guidance or representation transfer. | source: [Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future, Figure 2](https://arxiv.org/abs/2512.16760)*

This framing prevents a common category error. A model that answers a scene question, a model that predicts a waypoint string, and a model that outputs a continuous trajectory may share a VLM backbone while imposing very different control contracts. The action head decides how much of the semantic representation must survive contact with the vehicle's timing and geometry.

### End-to-end systems choose between language-shaped and control-shaped actions

The survey divides end-to-end VLA into textual and numerical action generators. Textual generators produce meta-actions such as “slow down” or “change lane,” or serialize waypoints and reasoning in language. Their advantages are inspectability and a natural interface for instruction following. Their weakness is the gap between discrete words and continuous motion: a downstream controller still has to resolve timing, curvature, and interaction with other agents.

Numerical generators attach a regression, diffusion, flow-matching, or action-token head that produces trajectories or control values. LMDrive and similar systems predict control directly; ORION uses a generative trajectory head; AutoVLA and OpenDriveVLA discretize trajectories or structured actions into tokens. Numerical outputs are easier to execute and score, but the reasoning state becomes less visible and the head may need extensive trajectory supervision.

The action distinction is not cosmetic. A textual model can be semantically right yet numerically ambiguous; a numerical model can drive smoothly while failing to preserve the instruction that caused the maneuver. The survey's useful design question is which interface should be exposed to the safety-critical loop and which can remain a deliberative or explanatory layer.

### Dual-system VLAs relocate reasoning instead of eliminating it

Dual-system architectures separate a slow, deliberative VLM from a fast planner or controller. The survey further splits them into **explicit action guidance**, where the VLM emits a meta-action, waypoint, or other visible instruction, and **implicit representation transfer**, where VLM-generated explanations or features supervise the fast model during training. DriveVLM, Senna, and DiffVLA illustrate explicit guidance; VLP and VLM-AD illustrate distillation or feature transfer.

The distinction predicts different failure modes. Explicit guidance keeps the semantic decision inspectable, but a bad VLM command can enter the planner at runtime and add latency. Implicit transfer removes the VLM from deployment and makes the fast path cheaper, but it can compress away the very rationale that made the system easier to audit. A dual-system paper should therefore report both the quality of the slow signal and the cost of converting it into safe fast actions.

![Roadmap of the survey from preliminary interfaces through VA, VLA, datasets, and open challenges](/assets/images/vision-language-action-models-for-autonomous-driving-past-present-and-future-source-figure-1.webp)
*Fig 2: The roadmap connects input modalities, action generators, end-to-end versus dual-system architectures, open- and closed-loop benchmarks, and the unresolved challenges around generalization and trust. | source: [Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future, Figure 1](https://arxiv.org/abs/2512.16760)*

The survey's benchmark section makes the comparison boundary explicit. nuScenes and NAVSIM navtest emphasize open-loop trajectory prediction; Bench2Drive provides closed-loop routes and interaction. Its metric table distinguishes L2 and collision rate from ADE/FDE, miss rate, heading error, and control errors. Those are not interchangeable: open-loop agreement with an expert future does not test whether a policy remains stable after its own action changes the future.

## High-Level Takeaways

- Classify a paper by its system boundary, action representation, and runtime VLM dependency before comparing its model name or benchmark score.
- Textual actions buy inspectability; numerical actions buy control precision. The missing bridge is a verified mapping between semantic intent and continuous trajectory.
- Explicit guidance and implicit transfer make different promises about latency and auditability; report the VLM's role during deployment, not only during training.
- A useful VLA evaluation must pair open-loop accuracy with closed-loop route, infraction, latency, and instruction-fidelity measurements under matched sensors and action horizons.
