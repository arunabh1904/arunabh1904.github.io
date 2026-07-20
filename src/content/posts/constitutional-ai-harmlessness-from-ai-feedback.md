---
title: 'Constitutional AI: Harmlessness from AI Feedback'
date: '2022-12-15T09:00:00.000Z'
section: paper-shorts
postSlug: constitutional-ai-harmlessness-from-ai-feedback
legacyPath: /paper shorts/2022/12/15/constitutional-ai-harmlessness-from-ai-feedback.html
tags:
  - Alignment
  - AI Feedback
field: 'Alignment & Post-Training'
summary: "2022 – Constitutional AI: Harmlessness from AI Feedback"
---

## 2022 – Constitutional AI: Harmlessness from AI Feedback

**arXiv:** [2212.08073](https://arxiv.org/abs/2212.08073)

Constitutional AI replaces most harmfulness labels with an explicit list of principles and two model-mediated stages. The supervised stage produces critiques and revisions of harmful responses. The reinforcement stage asks a model to choose between responses under a sampled principle, trains a preference model, and optimizes the assistant against that model.

## Paper Insights

The constitution is both specification and data generator. It makes behavioral constraints inspectable, but every generated critique, revision, and preference still passes through a model whose interpretation can be incomplete. The supervised stage also gives the RL policy a safer starting distribution, reducing the exploration burden in the reinforcement phase.

The paper reports a less harmful, less evasive assistant with far fewer direct human harmfulness labels. For embodied systems, the transferable idea is not “let a VLM decide safety.” It is to encode constraints explicitly, generate adversarial and corrective supervision from those constraints, and preserve human evaluation as the external authority.

| Stage | Generated supervision | Function |
| --- | --- | --- |
| Constitutional SFT | Critique and revised response | Moves the policy into a better initial region. |
| RLAIF | AI preference under a principle | Scales comparisons without one human label per pair. |
| Human evaluation | Independent behavior judgment | Tests whether the constitution and judge produced the intended behavior. |

## Decision Lens

Constitutional AI informs whether scarce human attention should label every example or define rules and audit the supervision that models generate from them. The fundamental units are a critique–revision example and a preference pair. The policy and preference model are separate, and errors in the AI judge can be amplified by RL.

In robotics, a constitution could encode forbidden contacts, workspace boundaries, uncertainty-triggered stops, and recovery priorities. The missing experiment is causal: compare rule-generated feedback with hand-labeled physical violations under matched human time, then evaluate novel hazards. At ten times the task diversity, principle conflicts and unmodeled geometry will dominate. The approach fails if the critic can verbalize the right rule while rewarding trajectories that violate it physically.

**Context:** Constitutional AI is a blueprint for scalable supervision, not proof that automated oversight is self-validating.

**Limits:** The work studies dialogue harmlessness; physical safety constraints require grounded state and calibrated uncertainty.

**Takeaway:** A constitution lowers labeling cost only when independent evaluation can detect how the judge misread it.
