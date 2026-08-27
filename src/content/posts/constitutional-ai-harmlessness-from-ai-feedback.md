---
title: 'Constitutional AI: Harmlessness from AI Feedback'
date: '2022-12-15T00:00:00.000Z'
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

## Summary

> Constitutional AI replaces most harmfulness labels with an explicit list of principles and two model-mediated stages. The supervised stage produces critiques and revisions of harmful responses. The reinforcement stage asks a model to choose between responses under a sampled principle, trains a preference model, and optimizes the assistant against that model.

## Core Insights

![Constitutional AI pipeline showing supervised self-critique and revision followed by reinforcement learning from AI feedback](/assets/images/constitutional-ai-harmlessness-from-ai-feedback-paper-figure.png)
*Fig 1: Separates the two mechanisms: critique-and-revision creates supervised targets, then constitutional preference judgments train the reward model used for RLAIF. | source: [Constitutional AI](https://arxiv.org/abs/2212.08073)*

![Figure 6 from Constitutional AI: Harmlessness from AI Feedback](/assets/images/constitutional-ai-harmlessness-from-ai-feedback-source-figure-6.webp)
*Fig 2: We show harmlessness PM scores of revised responses for varying number of constitutional principles used. Increasing the number of principles does not improve these PM scores, but we have found that it improves the diversity of revised responses, which improves exploration during the RL phase of CAI training. | source: [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)*

![Figure 2 from Constitutional AI: Harmlessness from AI Feedback](/assets/images/constitutional-ai-harmlessness-from-ai-feedback-source-figure-2.webp)
*Fig 3: We show harmlessness versus helpfulness Elo scores (higher is better, only differences are meaningful) computed from crowdworkers’ model comparisons for all 52B RL runs. Points further to the right are later steps in RL training. | source: [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)*


The constitution is both specification and data generator. It makes behavioral constraints inspectable, but every generated critique, revision, and preference still passes through a model whose interpretation can be incomplete. The supervised stage also gives the RL policy a safer starting distribution, reducing the exploration burden in the reinforcement phase.

The paper reports a less harmful, less evasive assistant with far fewer direct human harmfulness labels. For embodied systems, the transferable idea is not “let a VLM decide safety.” It is to encode constraints explicitly, generate adversarial and corrective supervision from those constraints, and preserve human evaluation as the external authority.

| Stage | Generated supervision | Function |
| --- | --- | --- |
| Constitutional SFT | Critique and revised response | Moves the policy into a better initial region. |
| RLAIF | AI preference under a principle | Scales comparisons without one human label per pair. |
| Human evaluation | Independent behavior judgment | Tests whether the constitution and judge produced the intended behavior. |

## High-Level Takeaways

- Constitutional AI informs whether scarce human attention should label every example or define rules and audit the supervision that models generate from them. The fundamental units are a critique–revision example and a preference pair. The policy and preference model are separate, and errors in the AI judge can be amplified by RL.
- In robotics, a constitution could encode forbidden contacts, workspace boundaries, uncertainty-triggered stops, and recovery priorities. The missing experiment is causal: compare rule-generated feedback with hand-labeled physical violations under matched human time, then evaluate novel hazards. At ten times the task diversity, principle conflicts and unmodeled geometry will dominate. The approach fails if the critic can verbalize the right rule while rewarding trajectories that violate it physically.
- Constitutional AI is a blueprint for scalable supervision, not proof that automated oversight is self-validating.
- The work studies dialogue harmlessness; physical safety constraints require grounded state and calibrated uncertainty.
- A constitution lowers labeling cost only when independent evaluation can detect how the judge misread it.
