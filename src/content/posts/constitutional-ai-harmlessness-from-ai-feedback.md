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
*Fig 1: A schematic of CAI's two stages: self-critique and revision create supervised data, then constitutional comparisons train the preference model used for RLAIF. | source: [Constitutional AI, Figure 1](https://arxiv.org/abs/2212.08073)*

![Figure 6 from Constitutional AI: Harmlessness from AI Feedback](/assets/images/constitutional-ai-harmlessness-from-ai-feedback-source-figure-6.webp)
*Fig 2: Across one to sixteen constitutional principles, additional principles do not raise the harmlessness preference-model score; the paper argues that they can still diversify revisions for later RL exploration. | source: [Constitutional AI, Figure 6](https://arxiv.org/abs/2212.08073)*

![Figure 2 from Constitutional AI: Harmlessness from AI Feedback](/assets/images/constitutional-ai-harmlessness-from-ai-feedback-source-figure-2.webp)
*Fig 3: Crowdworker Elo scores place helpfulness on the horizontal axis and harmlessness on the vertical axis; the RL-CAI runs trace a better frontier than the human-feedback baselines in this 52B comparison. | source: [Constitutional AI, Figure 2](https://arxiv.org/abs/2212.08073)*


The constitution is both specification and data generator. It makes behavioral constraints inspectable, but every generated critique, revision, and preference still passes through a model whose interpretation can be incomplete. The supervised stage also gives the RL policy a safer starting distribution, reducing the exploration burden in the reinforcement phase.

The paper reports a less harmful, less evasive assistant with far fewer direct human harmfulness labels. For embodied systems, the transferable idea is not “let a VLM decide safety.” It is to encode constraints explicitly, generate adversarial and corrective supervision from those constraints, and preserve human evaluation as the external authority.

| Stage | Generated supervision | Function |
| --- | --- | --- |
| Constitutional SFT | Critique and revised response | Moves the policy into a better initial region. |
| RLAIF | AI preference under a principle | Scales comparisons without one human label per pair. |
| Human evaluation | Independent behavior judgment | Tests whether the constitution and judge produced the intended behavior. |

### Decision test and boundary

Constitutional AI is a blueprint for scalable supervision, not proof that automated oversight is self-validating. Its atomic units are a critique–revision example and a preference pair: the constitution defines the intended rule, a model generates corrective text and judgments, and RL optimizes against that learned signal. The paper studies dialogue harmlessness, where human evaluation remains the external check. For robotics, a constitution would need grounded constraints such as forbidden contacts, workspace boundaries, uncertainty-triggered stops, and recovery priorities. The decisive experiment compares rule-generated feedback with hand-labeled physical violations under matched human time and novel hazards. If the critic can verbalize the right principle while rewarding trajectories that violate it, the scaling advantage is illusory. A constitution lowers labeling cost only when independent evaluation can expose how the judge misread it.
