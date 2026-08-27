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
## 2026 – UniDriveVLA

**arXiv:** [2604.02190](https://arxiv.org/abs/2604.02190)

**Project:** [UniDriveVLA](https://xiaomi-research.github.io/unidrivevla/)

**Code:** [xiaomi-research/UniDriveVLA](https://github.com/xiaomi-research/UniDriveVLA/)

### Method and reported result

UniDriveVLA argues that driving VLAs face an optimization conflict. Image-language models have strong semantic reasoning but weak spatial perception; 3D-enhanced systems improve geometry but can damage the VLM's reasoning behavior.

## Summary

> The paper's answer is expert decoupling. It uses specialized Transformer experts for understanding, perception, and action planning, then coordinates them with masked joint attention.

## Core Insights

UniDriveVLA uses a Mixture-of-Transformers design. A driving understanding expert handles semantic reasoning, a scene perception expert handles sparse spatial perception, and an action planning expert predicts driving actions. Masked joint attention lets experts exchange only the information they need. The paper also uses a three-stage progressive training recipe to stabilize the coupled system.

This is a sharper version of the VLA design problem: driving needs both language-level reasoning and precise spatial action grounding. UniDriveVLA tries to avoid forcing one Transformer stream to optimize all of those objectives at once. The abstract reports state-of-the-art results in open-loop nuScenes and closed-loop Bench2Drive among its comparisons.

![Figure 3 from UniDriveVLA showing the Mixture-of-Transformers architecture with understanding, perception, and action experts](/assets/images/unidrivevla-unifying-understanding-perception-and-action-planning-for-autonomous-driving-paper-figure.png)
*Fig 1: Shows UniDriveVLA's Mixture-of-Transformers architecture, where specialized experts are coordinated through masked joint attention. | source: [UniDriveVLA paper](https://arxiv.org/abs/2604.02190)*

![Figure 4 from UniDriveVLA: Unifying Understanding, Perception, and Action Planning for Autonomous Driving](/assets/images/unidrivevla-unifying-understanding-perception-and-action-planning-for-autonomous-driving-source-figure-4.webp)
*Fig 2: Masked joint attention lets prefix vision and text tokens, perception tokens, and suffix status or action tokens see only the dependencies required by their roles. | source: [UniDriveVLA: Unifying Understanding, Perception, and Action Planning for Autonomous Driving](https://arxiv.org/abs/2604.02190)*

![Figure 1 from UniDriveVLA: Unifying Understanding, Perception, and Action Planning for Autonomous Driving](/assets/images/unidrivevla-unifying-understanding-perception-and-action-planning-for-autonomous-driving-source-figure-1.webp)
*Fig 3: Comparison of VLA paradigms for autonomous driving. (a) Vanilla 2D VLA provides strong semantic reasoning but limited spatial perception. (b) 3D-enhanced VLA improves spatial perception but may degrade semantic reasoning. (c) UniDriveVLA decouples understanding, perception, and action with the Mixture-of-Transformers architecture, achieving both. | source: [UniDriveVLA: Unifying Understanding, Perception, and Action Planning for Autonomous Driving](https://arxiv.org/abs/2604.02190)*


**What to look at:**
- Understanding, perception, and action are separate experts.
- Masked joint attention controls cross-expert communication.
- Sparse perception keeps spatial grounding explicit without fully taking over the VLM.

### Reported evidence

| Design | Detail | Why it matters |
| ------ | ------ | -------------- |
| Experts | Understanding, perception, action planning | Decouples objectives that can fight each other. |
| Coordination | Masked joint attention | Shares information without collapsing every token into one stream. |
| Training | Three-stage progressive recipe | Stabilizes VLA optimization for driving. |
| Evaluation | nuScenes and Bench2Drive | Covers open-loop public data and closed-loop simulation. |

## High-Level Takeaways

- UniDriveVLA informs whether one fully shared Transformer should handle driving semantics, spatial perception, and action planning or whether those functions need specialized experts. Its atomic tokens enter expert-specific paths coordinated by masked joint attention, which controls information exchange without forcing every parameter to serve every objective.
- Specialization can reduce gradient conflict, but expert boundaries and joint-attention masks may encode the answer manually. The missing factorial ablation matches total parameters across fully shared, expert-only, and partially shared designs while measuring transfer and per-task gradients. At 10× modalities or tasks, routing imbalance and interface bandwidth dominate. The expert claim would fail if a dense shared model matched worst-task and closed-loop performance under equal training and inference FLOPs.
- UniDriveVLA makes expert decoupling a central design pattern for driving VLAs.
- The next VLA architecture decision is how to preserve semantic reasoning while adding spatial action competence, not how large to make the backbone.
