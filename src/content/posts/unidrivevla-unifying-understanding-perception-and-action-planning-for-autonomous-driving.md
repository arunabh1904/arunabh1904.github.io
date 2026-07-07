---
title: 'UniDriveVLA: Unifying Understanding, Perception, and Action Planning for Autonomous Driving'
date: '2026-04-02T04:00:00.000Z'
section: paper-shorts
postSlug: unidrivevla-unifying-understanding-perception-and-action-planning-for-autonomous-driving
legacyPath: /paper shorts/2026/04/02/unidrivevla-unifying-understanding-perception-and-action-planning-for-autonomous-driving.html
tags:
  - Other
field: Autonomous Driving
summary: UniDriveVLA decouples semantic understanding, spatial perception, and action planning into specialized Transformer experts coordinated by masked joint attention.
---
## 2026 - UniDriveVLA

**arXiv:** [2604.02190](https://arxiv.org/abs/2604.02190)

**Project:** [UniDriveVLA](https://xiaomi-research.github.io/unidrivevla/)

**Code:** [xiaomi-research/UniDriveVLA](https://github.com/xiaomi-research/UniDriveVLA/)

**Plain-language summary:** UniDriveVLA argues that driving VLAs face an optimization conflict. Image-language models have strong semantic reasoning but weak spatial perception; 3D-enhanced systems improve geometry but can damage the VLM's reasoning behavior.

The paper's answer is expert decoupling. It uses specialized Transformer experts for understanding, perception, and action planning, then coordinates them with masked joint attention.

## Paper Insights

UniDriveVLA uses a Mixture-of-Transformers design. A driving understanding expert handles semantic reasoning, a scene perception expert handles sparse spatial perception, and an action planning expert predicts driving actions. Masked joint attention lets experts exchange only the information they need. The paper also uses a three-stage progressive training recipe to stabilize the coupled system.

This is a sharper version of the VLA design problem: driving needs both language-level reasoning and precise spatial action grounding. UniDriveVLA tries to avoid forcing one Transformer stream to optimize all of those objectives at once. The abstract reports state-of-the-art results in open-loop nuScenes and closed-loop Bench2Drive among its comparisons.

![Figure 3 from UniDriveVLA showing the Mixture-of-Transformers architecture with understanding, perception, and action experts](/assets/images/unidrivevla-unifying-understanding-perception-and-action-planning-for-autonomous-driving-paper-figure.png)
_Figure 3 shows UniDriveVLA's Mixture-of-Transformers architecture, where specialized experts are coordinated through masked joint attention. From the [UniDriveVLA paper](https://arxiv.org/abs/2604.02190), via arXiv HTML._

**What to look at:**
- Understanding, perception, and action are separate experts.
- Masked joint attention controls cross-expert communication.
- Sparse perception keeps spatial grounding explicit without fully taking over the VLM.

**Evals / Benchmarks / Artifacts:**

| Design | Detail | Why it matters |
| ------ | ------ | -------------- |
| Experts | Understanding, perception, action planning | Decouples objectives that can fight each other. |
| Coordination | Masked joint attention | Shares information without collapsing every token into one stream. |
| Training | Three-stage progressive recipe | Stabilizes VLA optimization for driving. |
| Evaluation | nuScenes and Bench2Drive | Covers open-loop public data and closed-loop simulation. |

**Why it mattered:** UniDriveVLA makes expert decoupling a central design pattern for driving VLAs.

**Take-home message:** The next VLA architecture fight is not just bigger backbones; it is how to preserve semantic reasoning while adding spatial action competence.
