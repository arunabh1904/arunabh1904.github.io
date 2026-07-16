---
title: 'DriveLM: Driving with Graph Visual Question Answering'
date: '2023-12-21T05:00:00.000Z'
section: paper-shorts
postSlug: drivelm-driving-with-graph-visual-question-answering
legacyPath: /paper shorts/2023/12/21/drivelm-driving-with-graph-visual-question-answering.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: DriveLM formulates driving reasoning as graph visual question answering, linking perception, prediction, planning, behavior, and motion questions into a structured reasoning chain.
---
## 2023 - DriveLM

**arXiv:** [2312.14150](https://arxiv.org/abs/2312.14150)

**Project:** [OpenDriveLab DriveLM](https://opendrivelab.com/DriveLM/)

**Code:** [OpenDriveLab/DriveLM](https://github.com/OpenDriveLab/DriveLM)

**Summary:** DriveLM argues that driving VQA should be multi-step and graph-structured. Human drivers do not jump from pixels to steering in one question; they identify relevant objects, reason about interactions, decide behavior, and then produce motion.

DriveLM captures that process with Graph Visual Question Answering. Nodes represent reasoning stages such as perception, prediction, planning, behavior, and motion. Edges pass context between questions.

## Paper Insights

The paper contributes a task formulation, dataset, metrics, and a baseline agent. DriveLM-Data provides graph-structured QA annotations over driving scenes. DriveLM-Agent uses a vision-language model to answer stage-specific questions, propagate context through the graph, and convert behavior reasoning into a trajectory.

The paper's important move is making language reasoning structured rather than single-round. The risk is that graph QA can improve interpretability without solving all geometry and latency constraints. It is best understood as language-grounded scene reasoning that can complement conventional spatial modules.

![Figure 1 from DriveLM showing graph visual question answering, DriveLM-Data, DriveLM-Agent, metrics, and generalization](/assets/images/drivelm-driving-with-graph-visual-question-answering-paper-figure.png)
_Figure 1 shows DriveLM's task and artifacts: graph VQA, data construction, the DriveLM-Agent baseline, metrics, and generalization tests. From the [DriveLM paper](https://arxiv.org/abs/2312.14150), via the arXiv PDF._

**What to look at:**
- Questions are nodes in a logical graph, not isolated prompts.
- Earlier answers become context for later driving decisions.
- The task hierarchy separates perception, prediction, planning, behavior, and motion.

**Evals / Benchmarks / Artifacts:**

| Artifact | Detail | Why it matters |
| -------- | ------ | -------------- |
| Task | Graph Visual Question Answering | Makes driving reasoning multi-step and inspectable. |
| Dataset | DriveLM-Data | Provides graph-structured QA supervision. |
| Baseline | DriveLM-Agent | Shows how a VLM can use the graph to produce driving behavior. |
| Evaluation | DriveLM-Metrics and generalization settings | Tests semantic accuracy, trajectory quality, and unseen conditions. |

## Decision Lens

DriveLM informs whether driving reasoning should be supervised as isolated QA pairs or as a graph connecting perception, prediction, planning, behavior, and motion. The atomic item is a node question-answer pair with directed dependencies that encode which earlier facts support a later decision.

The graph makes reasoning supervision inspectable, but it can reward verbal consistency without improving control. The decisive study holds images and answer volume fixed while comparing graph-structured supervision, independent QA, and direct action labels on downstream closed-loop planning. At 10× graph depth, annotation cost and propagated label errors dominate. The formulation would fail if graph accuracy did not predict intervention rate or trajectory quality better than flat QA accuracy.

**Context:** DriveLM gave VLM-for-driving work a structured reasoning target instead of only asking open-ended scene questions.

**Takeaway:** Language helps driving most when it is grounded in a reasoning graph that can feed spatial planning.
