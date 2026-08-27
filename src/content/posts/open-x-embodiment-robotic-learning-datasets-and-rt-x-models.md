---
title: 'Open X-Embodiment: Robotic Learning Datasets and RT-X Models'
date: '2023-10-13T00:00:00.000Z'
section: paper-shorts
postSlug: open-x-embodiment-robotic-learning-datasets-and-rt-x-models
legacyPath: /paper shorts/2023/10/13/open-x-embodiment-robotic-learning-datasets-and-rt-x-models.html
tags:
  - Robotics
  - Data
field: 'Vision-Language-Action & Robotics'
summary: "2023 – Open X-Embodiment: Robotic Learning Datasets and RT-X Models"
---

## 2023 – Open X-Embodiment: Robotic Learning Datasets and RT-X Models

**arXiv:** [2310.08864](https://arxiv.org/abs/2310.08864)

**Project:** [robotics-transformer-x.github.io](https://robotics-transformer-x.github.io/)

## Summary

> Open X-Embodiment asks whether robotics can build a shared pretraining corpus despite incompatible robots, cameras, action spaces, and collection procedures. The collaboration standardizes data from 22 embodiments and 21 institutions, covering 527 skills and more than a million episodes in later releases.

## Core Insights

![Open X-Embodiment dataset composition across robot embodiments scenes trajectories skills and object categories](/assets/images/open-x-embodiment-robotic-learning-datasets-and-rt-x-models-paper-figure.png)
*Fig 1: Makes the mixture imbalance visible: embodiments contribute very different numbers of datasets, scenes, and trajectories even while the aggregate corpus covers a broad skill and object vocabulary. | source: [Open X-Embodiment](https://arxiv.org/abs/2310.08864)*

![Figure 1 from Open X-Embodiment: Robotic Learning Datasets and RT-X Models](/assets/images/open-x-embodiment-robotic-learning-datasets-and-rt-x-models-source-figure-1.webp)
*Fig 2: RT-1-X and RT-2-X both take images and a text instruction as input and output discretized end-effector actions. RT-1-X is an architecture designed for robotics, with a FiLM conditioned EfficientNet and a Transformer. | source: [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864)*

![Figure 2 from Open X-Embodiment: Robotic Learning Datasets and RT-X Models](/assets/images/open-x-embodiment-robotic-learning-datasets-and-rt-x-models-source-figure-2.webp)
*Fig 3: RT-1-X mean success rate is higher than that of either the Original Method or RT-1. RT-1 and RT-1-X have the same network architecture. | source: [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864)*


RT-X models trained on the mixture show positive transfer across several robots. The result makes dataset diversity a reusable asset, but standardization does not erase embodiment mismatch. A delta end-effector command, joint target, and mobile-base action can share a schema while retaining different physical meanings.

The paper's deeper contribution is infrastructural: data format, task language, mixture sampling, and evaluation must be designed together. Without per-dataset balancing, the largest source becomes the de facto objective. Without embodiment metadata, the policy cannot distinguish different control conventions.

| Heterogeneity | What must be normalized | What should remain explicit |
| --- | --- | --- |
| Sensors | Shapes, timestamps, calibration records | Viewpoint and missing modalities |
| Actions | Common trajectory container | Embodiment-specific semantics and scale |
| Tasks | Language or goal interface | Collection policy and success definition |

## High-Level Takeaways

- Open X-Embodiment informs whether to spend the next robot-data budget inside one platform or on a mixture that may transfer across platforms. Its atomic unit is a standardized trajectory, while the meaningful curriculum is the distribution over datasets, tasks, and embodiments.
- The RT-X results establish positive transfer for part of the mixture, not universal benefit from every dataset. The missing experiment is a full leave-one-embodiment-out matrix normalized by trajectory quality and sampling weight. At ten times the sources, hidden controller, timing, and annotation differences become dominant. The foundation-data claim fails if adding embodiments improves average benchmarks but increases adaptation data or harms safety on a target robot.
- Open X-Embodiment supplies the data substrate used by Octo, OpenVLA, and later generalist policies.
- Dataset scale counts episodes more easily than it measures coverage, quality, or recoveries.
- Cross-robot data becomes transferable only when the schema preserves the differences that matter for control.
