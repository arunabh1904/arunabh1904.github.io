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

> Open X-Embodiment asks whether robot learning can use one shared corpus despite incompatible robots, cameras, action spaces, and collection procedures. The paper standardizes more than one million trajectories from 22 embodiments and evaluates RT-1-X and RT-2-X as policies that can transfer experience across platforms.

## Core Insights

### Standardization must preserve meaning

The repository pools 60 existing datasets across 22 robot embodiments, 21 institutions, and 34 research labs. The paper reports 1M+ real robot trajectories, 527 skills, and 160,266 task instances. The point is not that all trajectories are interchangeable. It is to make the differences explicit enough that a policy can be trained on a shared container and still decode actions according to the robot that will execute them.

![RT-1-X and RT-2-X input and action interfaces across heterogeneous robots](/assets/images/open-x-embodiment-robotic-learning-datasets-and-rt-x-models-source-figure-1.webp)
*Fig 1: The common interface feeds an image and instruction into RT-1-X or RT-2-X, then maps their discretized end-effector outputs to robot-specific rates, grippers, and motion variables. | source: [Open X-Embodiment: Robotic Learning Datasets and RT-X Models, Figure 3](https://arxiv.org/abs/2310.08864)*

The consolidation uses RLDS serialized records, a canonical camera view per dataset, common image resizing, and a conversion of each original action into a seven-dimensional end-effector vector. Actions are normalized before discretization, then de-normalized according to the target embodiment. The output schema has seven end-effector dimensions plus a termination dimension, bucketed into 256 values for the policy experiments.

That alignment is deliberately coarse. The authors do not align coordinate frames, and a vector can represent an absolute position, a relative position, or a velocity depending on the original controller. A gripper command, camera viewpoint, timestamp, and success definition also retain embodiment-specific meaning. The visual figure makes this visible: RT-1-X and RT-2-X share a language-conditioned shape, but their output rates and controlled variables still differ.

The repository therefore solves a data plumbing problem while leaving a scientific question intact: does the shared model learn useful structure from many platforms, or does the largest and easiest dataset merely dominate the objective?

### RT-X turns the question into matched comparisons

The experiments train RT-1-X on the robotics mixture and RT-2-X by co-fine-tuning a PaLI-X vision-language model with the same robotics data. At the time of the experiments, the mixture covered nine manipulators, fewer than the full 22-embodiment repository. Models run at the rate required by the target robot, between 3 and 10 Hz; RT-2-X is served from a cloud system while RT-1-X runs locally.

The small-data evaluation compares five domains: Kitchen Manipulation, Cable Routing, NYU Door Opening, AUTOLab UR5, and Task-Agnostic Robot Play. The evaluation uses the original robot and task protocol for each domain, so success is not a single synthetic metric. RT-1-X outperforms the domain-specific original method on four of the five domains, and the paper reports a mean success rate about 50% higher than either the original methods or an RT-1 model trained in isolation.

![RT-1-X mean success across five small-data robot domains](/assets/images/open-x-embodiment-robotic-learning-datasets-and-rt-x-models-source-figure-2.webp)
*Fig 2: Across five small-data domains, the shared RT-1-X model's bars rise above the original methods and isolated RT-1 baselines, with the largest gains on cable routing and task-agnostic play. | source: [Open X-Embodiment: Robotic Learning Datasets and RT-X Models, Figure 4](https://arxiv.org/abs/2310.08864)*

The comparison is strongest when read as a data-transfer test. The RT-1 and RT-1-X networks have the same architecture in this evaluation; the major change is co-training on the multi-robot mixture. The model can use trajectories from other robots when the target domain has limited data. The result does not show that the common schema has learned a robot-independent dynamics model.

The large-data domains reveal the capacity boundary. On Bridge and the RT-1 paper's data, RT-1-X performs worse than the original domain-specific model, which the authors interpret as underfitting a 35M-parameter architecture on a larger and more heterogeneous objective. RT-2-X, with a 55B VLM backbone, performs strongly on those same domains. More data can therefore expose a model-capacity bottleneck rather than automatically improving a fixed policy.

### Data transfer is measurable on a held-out robot

The clearest experiment holds out skills from the Google Robot's own dataset. Bridge contains those skills for a different embodiment, WidowX. A 55B RT-2 trained on Google Robot data alone reaches 27.3% on the emergent-skill evaluation. RT-2-X trained with the robotics mixture reaches 75.8%; removing Bridge from that mixture reduces the score to 42.8%. The Bridge data is therefore connected to the additional skills, although the evaluation still reuses the Google Robot's action interface and visual setup.

The ablations separate several ingredients:

| Change | Emergent skills | RT-2-style generalization |
| --- | ---: | ---: |
| 55B RT-2, Google Robot data | 27.3% | 62% |
| 55B RT-2-X, full robotics mixture | 75.8% | 61% |
| 55B RT-2-X, mixture without Bridge | 42.8% | 54% |
| 5B RT-2-X, two-frame history | 44.4% | 52% |
| 5B RT-2-X, no history | 14.5% | 30% |
| 5B RT-2-X, two frames, from scratch | 0% | 1% |

The 55B versus 5B comparison points to capacity. The two-frame versus no-history comparison points to temporal context, which can matter when one image does not reveal the stage of a manipulation. The from-scratch row shows that web initialization is a major part of the VLA recipe. Because these rows change more than one feature at a time in places, they are design evidence rather than a complete causal decomposition.

RT-2-X is roughly on par with RT-2 on the paper's unseen-object, background, and environment evaluation. That is a useful negative result: cross-embodiment robot data adds physical skills, while the existing VLM backbone already supplies much of the semantic generalization.

### The repository's limits are part of its contribution

Open X-Embodiment's dataset diversity is uneven. Figure 2 shows that Franka contributes many visually distinct scenes, while xArm and Google Robot contribute large trajectory counts. Most language annotations and behaviors belong to the pick-and-place family, with a long tail of wiping, assembling, and other skills. Sampling and metadata determine which parts of this distribution the model actually sees.

The evaluation also does not test new robot embodiments, radically different sensors, navigation, or legged locomotion. A normalized action vector does not resolve force limits, coordinate frames, camera latency, or controller semantics. The repository is best understood as a common research substrate and a positive-transfer testbed, not evidence that one policy can immediately control any robot.

The decision lesson is concrete: when pooling robot data, keep embodiment-specific semantics in the record and evaluate transfer on a target robot with domain-specific baselines. A shared schema makes scaling possible; it does not make physical meaning disappear.

## High-Level Takeaways

- Open X-Embodiment makes cross-robot training reproducible by standardizing storage and a coarse action interface while retaining robot-specific decoding.
- RT-1-X transfers most reliably in small-data domains; larger heterogeneous mixtures can underfit a small architecture.
- The Google Robot holdout shows a large gain from skills present in Bridge data, with Bridge removal reducing that gain.
- Web initialization, model capacity, and short visual history each matter in the RT-2-X ablations.
- Dataset balance, action semantics, and evaluation protocols remain part of the transfer problem.
