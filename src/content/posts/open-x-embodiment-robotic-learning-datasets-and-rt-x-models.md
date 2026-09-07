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

![RT-1-X and RT-2-X input and action interfaces across heterogeneous robots](/assets/images/open-x-embodiment-source-figure-3-white.png)
*Fig 1: The common interface feeds an image and instruction into RT-1-X or RT-2-X, then maps their discretized end-effector outputs to robot-specific rates, grippers, and motion variables. | source: [Open X-Embodiment: Robotic Learning Datasets and RT-X Models, Figure 3](https://arxiv.org/abs/2310.08864)*

The repository uses RLDS serialized records while retaining diverse sensor and action spaces. The RT-X policy experiments make a narrower choice: they select nine manipulators, one canonical camera view per dataset, common image resizing, and seven end-effector action dimensions plus termination. These experimental actions are normalized and bucketed into 256 values, then de-normalized for the target robot. The entire 60-dataset repository is not converted into one identical physical action space.

That alignment is deliberately coarse. The authors do not align coordinate frames, and a vector can represent an absolute position, a relative position, or a velocity depending on the original controller. A gripper command, camera viewpoint, timestamp, and success definition also retain embodiment-specific meaning. The visual figure makes this visible: RT-1-X and RT-2-X share a language-conditioned shape, but their output rates and controlled variables still differ.

The repository therefore solves a data plumbing problem while leaving a scientific question intact: does the shared model learn useful structure from many platforms, or does the largest and easiest dataset merely dominate the objective?

### RT-X turns the question into matched comparisons

The experiments train RT-1-X on the robotics mixture and RT-2-X by co-fine-tuning a PaLI-X vision-language model with the same robotics data. At the time of the experiments, the mixture covered nine manipulators, fewer than the full 22-embodiment repository. The study reports more than 3,600 evaluation trials across six robots. Models run at the rate required by the target robot, between 3 and 10 Hz; RT-2-X is served from a cloud system while RT-1-X runs locally.

The small-data evaluation compares five domains: Kitchen Manipulation, Cable Routing, NYU Door Opening, AUTOLab UR5, and Task-Agnostic Robot Play. The evaluation uses the original robot and task protocol for each domain, so success is not a single synthetic metric. RT-1-X outperforms the domain-specific original method on four of the five domains, and the paper reports a mean success rate about 50% higher than either the original methods or an RT-1 model trained in isolation.

![RT-1-X mean success across five small-data robot domains](/assets/images/open-x-embodiment-robotic-learning-datasets-and-rt-x-models-source-figure-2.webp)
*Fig 2: The shared RT-1-X model improves over the original method in four of five small-data domains. Compare each group separately: a higher average does not mean every target robot benefits. | source: [Open X-Embodiment: Robotic Learning Datasets and RT-X Models, Figure 4](https://arxiv.org/abs/2310.08864)*

The comparison is strongest when read as a data-transfer test. The RT-1 and RT-1-X networks have the same architecture in this evaluation; the major change is co-training on the multi-robot mixture. The model can use trajectories from other robots when the target domain has limited data. The result does not show that the common schema has learned a robot-independent dynamics model.

The large-data domains reveal a more qualified capacity boundary. On the two Bridge evaluations, isolated RT-1 scores 40/30 and RT-1-X scores 27/27, although RT-1-X still beats the original LCBC baseline at 13/13. On Google Robot, RT-1 falls from 92 to 73 after cross-robot training. The authors interpret these drops against isolated RT-1 as underfitting by its 35M-parameter architecture. The 55B RT-2-X scores 50/30 on Bridge and 91 on Google Robot: it improves one Bridge setting, ties the other, and nearly matches the Google-only baseline. More data can expose a capacity bottleneck, but the baseline and domain determine whether there is a gain.

### Skills can transfer to a robot that never demonstrated them

The clearest experiment tests skills absent from the Google Robot's own dataset, while the robot itself remains represented in training. Bridge contains those skills for a different embodiment, WidowX. A 55B RT-2 trained on Google Robot data alone reaches 27.3% on the emergent-skill evaluation. RT-2-X trained with the robotics mixture reaches 75.8%; removing Bridge from that mixture reduces the score to 42.8%. The Bridge data is therefore connected to the additional skills, although the evaluation still reuses the Google Robot's action interface and visual setup.

The ablations separate several ingredients:

| Change | Emergent skills | RT-2-style generalization |
| --- | ---: | ---: |
| 55B RT-2, Google Robot data, no history | 27.3% | 62% |
| 55B RT-2-X, full robotics mixture, no history | 75.8% | 61% |
| 55B RT-2-X, mixture without Bridge, no history | 42.8% | 54% |
| 5B RT-2-X, two-frame history | 44.4% | 52% |
| 5B RT-2-X, no history | 14.5% | 30% |
| 5B RT-2-X, two frames, web initialization, no web co-fine-tuning | 48.7% | 47% |
| 5B RT-2-X, two frames, from scratch, no web co-fine-tuning | 0% | 1% |

All rows except the last two use web co-fine-tuning. At 5B, adding two-frame history raises both scores, consistent with one image failing to reveal the stage of a manipulation. The last two rows hold the absence of web co-fine-tuning fixed: web initialization raises emergent skills from 0 to 48.7 and generalization from 1 to 47. Retaining web tasks during fine-tuning has mixed effects, however: 44.4/52 with co-fine-tuning versus 48.7/47 without it. The authors find these recipes comparable with the more diverse robot mixture. The 55B and two-frame 5B rows also differ in visual history, so their gap is not a pure capacity comparison.

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
