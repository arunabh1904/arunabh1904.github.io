---
title: 'VLA-REPLICA: A Low-Cost, Reproducible Benchmark for Real-World VLA Evaluation'
date: '2026-05-20T00:00:00.000Z'
section: paper-shorts
postSlug: vla-replica-low-cost-reproducible-real-world-evaluation
legacyPath: /paper shorts/2026/05/20/vla-replica-low-cost-reproducible-real-world-evaluation.html
tags:
  - Robotics
  - Evaluation
field: 'Robot Post-Training & Evaluation'
summary: "2026 – VLA-REPLICA: A Low-Cost, Reproducible Benchmark for Real-World VLA Evaluation"
---


**arXiv:** [2605.20774](https://arxiv.org/abs/2605.20774)

**Project:** [VLA-REPLICA](https://irvlutd.github.io/VLAReplica/)

## Summary

> VLA-REPLICA addresses a gap between scalable simulation and expensive centralized robot evaluation. It specifies a low-cost SO-101 arm, cameras, lighting enclosure, fixed workspace, ten manipulation tasks, adaptation demonstrations, and in-/out-of-distribution protocols that independent labs can assemble locally.

## Core Insights

![VLA-REPLICA procedure for aligning camera and robot poses reproducing object placement and running standardized policy evaluations](/assets/images/vla-replica-low-cost-reproducible-real-world-evaluation-paper-figure.png)
*Fig 1: Shows the reproducibility controls: AprilTag and video-overlay alignment fix viewpoint geometry, task reference images fix object placement, and the same suite is then run across policies. | source: [VLA-REPLICA](https://arxiv.org/abs/2605.20774)*

![Figure 1 from VLA-REPLICA: A Low-Cost, Reproducible Benchmark for Real-World VLA Evaluation](/assets/images/vla-replica-low-cost-reproducible-real-world-evaluation-source-figure-1.webp)
*Fig 2: Overview of the benchmark's hardware, workspace, and task protocol, including the low-cost arm, cameras, lighting enclosure, and standardized tabletop scene. | source: [VLA-REPLICA, Figure 1](https://arxiv.org/abs/2605.20774)*

### A benchmark is a physical interface contract

VLA-REPLICA specifies a $\sim\$1,050 setup built around a 6-DoF SO-101 arm, an RGB webcam, a RealSense D455, and a 32-inch light box at roughly 5,600 K. Calibration uses a matrix for action normalization, while reference images and video overlays constrain camera pose and object placement. The point is not that this tabletop is realistic in every way; it is that another lab can rebuild the same interface to the policy.

The suite contains ten tasks: four pick-and-place tasks, three object-interaction tasks, and three memory/counting tasks. The authors collect 500 demonstrations, 50 per task. Evaluation has 90 scenes: 50 in-distribution scenes, five per task across ten tasks, and 40 OOD scenes, five per task across eight selected tasks. OOD tests change color and shape; memory tasks train on counts 1 and 3 and test on 2, 4, and 5. Those details make “OOD” a specified intervention rather than a vague claim about generalization.

### The reproduced numbers expose what the setup does and does not prove

On the reproduced benchmark, the ID average success rates are ACT 18%, DiT-D 16%, DiT-F 12%, SmolVLA 26%, X-VLA 14%, $\pi_0$ 34%, and $\pi_{0.5}$ 54%. The OOD averages are 7.5%, 5%, 2.5%, 30%, 7.5%, 30%, and 35%, respectively. The sharp drop is itself useful evidence: a policy can rank well in a controlled scene and still fail when the object appearance or count changes. The memory/counting tasks are especially unforgiving because the policy must preserve information across time instead of reacting to one image.

The reproducibility experiment uses one independently assembled second setup at a different location. It selects five ID and three OOD tasks with the highest original success rather than replaying every task, and reports average success of 49% versus 48% on ID and 55% versus 50% on OOD for the original and replica setups. That is encouraging evidence that the calibration and workspace controls transfer, but it is not a multi-lab variance estimate and the task selection biases the comparison toward easier behaviors.

Low cost changes the cadence of evaluation. Instead of one lab reporting a small number of real trials, multiple groups can reproduce the setup and accumulate evidence about hardware, operator, and site variability.

| Design choice | Benefit | Tradeoff |
| --- | --- | --- |
| Off-the-shelf hardware | Broad reproducibility | Limited dexterity and task envelope |
| Controlled workspace | Comparable trials | Understates open-world variation |
| Local execution | Fast, transparent iteration | Requires calibration discipline across sites |

## High-Level Takeaways

- VLA-REPLICA informs whether to centralize evaluation on expensive hardware or distribute a standardized low-cost real setup. Its unit is a real closed-loop trial with a fixed protocol and explicit shift condition. Replication across independently assembled systems is the scaling variable that matters.
- The paper provides an initial second-setup check, not a population estimate of inter-lab variance. The next useful experiment is repeated assembly across sites and months with calibration drift, wear, and operator choice logged.
- The OOD protocol is concrete but narrow: color, shape, and count changes do not cover contact dynamics, lighting outside the enclosure, or household clutter.
- The reproduced ranking is most credible for the selected tasks; the authors' exclusion of already-low original tasks should remain attached to the claim.
- A smaller real benchmark can be more decision-useful than a broader one that nobody else can rebuild, provided its selection effects and physical envelope are reported.
