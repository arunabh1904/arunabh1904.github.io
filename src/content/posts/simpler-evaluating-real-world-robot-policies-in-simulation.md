---
title: 'SIMPLER: Evaluating Real-World Robot Manipulation Policies in Simulation'
date: '2024-05-09T00:00:00.000Z'
section: paper-shorts
postSlug: simpler-evaluating-real-world-robot-policies-in-simulation
legacyPath: /paper shorts/2024/05/09/simpler-evaluating-real-world-robot-policies-in-simulation.html
tags:
  - Robotics
  - Evaluation
field: 'Robot Post-Training & Evaluation'
summary: "2024 – SIMPLER: Evaluating Real-World Robot Manipulation Policies in Simulation"
---

## 2024 – SIMPLER: Evaluating Real-World Robot Manipulation Policies in Simulation

**arXiv:** [2405.05941](https://arxiv.org/abs/2405.05941)

**Project:** [simpler-env.github.io](https://simpler-env.github.io/)

## Summary

> SIMPLER is a real-to-sim evaluation pipeline for policies trained on robot data. It does not ask a simulator to reproduce every physical detail; it asks whether simulated rollouts preserve the real-world ranking and failure sensitivities of the policies being compared. System identification closes the control gap, visual matching closes the observation gap, and paired experiments over roughly 1,500 episodes show strong agreement across Google Robot and WidowX setups.

## Core Insights

![SIMPLER comparison of expensive real-robot evaluation with reproducible simulated evaluation matched to the same task](/assets/images/simpler-evaluating-real-world-robot-policies-in-simulation-paper-figure.png)
*Fig 1: The benchmark proposition is ranking fidelity: a useful simulated evaluation should select the better real policy even when its images and physics are not a perfect digital twin. | source: [SIMPLER](https://arxiv.org/abs/2405.05941)*

### A useful simulator preserves decisions, not pixels

SIMPLER formalizes evaluation as a relative-performance problem. Pearson correlation measures a linear relationship between real and simulated success, but it can punish a simulator that gets the ranking right with a nonlinear scale. The paper adds Mean Maximum Rank Violation (MMRV): for each policy, find the largest real-world performance gap among pairs whose order the simulator reverses, then average those worst violations. Low MMRV means a wrong ordering occurs only between policies that are close in real performance; high Pearson $r$ is useful when the scale is also approximately linear.

That framing changes what to build. The target is not a photorealistic world in the abstract. It is a simulator whose control interface, camera geometry, object appearance, and task termination preserve the distinctions between the policies under test.

![Figure 4 from SIMPLER: system identification closing the real-to-sim control gap](/assets/images/simpler-evaluating-real-world-robot-policies-in-simulation-source-figure-2.webp)
*Fig 2: The source paper’s Figure 4 shows the same open-loop end-effector actions before and after system identification: the tuned simulator tracks the real arm closely enough to recover the coke-can grasp. | source: [SIMPLER, Figure 4](https://arxiv.org/abs/2405.05941)*

### Control calibration comes before visual polish

The Google Robot controller runs at 500 Hz in simulation while the policy outputs at 3 Hz. SIMPLER estimates stiffness and damping and other controller parameters from real open-loop trajectories with an offline simulated-annealing procedure. The intuition in Figure 2 is physical rather than cosmetic: a visually plausible arm that follows the commanded pose incorrectly can turn a successful real grasp into a simulated miss before the image encoder is even tested.

The visual pipeline then matches fixed-camera observations by green-screening real backgrounds, tuning foreground robot and object textures, aligning camera and robot poses, and optionally aggregating controlled variants. A key ablation supports the ordering. On Google Robot tasks, Visual Matching gives MMRV 0.056 and Pearson $r=0.924$ on average, while validation action MSE gives 0.375 and 0.308. Visual Matching beats Variant Aggregation as the default in this study: aggressive randomization can move some policies farther from the real observation distribution.

![Figure 2 from SIMPLER: the released suite of Google Robot and BridgeData V2 environments](/assets/images/simpler-evaluating-real-world-robot-policies-in-simulation-source-figure-0.webp)
*Fig 3: The source paper’s Figure 2 catalogs the open-source Google Robot and BridgeData V2 tasks and the standard Gym-style interface used for real-to-sim evaluation. | source: [SIMPLER, Figure 2](https://arxiv.org/abs/2405.05941)*

### The paired result is strong, but its scope is visible

On Google Robot tasks, the Visual Matching plots in the paper report MMRV/Pearson pairs of 0.031/0.976 for Pick Coke Can, 0.111/0.855 for Move Near, 0.055/0.915 for Open/Close Drawer, and 0.000/0.969 for Open Drawer and Place Apple. On the separate WidowX plus BridgeData V2 setup, the average is MMRV 0.014 and $r=0.890$. These are comparisons among six open-source policy checkpoints, including RT-1 checkpoints at different training stages, RT-1-X, RT-2-X, and Octo variants; Octo simulation scores are averaged over three random seeds and Google Robot scores over four arm/gripper color variants.

SIMPLER also reproduces within-policy sensitivity. Camera pose and table texture changes affect both real and simulated policies more than lighting or distractors in the reported shifts. A real arm-texture test confirms a trend first seen in simulation: Octo-Base is more sensitive to the arm appearance than RT-1-X. The result is a useful screening layer, not a proof that a policy is safe on hardware. The paper’s environments focus on rigid-object manipulation, green-screening assumes fixed cameras, and asset construction still includes manual curation.

## High-Level Takeaways

- SIMPLER’s unit of validity is a decision: does the simulated evaluation preserve which policy is better and which distribution shifts matter?
- The strongest pipeline combines system identification with visual matching. On Google Robot, it reduces average MMRV from 0.375 for validation MSE ranking to 0.056 and raises Pearson correlation from 0.308 to 0.924.
- The figures make the causal order intuitive: controller mismatch can cause a missed grasp before visual realism matters, while a matched camera and object texture preserve the policy’s observation distribution.
- Variant Aggregation is not automatically safer than matching. Randomized visuals can be farther from the real camera and invert a policy ranking even when they look diverse.
- The evidence covers fixed-camera rigid-object tasks and known policy families. Prospective validation is still needed when the action interface, embodiment, camera motion, or manipulated objects change.
