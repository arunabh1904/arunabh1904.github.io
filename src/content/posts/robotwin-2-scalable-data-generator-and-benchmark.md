---
title: 'RoboTwin 2.0: A Scalable Data Generator and Benchmark for Bimanual Manipulation'
date: '2025-06-22T00:00:00.000Z'
section: paper-shorts
postSlug: robotwin-2-scalable-data-generator-and-benchmark
legacyPath: /paper shorts/2025/06/20/robotwin-2-scalable-data-generator-and-benchmark.html
tags:
  - Robotics
  - Evaluation
field: 'Robot Post-Training & Evaluation'
summary: "2025 – RoboTwin 2.0: A Scalable Data Generator and Benchmark for Bimanual Manipulation"
---
## 2025 – RoboTwin 2.0: A Scalable Data Generator and Benchmark

**arXiv:** [2506.18088](https://arxiv.org/abs/2506.18088)

**Project:** [RoboTwin](https://robotwin-platform.github.io/)

## Summary

> RoboTwin 2.0 combines an automated bimanual data factory with a 50-task benchmark. An MLLM writes programs against a skill API, executes them in simulation, and repairs failures with execution logs and optional visual feedback. The resulting trajectories vary objects, clutter, lighting, language, table height, and robot embodiment so that policy training and evaluation expose the same axes of variation.

## Core Insights

### Generate a program, then let the simulator argue with it

RoboTwin 2.0 starts with a natural-language task and an annotated object library, RoboTwin-OD, containing 731 objects across 147 categories. An MLLM retrieves relevant object and skill information, writes executable task code, and runs that code through the simulator. Execution logs report failures; a multimodal observer can inspect rendered observations and localize errors that logs alone do not describe. The loop terminates after a successful program or after repeated failed refinements, and only successful demonstrations enter the expert trajectory set.

![RoboTwin 2.0 pipeline from task language through MLLM code generation, simulation feedback, and randomized trajectories](/assets/images/robotwin-2-scalable-data-generator-and-benchmark-paper-figure.png)
*Fig 1: The data factory turns a task description into executable skill code, uses simulation and visual feedback to repair it, and randomizes successful rollouts for policy training. | source: [RoboTwin 2.0: A Scalable Data Generator and Benchmark for Bimanual Manipulation, Figure 2](https://arxiv.org/abs/2506.18088)*

Fig. 1 is the mechanism to keep in mind. Code generation is only the first guess. The simulator provides a cheap test of whether the two arms, object affordances, and API calls actually produce the requested outcome. The observer is optional, which makes its cost visible: symbolic execution feedback can be enough for some failures, while image-based diagnosis supplies information about contact and scene state that a log cannot encode.

### Diversity is structured around the failure modes of bimanual control

The pipeline randomizes five axes: scene clutter, lighting, background textures, table height, and language instructions. Table height varies by up to 3 cm in the reported domain-randomization setting. Embodiment-aware generation adds candidate grasps, operation directions, approach poses, and robot-specific reachability. Curobo provides GPU-accelerated motion planning across the supported configurations. The release contains more than 100,000 expert trajectories across 50 tasks and five dual-arm platforms: Aloha-AgileX, Piper, Franka, UR5, and ARX-X5.

![Figure 1 from RoboTwin 2.0 showing the object library, benchmark scale, and data-generation system](/assets/images/robotwin-2-scalable-data-generator-and-benchmark-source-figure-1.webp)
*Fig 2: RoboTwin 2.0 joins a 50-task benchmark, 731-object library, five embodiments, domain randomization, and an open data-generation pipeline. | source: [RoboTwin 2.0: A Scalable Data Generator and Benchmark for Bimanual Manipulation, Figure 1](https://arxiv.org/abs/2506.18088)*

The grasp augmentation has a clear embodiment boundary. Across 50 tasks, automated data-collection success rises from 52.2% in RoboTwin 1.0 to 60.5% in RoboTwin 2.0 on average. The largest gains are on lower-DoF platforms: Aloha-AgileX improves 65.1%→78.8%, Piper 2.4%→25.1%, and ARX-X5 68.6%→74.2%. Franka and UR5 change little (67.3%→67.2% and 57.6%→57.1%), which is consistent with flexible 7-DoF arms already having enough reachable grasp options.

![Figure 4 from RoboTwin 2.0 visualizing domain randomization and its texture library](/assets/images/robotwin-2-scalable-data-generator-and-benchmark-source-figure-4.webp)
*Fig 3: Domain randomization changes clutter, lighting, table height, backgrounds, and textures while retaining the task structure. | source: [RoboTwin 2.0: A Scalable Data Generator and Benchmark for Bimanual Manipulation, Figure 4](https://arxiv.org/abs/2506.18088)*

Fig. 3 explains why “more synthetic data” is too vague. A policy sees the same task under controlled changes, so a success can be attributed to a variation axis. The benchmark later reuses those axes at test time, making the data generator and the evaluation protocol deliberately coupled.

### Feedback improves the expert programs before policy learning

The code-generation experiment evaluates 10 tasks with multiple candidate programs. RoboTwin 1.0 one-shot generation reaches 47.4% average success; execution-log feedback raises it to 60.4%, and multimodal feedback reaches 63.9%. RoboTwin 2.0 starts at 62.1%, reaches 66.7% with logs, and 71.3% with multimodal feedback. Its top-five success is 78.6% with multimodal feedback. Refinement takes 1.76 iterations on average for RoboTwin 2.0 versus 2.42 for RoboTwin 1.0 under multimodal feedback, and one-shot code is 569.4 tokens versus 1,236.6. The feedback loop is therefore improving both candidate quality and program compactness, not only selecting a lucky rollout.

### Domain-randomized pretraining transfers unevenly

For the simulation robustness experiment, RDT and π0 are pretrained on 9,600 trajectories from 32 tasks, 300 per task, under either clean or randomized scenes. The downstream policy then receives 50 clean demonstrations on each of five unseen tasks and is evaluated under randomized conditions. RDT reaches 18.8% after clean pretraining and 24.8% after randomized pretraining; π0 reaches 22.5% and 29.1%. Clean data barely improves the robustness test, while randomized pretraining helps even though downstream fine-tuning remains clean.

The real experiment uses RDT on a COBOT-Magic dual-arm platform and four tasks: Stack Bowls, Handover Block, Pick Bottle, and Click Bell. It compares ten real demonstrations alone, those ten plus 1,000 randomized synthetic trajectories, and 1,000 synthetic trajectories without real demonstrations. The combined setting improves Stack Bowls from 22% to 64% on a seen uncluttered background and from 12% to 58% on a seen cluttered background. On the unseen cluttered Handover Block setting, synthetic-only reaches 20%, while the mixed set reaches 36%. Across the four test configurations, the paper reports a 24.4-point average improvement for the few-shot mixed setting; synthetic-only gains appear on unseen-background cases but are not uniformly better.

Finally, the benchmark tests ACT, Diffusion Policy, RDT, π0, and DP3 after 50 clean demonstrations per task on all 50 tasks, with 100 rollouts in clean Easy and randomized Hard conditions. Average Easy/Hard success is 34.5/13.7 for RDT, 46.4/16.3 for π0, 29.7/1.7 for ACT, 28.0/0.6 for Diffusion Policy, and 55.2/5.0 for DP3. RDT and π0 lose 20.8 and 30.1 points under the shift. DP3’s strong Easy score partly relies on perfect point clouds and clean segmentation in simulation, so the table is also a reminder that sensor assumptions can dominate benchmark rankings.

RoboTwin 2.0’s central risk is co-adaptation. The same simulator supplies the expert programs, the randomization axes, and the evaluation environments. The mixed real-world experiment is the most convincing counterweight because it changes backgrounds and clutter on physical hardware. The benchmark is most useful when its clean and Hard settings are read as controlled probes of transfer, not as a complete substitute for deployment diversity.

## High-Level Takeaways

- Simulation feedback turns MLLM code generation into an executable repair loop before trajectories are used for policy learning.
- Embodiment-aware grasps help constrained robots most, while flexible 7-DoF arms show little change in the data-collection ablation.
- Domain-randomized pretraining improves unseen-scene robustness and combines well with a small amount of real data, but synthetic-only transfer is uneven.
- The benchmark’s shared generator and test axes make it diagnostic and also create a co-adaptation risk that physical evaluation must expose.
