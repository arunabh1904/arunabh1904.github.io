---
title: 'Colosseum V2: Benchmarking Generalization for Vision-Language-Action Models'
date: '2026-10-01T09:00:00.000Z'
section: paper-shorts
postSlug: colosseum-v2-benchmarking-generalization-for-vision-language-action-models
legacyPath: /paper shorts/2026/10/01/colosseum-v2-benchmarking-generalization-for-vision-language-action-models.html
tags:
- Robotics
- Benchmark
field: Robot Post-Training & Evaluation
summary: '2026 – Colosseum V2: Benchmarking Generalization for Vision-Language-Action Models'
---

## 2026 – Colosseum V2: Benchmarking Generalization for Vision-Language-Action Models

**Paper:** [arXiv:2605.27759](https://arxiv.org/abs/2605.27759) · [Full text, v4](https://arxiv.org/html/2605.27759v4) · [Project and supplementary results](https://colosseum-v2.github.io/)

## Summary

> Colosseum V2 separates task proficiency from robustness by training policies on clean demonstrations and testing controlled visual, language, and physical changes. Across 28 simulated tasks, ACT has higher unperturbed success than π0.5, while π0.5 is less sensitive to several visual shifts. The benchmark's most useful warning is that unchanged performance under corrupted instructions can indicate that a policy ignores language. Its hardware validation supports relative robustness trends within five tasks and one policy, not universal sim-to-real prediction.

## Core Insights

### Construct a clean task, then change a specified factor

Colosseum V2 uses GPU-parallel ManiSkill simulation with Franka Panda robots: a single arm with a parallel-jaw gripper, or two arms sharing a workspace. The 28 tasks cover 13 categories counted across those setups. Single-arm categories include picking, placing, reorientation, insertion, tools, articulated objects, and long-horizon sequences; the bimanual set adds handover and cooperative lifting while retaining coordination-heavy versions of other primitives.

The benchmark is built around **task–perturbation pairs**. Expert motion-planning solvers generate demonstrations using privileged simulator state. Each recorded timestep contains robot state, observation, and action. Training uses 100 unperturbed demonstrations per task. Evaluation then changes a specified factor and runs 200 episodes for each task-condition pair. Four separate multitask policies are trained: ACT and π0.5 for each morphology. This is not zero-shot transfer of one policy from a single arm to two arms.

The source overview connects the tasks to the perturbation dimensions. The changed factor, rather than a new task label, defines the main out-of-distribution test.

![Source Figure 2: Colosseum V2 tasks, robot morphologies, and perturbation factors](/assets/images/october-2605.27759-s1-f2.webp)
*Fig 1: The benchmark crosses manipulation tasks with controlled changes to observations, instructions, and physical configuration. Separate single-arm and bimanual suites expose different coordination demands. | source: [Colosseum V2, Figure 2](https://arxiv.org/html/2605.27759v4#S1.F2)*

[Open figure at full resolution](/assets/images/october-2605.27759-s1-f2.webp)

### The benchmark changes more than visual appearance

The paper organizes 19 conditions/factors around visual, language, and action robustness, including a combined “All” condition. The families need to be distinguished because a perturbation can change appearance, task information, or physical difficulty.

| Family | How the test is produced | What must be checked when interpreting it |
| --- | --- | --- |
| Visual | Object/table/background colors and textures, lighting, distractors, camera pose | A camera change also changes visible geometry; it is not equivalent to recoloring an object |
| Language | Equivalent paraphrase, another task's instruction, random words, or no instruction | Paraphrase preserves the task; the other conditions deliberately remove or corrupt the task information |
| Action/physical | Manipulated/receiving-object size and randomized initial poses | A larger object may make a task easier rather than harder |
| Combined | Several changes applied together in “All” | Combined failure does not identify which individual factor caused it |

The observation contract uses two external RGB cameras at 224 × 224 and a wrist camera at 128 × 128 on each arm: three views for single-arm tasks and four for bimanual tasks. Robot state contains joint positions and velocities. Single-arm demonstrations use absolute pose control; bimanual demonstrations use absolute joint control. Embodiment comparisons therefore also change action representation and coordination requirements.

### Baseline recipes are explicit but not compute matched

ACT uses the ManiSkill implementation modified to accept embeddings from a frozen CLIP text encoder. The π0.5 implementation comes from LeRobot. ACT's visual backbone is ResNet18; π0.5 uses the much larger pretrained SigLIP encoder. The benchmark is not proposing a new projector architecture, so these inherited model choices should not be mistaken for its contribution.

All four policies train on one H100. ACT uses batch 256 and learning rate $10^{-4}$, reaching convergence after 130k single-arm or 230k bimanual iterations. π0.5 uses batch 4 and learning rate $10^{-5}$ for 500k or 250k iterations. These are substantially different optimization budgets and prior pretraining histories. The comparison establishes performance under the reported recipes, not the isolated effect of adding a VLM backbone.

The nominal training set contains 2,800 task demonstrations across the benchmark, divided between morphology-specific policies. The principal distribution shift is clean training versus perturbed evaluation on the defined tasks; it should not be described as an unseen-task split. Simulator success conditions provide outcome labels, rather than an LLM judge grading a verbal answer.

### Absolute success and degradation answer different questions

Without perturbations, average single-arm/bimanual success is **29.2%/44.5% for ACT** and **21.2%/12.6% for π0.5**. A model can therefore show a smaller robustness drop while completing fewer tasks overall. Both the baseline success and its change are needed.

The perturbation figure reports changes in unnormalized success, with aggregation restricted to environments whose base success is at least 10%. Elsewhere, the paper also discusses normalized relative changes. These denominators and task filters are part of the metric, not cosmetic plotting choices.

![Source Figure 5: per-perturbation success changes for ACT and pi0.5](/assets/images/october-2605.27759-s3-f5.webp)
*Fig 2: Perturbation effects differ across model families and robot setups. This figure filters tasks by minimum baseline success, so its averages should be read alongside the all-task base scores. | source: [Colosseum V2, Figure 5](https://arxiv.org/html/2605.27759v4#S3.F5)*

[Open figure at full resolution](/assets/images/october-2605.27759-s3-f5.webp)

ACT changes little under all four language perturbations. The authors interpret this as evidence that it may ignore language, rather than evidence of perfect semantic generalization. π0.5 is more sensitive to language corruption, including paraphrases. This distinction matters: an observation can identify a task well enough for a policy to succeed even when the command is useless. A stronger language-grounding test would hold the scene fixed while requiring different valid actions for different instructions.

Initial-pose changes are particularly difficult in the bimanual suite. Object-size changes sometimes help by making a grasp or scoop easier. A robustness benchmark must therefore inspect how a perturbation changes task difficulty rather than assume every deviation is adversarial.

### Hardware validation supports a bounded claim

This note follows arXiv v4; the project landing page and its separate appendix still describe an earlier three-task hardware study. Those older counts and correlations should not be mixed with v4. The v4 hardware study reconstructs five single-arm tasks and trains one ACT policy from scratch using 60 demonstrations per task. Seven conditions are tested with 20 trials each, yielding 35 task-condition comparisons. The simulation/hardware success relationship has $R^2=0.49$. Normalizing each condition by its unperturbed score gives a mean absolute simulation-to-hardware difference of 12.3 percentage points.

That is useful evidence that relative perturbation effects can transfer, but it is a small validation slice: one policy, five tasks, and seven conditions. It does not validate bimanual π0.5 transfer or every benchmark perturbation. Rigid-body simulation also omits deformable manipulation and aspects of friction, compliance, force feedback, and contact behavior. My decision would be to use Colosseum V2 for broad failure diagnosis, then spend hardware trials on the specific shifts most consequential to the intended deployment.

## High-Level Takeaways

- The benchmark is constructed from clean planner-generated demonstrations and controlled task–perturbation evaluations, with 200 episodes per condition.
- Training morphology, camera count, and action representation differ across the two suites; they are not a pure embodiment-only control.
- Stronger visual robustness can coexist with lower base success, so report both absolute proficiency and degradation.
- Insensitivity to nonsense or missing language can expose an ignored input rather than a robust language-conditioned policy.
- Hardware results support relative robustness trends in a limited slice; contact-rich and broader cross-policy transfer remain unresolved.
