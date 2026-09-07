---
title: 'LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning'
date: '2023-06-05T00:00:00.000Z'
section: paper-shorts
postSlug: libero-benchmarking-knowledge-transfer-for-lifelong-robot-learning
legacyPath: /paper shorts/2023/06/05/libero-benchmarking-knowledge-transfer-for-lifelong-robot-learning.html
tags:
  - Robotics
  - Evaluation
field: 'Robot Post-Training & Evaluation'
summary: "2023 – LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning"
---

## 2023 – LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning

**arXiv:** [2306.03310](https://arxiv.org/abs/2306.03310)

**Project:** [libero-project.github.io](https://libero-project.github.io/)

## Summary

> LIBERO turns lifelong robot learning into a controlled transfer problem. Its procedural generator varies objects, spatial relations, and goals separately, while its metrics distinguish learning the next task from retaining the tasks already learned.

## Core Insights

### A benchmark built around what changes

![LIBERO's four task suites separate spatial, object, goal, and entangled knowledge transfer](/assets/images/libero-benchmarking-knowledge-transfer-for-lifelong-robot-learning-paper-figure.png)
*Fig 1: The benchmark contains three ten-task suites that isolate spatial, object, and goal changes, plus LIBERO-100 with entangled transfer; the lower panel maps these suites to five lifelong-learning questions. | source: [LIBERO, Figure 1](https://arxiv.org/abs/2306.03310)*

LIBERO-SPATIAL, LIBERO-OBJECT, and LIBERO-GOAL each contain ten tasks. Spatial tasks reuse the objects and goal while changing relationships; Object tasks request a new object; Goal tasks keep objects and layout fixed while changing the desired behavior. LIBERO-100 contains 100 mixed tasks, split in this paper into 90 short-horizon LIBERO-90 tasks for pretraining and 10 long-horizon LIBERO-LONG tasks for downstream lifelong learning. Together the four suites contain 130 standardized, language-conditioned tasks with 50 teleoperated trajectories per task.

This decomposition gives a useful diagnostic. A policy can fail because it cannot remember a new object, because it cannot preserve a previously learned manipulation, or because its action repertoire does not transfer to a new goal. A single average success rate hides those causes.

### From language template to executable goal

![LIBERO's procedural pipeline maps activity language to a scene, initial state, and PDDL goal](/assets/images/libero-benchmarking-knowledge-transfer-for-lifelong-robot-learning-source-figure-2.webp)
*Fig 2: Task generation extracts behavioral templates from Ego4D language, chooses a compatible Robosuite scene, and writes PDDL for object placement, initial status, and the predicates that terminate the task. | source: [LIBERO, Figure 2](https://arxiv.org/abs/2306.03310)*

The generator has three linked decisions. First, language annotations from Ego4D are summarized into templates such as “Open …” and combined with simulator objects to produce an instruction. Second, a scene and initial state $\mu_0$ are written in PDDL, including object categories, placements, and statuses. Third, the goal $g$ is a conjunction of predicates such as $\operatorname{Open}(X)$, $\operatorname{TurnOff}(X)$, $\operatorname{On}(A,B)$, or $\operatorname{In}(A,B)$. The episode ends when all goal predicates are true.

That construction keeps the language, scene, and success condition tied together. It also creates an expandable task family: a new task can alter a relation or goal while reusing the same procedural ingredients. The benchmark is still simulation; the procedural generator tests transfer structure, not whether a real camera sees every object or a real gripper survives contact.

### Metrics that separate learning from forgetting

![LIBERO's FWT, NBT, and AUC metrics expose forward transfer, forgetting, and overall sequential performance](/assets/images/libero-benchmarking-knowledge-transfer-for-lifelong-robot-learning-source-figure-3.webp)
*Fig 3: The metrics track how quickly a policy learns the current task, how much success drops on earlier tasks, and the combined area under the sequential success curve. | source: [LIBERO, Figure 3](https://arxiv.org/abs/2306.03310)*

For task $i$, the paper evaluates success at epochs $e\in\{0,5,\ldots,50\}$ and selects the best current-task checkpoint. Forward transfer (FWT) averages all eleven evaluation points on the current task's learning curve, holding the values after the best checkpoint at that checkpoint's score. Higher FWT therefore rewards faster acquisition without penalizing later fluctuations past the selected checkpoint. Negative backward transfer (NBT) measures the drop on previously seen tasks, so lower means less forgetting. AUC combines current-task learning and retention, so higher is better overall. The metrics are computed from success rate rather than training loss because manipulation loss is a weak proxy for task completion.

The distinction changes the algorithm ranking. With a fixed RESNET-T policy, LIBERO-LONG gives sequential fine-tuning (SEQL) FWT $0.54\pm0.01$, NBT $0.63\pm0.01$, and AUC $0.15\pm0.00$. Experience Replay (ER) trades some forward transfer for better retention: $0.48\pm0.02$, $0.32\pm0.04$, and $0.32\pm0.01$. PACKNET pushes NBT down to $0.08\pm0.01$, but FWT falls to $0.22\pm0.01$ and AUC to $0.25\pm0.00$. In the shorter LIBERO suites, PACKNET often wins AUC; on the long suite, its partitioned capacity appears to limit new learning. EWC is worse than SEQL on several suites, showing that a regularizer can protect old parameters while still harming useful adaptation.

### The controls that prevent easy stories

The authors compare RESNET-RNN, RESNET-T, and VIT-T policies with BERT language embeddings and Gaussian-mixture action heads. On LIBERO-LONG with RESNET-T and ER, BERT obtains FWT/NBT/AUC $0.48/0.32/0.32$, CLIP $0.52/0.34/0.35$, GPT-2 $0.46/0.34/0.30$, and a Task-ID embedding $0.50/0.37/0.33$ (Table 3). None of the differences is statistically significant. In this setup, a task identifier carries as much sequential-learning signal as a semantically richer sentence embedding; that is a finding about the benchmark and encoder interface, not a general statement that language semantics never help.

Task order is another decisive control. Figure 4 evaluates ER and PACKNET with RESNET-T under five orderings, and performance varies substantially; the difference is statistically significant for PACKNET. Finally, supervised behavioral-cloning pretraining on the 90 LIBERO-90 tasks can hurt downstream LIBERO-LONG learning. A checkpoint that is good at the short tasks may already have committed capacity or representations in ways that interfere with sequential adaptation.

LIBERO therefore works best as a diagnosis of transfer decisions. Report the suite, task order, architecture, lifelong algorithm, language encoding, pretraining, and success-based metric together. A high LIBERO average without those controls cannot tell whether a policy learned new objects, preserved old skills, or benefited from an easier ordering.

## High-Level Takeaways

- LIBERO separates spatial, object, goal, and entangled transfer across 130 procedural language-conditioned tasks.
- FWT, NBT, and AUC expose the trade-off between learning the next task and retaining earlier tasks.
- PACKNET reduces forgetting on long sequences but can sacrifice forward transfer; ER is more balanced in LIBERO-LONG.
- Task order, architecture, language encoding, and supervised pretraining materially change sequential results.
- Use LIBERO to identify the failure mode, then test paraphrases, disturbances, and real-robot recovery separately.
