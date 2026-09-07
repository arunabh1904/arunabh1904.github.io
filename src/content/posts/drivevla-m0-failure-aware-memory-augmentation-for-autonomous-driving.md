---
title: "DriveVLA-M0: Failure-Aware Memory Augmentation for Autonomous Driving"
date: '2026-08-11T00:00:00.000Z'
section: paper-shorts
postSlug: drivevla-m0-failure-aware-memory-augmentation-for-autonomous-driving
legacyPath: /paper shorts/2026/08/11/drivevla-m0-failure-aware-memory-augmentation-for-autonomous-driving.html
tags:
  - Autonomous Driving
  - VLA
  - Test-Time Training
field: 'Autonomous Driving: VLA & Planning'
summary: "2026 – DriveVLA-M0: Failure-Aware Memory Augmentation for Autonomous Driving"
---

## 2026 – DriveVLA-M0: Failure-Aware Memory Augmentation for Autonomous Driving

**arXiv:** [2608.10413](https://arxiv.org/abs/2608.10413)<br />
**Code:** [DriveVLA-M0](https://github.com/ZebinX/DriveVLA-M0)

## Summary

> DriveVLA-M0 turns previous failures into a retrieval and correction mechanism for end-to-end driving. A latent memory stores failed cases with structural scene representations and expert trajectories; a retriever separates static road structure from dynamic agents; and a lightweight decoupled LoRA test-time update adapts the action decoder to the retrieved case. The paper reports 94.1 PDMS on NAVSIM v1 navtest and 47.0 EPDMS for its base model on NAVSIM v2 navhard.

## Core Insights

DriveVLA-M0 changes the unit of adaptation. A conventional driving VLA sees a difficult scene, predicts a trajectory, and moves on even if the trajectory was unsafe. This paper stores low-scoring cases and reuses them when a later scene looks structurally similar. The retrieved examples do not update the whole VLA permanently; they guide a temporary action-decoder update for the current scenario. That makes the method closer to a case-based correction loop than to ordinary offline fine-tuning.

### What enters the memory

The memory is built from scenarios where the base model’s predicted trajectory receives a low oracle PDM score. Each entry contains two retrieval keys—one for static map structure and one for dynamic agents—plus the action decoder’s language, ego-state, and intermediate trajectory-cluster features. Its labels are the expert trajectory and the oracle scores for the candidate trajectories. Cosine-similarity deduplication keeps nearly identical cases from filling the pool.

The compression is substantial. The base model’s last-layer representation has 2,800 tokens, each 1,536-dimensional; 16 learnable queries reduce it to a 16×256 retrieval feature. The action decoder represents its candidates as an $M\times8\times3$ trajectory cluster: $M$ proposals, eight future waypoints, and three coordinates per waypoint. The memory therefore keeps the information needed to correct planning without replaying every image token. The paper reports memory pools of about 4K cases for Base and 10K for Scale, with no retraining when the pool is expanded.

![DriveVLA-M0 memory generation and retrieval-augmented test-time training](/assets/images/drivevla-m0-overview-paper-figure.png)
*Fig 1: The upper path records low-scoring scenarios and their intermediate features; the lower path retrieves similar map/agent cases and adapts separate action-decoder branches with LoRA before selecting a trajectory. | source: [DriveVLA-M0: Failure-Aware Memory Augmentation for Autonomous Driving, Figure 2](https://arxiv.org/abs/2608.10413)*

### Retrieval is split by the kind of similarity

The retrieval model is a DINOv2-based encoder with two LoRA branches. The map branch is trained to attend to static road geometry such as lane boundaries and markings. The agent branch attends to moving vehicles and their spatial arrangement. At inference, the current front-view scene is encoded by both branches, and the paper’s basic method retrieves top-$k$ cases from each key. A cosine trigger decides whether the similarity is high enough to justify test-time adaptation; the default threshold is $\lambda=0.9$, and the default update uses three optimization steps.

The same separation controls the update. Map-retrieved cases train a static LoRA branch, while agent-retrieved cases train a dynamic branch. The LoRA weights are reinitialized for each test scenario, so one correction does not become an unreviewed permanent policy change. The final score fusion also follows the pathway: static predictions are used for road-compliance criteria such as drivable area and progress, while dynamic predictions guide collision-related criteria such as no-at-fault collision and time to collision. If neither branch returns a case above the trigger, Algorithm 1 returns the base trajectory and skips TTT.

The deployment implementation adds a useful guard for empty-road scenes. Agent-only similarity can be noisy when there are no nearby vehicles or pedestrians, so Appendix A first keeps nine map-similar candidates ($k_1=9$), then refines them with the agent branch to three ($k_2=3$). This hierarchy makes road topology the first filter and lets agent layout break ties only where that signal exists.

![DriveVLA-M0 map and agent retrieval attention](/assets/images/drivevla-m0-failure-aware-memory-augmentation-for-autonomous-driving-source-figure-3.webp)
*Fig 2: In the query scene and retrieved cases, the map embedding highlights road topology while the agent embedding concentrates on surrounding traffic; yellow denotes stronger attention than gray-purple. | source: [DriveVLA-M0: Failure-Aware Memory Augmentation for Autonomous Driving, Figure 3](https://arxiv.org/abs/2608.10413)*

### NAVSIM results and the decisive ablations

The paper evaluates NAVSIM v1, a non-reactive benchmark that scores a short planned trajectory against BEV scene abstractions, and NAVSIM v2 navhard, which adds a two-stage pseudo-closed-loop protocol and four extended metrics. Table 1 reports 92.3 PDMS for DriveVLA-M0-Base and 94.1 for DriveVLA-M0-Scale on navtest. The scaled memory model uses a front camera and exceeds the listed VLA baselines, although the table still measures the benchmark’s planned-trajectory protocol. On navhard, Table 2 reports 47.0 EPDMS for DriveVLA-M0-Base; its displayed Stage 1 and Stage 2 rows include extended-comfort values of 64.4 and 72.1, respectively.

![DriveVLA-M0 performance on navtest and navhard](/assets/images/drivevla-m0-failure-aware-memory-augmentation-for-autonomous-driving-source-figure-1.webp)
*Fig 3: The paper contrasts a classic VLA failure path with retrieval-guided correction and plots the resulting navtest and navhard scores; the retrieved route is intended to move the selected trajectory toward the safer candidates. | source: [DriveVLA-M0: Failure-Aware Memory Augmentation for Autonomous Driving, Figure 1](https://arxiv.org/abs/2608.10413)*

Table 3 isolates the retrieval key. The no-memory base is 91.0 PDMS, language-only retrieval falls to 90.7, map retrieval reaches 91.7, and map-plus-agent retrieval reaches 92.3. The map branch therefore captures more useful driving structure than a language embedding, while the agent branch adds another 0.6 over map-only and improves no-at-fault collision, progress, and time-to-collision sub-scores by 0.5 each. The gain is evidence for the split representation, not for memory retrieval in the abstract.

Table 4 then compares how the retrieved cases are injected. Offline action-decoder training for ten epochs reaches only 91.2 PDMS, full action-decoder TTT reaches 92.4, and decoupled LoRA TTT reaches 92.3. Scenario-specific adaptation matters more than simply adding failure cases to the offline mixture. Table 5 gives the trigger trade-off: a permissive $\lambda=0.7$ reaches 90.4, the default $0.9$ reaches 91.7 with map-only retrieval, $0.95$ reaches 91.4, and an overly strict $0.99$ falls to 89.4. Irrelevant cases inject noise, while too few cases leave the base failure untouched.

### The correction has a measurable cost

On one NVIDIA H20 with a 4,000-case memory, Table 6 measures 15.19 ms for retrieval, 30.79 ms for a forward pass, 26.44 ms for a decoupled-LoRA backward pass, and 55.42 ms for a full action-decoder backward pass. The lightweight update is therefore roughly half the backward cost of full TTT, but it still makes adaptation part of the driving loop. The paper’s robustness sweep also finds only small PDMS variation across the tested learning rates and one to five gradient steps; one step reaches 92.0–92.2 PDMS, which is useful when the latency budget is tight.

The central limitation is the oracle used to decide what counts as a failure and what trajectory should supervise the update. Failure-only memory can overrepresent the benchmark’s scoring function, and structural similarity does not guarantee the same interaction outcome. Held-out routes, random or success-only memories, equal-storage comparisons, and genuinely reactive vehicle evaluation are the controls needed to separate retrieval quality from oracle curation.

## High-Level Takeaways

- DriveVLA-M0 stores low-scoring scenarios, retrieves them by separate map and agent structure, and adapts a temporary action decoder with decoupled LoRA.
- Map-plus-agent retrieval raises the no-memory base from 91.0 to 92.3 PDMS (Table 3); full TTT reaches 92.4 while LoRA reaches 92.3 (Table 4).
- The default similarity trigger balances noisy and empty updates: map-only retrieval peaks at 91.7 for $\lambda=0.9$ and falls to 89.4 at $\lambda=0.99$ (Table 5).
- Retrieval and TTT are inexpensive relative to full adaptation, but the oracle-defined failure memory and pseudo-closed-loop NAVSIM protocol leave real interaction safety unresolved.
