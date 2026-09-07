---
title: "Not All History Helps: Velocity-Aware Selective Memory for Long-Horizon End-to-End Autonomous Driving"
date: '2026-08-16T00:00:00.000Z'
section: paper-shorts
postSlug: not-all-history-helps-velocity-aware-selective-memory-for-long-horizon-end-to-end-autonomous-driving
legacyPath: /paper shorts/2026/08/16/not-all-history-helps-velocity-aware-selective-memory-for-long-horizon-end-to-end-autonomous-driving.html
tags:
  - Autonomous Driving
  - Planning
  - Temporal Memory
field: 'Autonomous Driving: VLA & Planning'
summary: "2026 – Not All History Helps: Velocity-Aware Selective Memory for Long-Horizon End-to-End Autonomous Driving"
---

## 2026 – Not All History Helps: Velocity-Aware Selective Memory for Long-Horizon End-to-End Autonomous Driving

**arXiv:** [2608.15573](https://arxiv.org/abs/2608.15573)

## Summary

> StableDrive argues that planning history is useful only when it matches the current motion stage. Selective Momentum Memory gates the immediately previous plan before a causal Mamba update, while a Motion-Stage Training Scaffold teaches stage-aware behavior and is removed at inference. A fixed midpoint between the SMM-only and MSTS-trained endpoints gives one deployable planner. On full nuScenes validation it reports 1.20 m average L2, 0.66% average collision rate, and 0.85 m trajectory-consistency error; on transition-focused LT-nuScenes it reduces collision rate to 1.49% from MomAD's 3.08%. The evaluation remains offline and simulator-based.

## Core Insights

### Cache one cycle, then ask if it belongs

StableDrive does not treat the history window as a second sensor stream that should always grow. The planner produces three command branches with six trajectory modes each, twelve future waypoints at 0.5-second intervals, and caches the resulting 18 query slots after detaching them from the previous graph. Selective Momentum Memory scores those slots against the current planning state, passes the gated memory through a causal Mamba update, and injects the result as a residual. A collision-aware command rescoring step inherited from the base planner still chooses the executed branch.

The training scaffold adds a second view of the same problem. It divides motion into stationary, accelerating, cruising, and decelerating stages and feeds stage tokens through a horizon-wise Mamba. Ground-truth kinematics supervise this scaffold during training; the scaffold is retired at inference. The model therefore learns a stage-conditioned response without requiring a privileged motion label at deployment.

The released StableDrive point is a fixed midpoint, alpha = 0.5, between the SMM-only and MSTS-trained endpoints. The paper evaluates one checkpoint and one forward pass, rather than averaging two models or optimizing trajectories at test time.

![StableDrive framework with selective momentum memory and the train-and-retire motion-stage scaffold](/assets/images/not-all-history-helps-velocity-aware-selective-memory-for-long-horizon-end-to-end-autonomous-driving-source-figure-2.webp)
*Fig 1: The framework shows the current candidate queries, one-cycle cached plans, SMM gating, and the MSTS branch that shapes training but is removed before deployment. | source: [Not All History Helps, Figure 2](https://arxiv.org/abs/2608.15573)*

Read the figure from left to right as a compatibility test. The current scene proposes a new plan; the memory module decides how much of the previous plan still agrees with it; the stage scaffold teaches the model why that agreement should differ between braking and cruising. The useful object is not “more temporal context.” It is a filtered transition from one plan to the next.

### Measure long horizon where transitions occur

The paper evaluates camera-only planning on the complete nuScenes split of 700 training, 150 validation, and 150 test scenes. Each sample predicts twelve half-second waypoints across six seconds. To stress the failure mode directly, LT-nuScenes selects 16 complete scenes, 642 inference frames, and 189 scored transition targets using ground-truth kinematics. It covers sustained stationary, acceleration, and deceleration segments while excluding turns and lane changes with explicit heading and displacement thresholds. NAVSIM adds 12,146 navtest samples from 136 logs and a harder two-stage set of 225 paired groups from 76 logs.

On full nuScenes validation, StableDrive's average L2 is 1.20 m, average collision rate is 0.66%, and average trajectory-consistency error is 0.85 m. Across horizons, L2 rises from 0.28 to 2.35 m and collision rate from 0.01% to 1.78%, exposing long-horizon accumulation. On LT-nuScenes, the average L2 is 1.32 m, collision rate 1.49%, and trajectory-consistency error 0.79 m; the matched MomAD reproduction reports 1.37 m, 3.08%, and 0.85 m.

![StableDrive qualitative comparison with MomAD under long-horizon planning](/assets/images/not-all-history-helps-velocity-aware-selective-memory-for-long-horizon-end-to-end-autonomous-driving-source-figure-1.webp)
*Fig 2: The qualitative source case contrasts a stale MomAD plan that diverges and collides with StableDrive's selectively weighted history, which stays near the ground-truth path. | source: [Not All History Helps, Figure 1](https://arxiv.org/abs/2608.15573)*

The case explains why collision rate changes more than short-horizon L2. A stale plan can look locally plausible while its terminal mode is already incompatible with the current velocity. In this example, suppressing that mode before the recurrent update is consistent with avoiding the later collision; the qualitative panel does not by itself prove that causal path.

### The result is selective memory, not more memory

The endpoint ablation separates the two training ingredients. The SMM-only endpoint reports 2.00 m L2, 1.24% collision, and 1.23 m trajectory-consistency error. The MSTS-trained checkpoint with the scaffold active at inference reports 1.89 m, 1.35%, and 1.23 m; retiring the scaffold gives 1.88 m, 1.35%, and 1.22 m. The fixed midpoint improves the three averages to 1.83 m, 1.19%, and 1.20 m. These are Table VIII Panel A's 4–6 s averages under equal training budgets, not the complete 1–6 s means in Table I. The midpoint is therefore a tested operating point, while the scaffold's main role is training supervision.

The history-length comparison is a useful negative result. One frame gives 1.20 m L2, 0.66% collision, and 0.85 m trajectory-consistency error. Two frames degrade them to 1.27 m, 0.83%, and 0.93 m; four frames recover only to 1.24 m, 0.87%, and 0.89 m. Longer context is not a substitute for deciding whether a particular prior is still compatible.

The deployment cost is measurable: on one RTX 4090 the fixed model has 87.153M active parameters, 192.728G FLOPs, 192.3 ms latency, and 5.2 FPS in FP16. Those numbers are hardware-specific, and the NAVSIM protocol is non-reactive, so the paper has not yet shown that selective memory remains stable when other agents respond to the ego vehicle.

## High-Level Takeaways

- The memory decision happens at the level of one detached 18-slot plan from the previous cycle, scored against the current motion state.
- MSTS is useful as a training signal because its privileged stage labels disappear before inference; the fixed midpoint is the deployed model.
- The one-, two-, and four-frame comparison supports selective reuse over indiscriminate history accumulation.
- The unresolved deployment question is whether this gated memory stays stable when other agents respond to the ego vehicle; the current NAVSIM protocol cannot answer it.
