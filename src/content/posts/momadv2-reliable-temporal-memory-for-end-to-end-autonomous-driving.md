---
title: 'MomADv2: Reliable Temporal Memory for End-to-End Autonomous Driving'
date: '2026-08-24T09:00:00.000Z'
section: paper-shorts
postSlug: momadv2-reliable-temporal-memory-for-end-to-end-autonomous-driving
legacyPath: /paper shorts/2026/08/24/momadv2-reliable-temporal-memory-for-end-to-end-autonomous-driving.html
tags: [Autonomous Driving]
field: 'Autonomous Driving: VLA & Planning'
summary: '2026 – MomADv2: Reliable Temporal Memory for End-to-End Autonomous Driving'
---

## 2026 – MomADv2: Reliable Temporal Memory for End-to-End Autonomous Driving

**Paper:** [arXiv:2608.23405](https://arxiv.org/abs/2608.23405) · [Full text](https://arxiv.org/html/2608.23405v1)

## Summary

> MomADv2 makes a driving planner selective about which past planning queries it reuses, then applies a small flow-matching correction to its current trajectory. The decisive ablation is that naive history raises average collision rate from 0.92% without history to 1.01%, while validated and aligned memory reduces it to 0.76%. More context is useful only when the stored state still refers to the current planning decision. The paper supports reliability filtering and bounded refinement as a combined recipe, with distinct open-loop, pseudo-simulation, and continuous closed-loop evaluations.

## Core Insights

### A recent plan can already be the wrong memory

Imagine a planner that was preparing to turn left but now receives a straight-ahead command. Its last query is recent and may be visually plausible, yet preserving its intention would be a mistake. Even without a command change, candidate ordering can change: candidate three now need not represent candidate three from the previous cycle. MomADv2 addresses the identity of remembered plans before asking a temporal model to combine them.

The memory buffer stores raw planning queries before temporal enhancement, together with their baseline command-specific trajectories. It checks scene continuity, command agreement, token continuity, and buffer availability. Valid historical candidates are then matched to the current candidate by distance between temporally shifted trajectories. This matches the physical path being proposed rather than relying on a fixed candidate index.

Storing the raw query has a second purpose. Repeatedly writing back an already enhanced state could reintroduce the same old evidence at every planning cycle and amplify an earlier mistake. The buffer instead retains the pre-enhancement state; the current temporal module decides how much of it remains useful.

### Memory strength depends on agreement and age

Valid history receives a weight that decays with trajectory disagreement and age. A separate gate also accounts for how much valid evidence exists. A selective state-space module processes this history causally, and its current output supplies a bounded residual to the planning query. A norm limit ties the permitted correction to the size of the current query.

These safeguards have different jobs. Continuity checks reject incompatible records, alignment establishes candidate correspondence, weights prefer nearby and recent plans, and bounded residuals prevent the temporal state from overwhelming the current proposal. Replacing all of them with a larger sequence model would leave the underlying correspondence problem unresolved.

The nuScenes memory ablation is unusually informative because every MomADv2 variant keeps the trajectory refiner. It tests the history policy within that shared downstream setup.

| History policy | Average collision rate, lower is better |
| --- | ---: |
| No history | 0.92% |
| Naive history reuse | 1.01% |
| Command filtering | 0.89% |
| Add continuity checks | 0.83% |
| Add trajectory alignment | 0.79% |
| Full reliability weighting | 0.76% |

The first two rows show that temporal context can actively hurt. The later rows support the cumulative reliability recipe, but are not independent factorial estimates of every component. A second ablation reaches its best NAVSIM v1 PDMS with four history entries: 89.9, compared with 88.2 for one and 87.7 for eight. Longer memory is not automatically more evidence for the current decision.

### Refine a plausible path instead of generating one from scratch

The flow model starts at the SSM-enhanced planner's trajectory. During training, that path is interpolated toward the expert trajectory, and the target velocity is the expert-minus-planner displacement. It learns a local correction problem, rather than recovering an entire path from Gaussian noise.

The source figure makes the deployment boundary clear. Expert trajectories appear in supervision on the left. At inference, the right-hand branch integrates the learned field using a few Euler steps, then limits the resulting residual with a gate. The reported implementation uses two steps.

![MomADv2 source Figure 3: supervised flow correction from a planner trajectory and bounded inference-time refinement](/assets/images/momadv2-source-figure-3.png)
*Fig 1: Training interpolates from the planner trajectory toward the expert. Inference starts from the planner's own path and applies a gated correction; no expert trajectory is available. | source: [MomADv2, Figure 3](https://arxiv.org/abs/2608.23405)*

The implementation first accumulates displacement outputs into absolute positions and normalizes them before flow training. Planner trajectories and query conditions are detached under the flow loss. Otherwise, the auxiliary objective could change the starting proposal while learning to correct it, making the refinement task a moving target. The residual gate preserves the original planner as an anchor, though it can also limit recovery when that anchor is badly wrong.

### Keep the evaluation protocols and teacher supervision separate

On NAVSIM v1, the reported baseline rises from 84.0 PDMS to 86.9 with the state-space memory and 89.9 with memory plus flow refinement. A flow-only row is absent, so this table does not establish the refiner's standalone contribution. NAVSIM v2 reports 87.9 EPDMS on navtest and 39.5 on navhard; those are different scenario and rollout settings, not successive measurements of the same difficulty.

Bench2Drive supplies a continuous closed-loop comparison. The non-distilled MomAD baseline scores 47.91 driving score and 18.11% success, versus 52.32 and 24.24% for MomADv2. The much higher 78.82 driving score and 46.50% success belong to a distilled variant. Mixing that row into a comparison with a non-distilled baseline would attribute teacher supervision to the memory mechanism.

For six-second nuScenes planning, mean L2 error across evaluated horizons is 1.21 m and average collision rate is 0.76%, compared with MomAD's 1.42 m and 0.90%. The final six-second L2 error is 2.40 m. Calling the average a six-second endpoint error would make long-horizon accuracy look substantially better than reported.

My read is that the strongest transferable idea is to validate what a memory entry means before reusing it. Agreement with a past trajectory is still not proof of correctness: an abrupt but necessary maneuver may disagree with a stable wrong plan. Testing command changes, emergency responses, and recovery after a bad anchor would establish where the reliability gate becomes too conservative.

## High-Level Takeaways

- Match historical planning candidates by shifted physical trajectories, after checking command and scene continuity; candidate indices are not durable identities.
- Write raw queries to memory so repeated enhancement does not recursively amplify old evidence.
- Naive history and excessive history both hurt in the reported ablations. Memory selection is part of the planner, not just a buffer setting.
- Flow matching learns a bounded correction from an existing proposal, with detached training conditions and no expert input at deployment.
- Separate distilled Bench2Drive results from ordinary training, and average open-loop error from the final-horizon error.
