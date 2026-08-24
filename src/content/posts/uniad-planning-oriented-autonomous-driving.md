---
title: 'UniAD: Planning-oriented Autonomous Driving'
date: '2022-12-20T00:00:00.000Z'
section: paper-shorts
postSlug: uniad-planning-oriented-autonomous-driving
legacyPath: /paper shorts/2022/12/20/uniad-planning-oriented-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2022 – UniAD: Planning-oriented Autonomous Driving"
---
## 2022 – UniAD

**arXiv:** [2212.10156](https://arxiv.org/abs/2212.10156)

**Code:** [OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD)

### Method and reported result

UniAD is a dense BEV-oriented end-to-end driving system. It does not simply train detection, mapping, forecasting, occupancy, and planning heads side by side. It arranges them so upstream tasks serve the final planning objective.

## Summary

> That design made UniAD a reference point for "planning-oriented" driving: perception and prediction are useful because they improve the ego vehicle's planned trajectory.

## Core Insights

UniAD uses a BEV backbone followed by a sequence of modules: TrackFormer for object tracking, MapFormer for map elements, MotionFormer for future trajectories, OccFormer for occupancy, and a planner for ego waypoints. Queries carry task-specific state between modules, so the system can represent agents, maps, future motion, occupancy, and ego planning inside one trainable pipeline.

The important modeling claim is coordination. Modular stacks can accumulate errors across task boundaries; naive multi-task stacks can optimize tasks that do not help planning. UniAD tries to make the intermediate tasks useful for the final driving decision. The caveat is that the dense BEV pipeline is heavy, which is one reason later work such as VAD and SparseDrive pushes sparse/vectorized alternatives.

![Figure 2 from UniAD showing the planning-oriented pipeline from multi-view images to perception, prediction, occupancy, and planning](/assets/images/uniad-planning-oriented-autonomous-driving-paper-figure.png)
*Figure 2 shows UniAD's pipeline: BEV features feed tracking and mapping, those queries support motion and occupancy, and the planner consumes the resulting scene knowledge. From the [UniAD paper](https://arxiv.org/abs/2212.10156). source: [UniAD paper](https://arxiv.org/abs/2212.10156)*

![Figure 3 from UniAD: Planning-oriented Autonomous Driving](/assets/images/uniad-planning-oriented-autonomous-driving-source-figure-3.webp)
*Figure 3 Visualization results. We show results for all tasks in surround-view images and BEV. Predictions from motion and occupancy modules are consistent, and the ego vehicle is yielding to the front black car in this case. Each agent is illustrated with a unique color. Only top-1 and top-3 trajectories from motion forecasting are selected for visualization on image-view and BEV respectively. source: [UniAD: Planning-oriented Autonomous Driving](https://arxiv.org/abs/2212.10156)*

![Figure 1 from UniAD: Planning-oriented Autonomous Driving](/assets/images/uniad-planning-oriented-autonomous-driving-source-figure-1.webp)
*Figure 1 Comparison on the various designs of autonomous driving framework. (a) Most industrial solutions deploy separate models for different tasks. (b) The multi-task learning scheme shares a backbone with divided task heads. (c) The end-to-end paradigm unites modules in perception and prediction. Previous attempts either adopt a direct optimization on planning in (c.1) or devise the system with partial components in (c.2). Instead, we argue in (c. source: [UniAD: Planning-oriented Autonomous Driving](https://arxiv.org/abs/2212.10156)*


**What to look at:**
- Planning is the organizing objective, not an afterthought.
- Task queries become interfaces between perception, prediction, and planning.
- Occupancy gives the planner a scene-level safety signal.

### Reported evidence

| Module | Role | Why it matters |
| ------ | ---- | -------------- |
| TrackFormer | Tracks dynamic agents | Provides agent state for future reasoning. |
| MapFormer | Predicts map elements | Grounds motion and planning in road structure. |
| MotionFormer | Forecasts multi-agent futures | Models how other actors may move. |
| OccFormer | Predicts occupancy | Adds a dense safety-oriented future signal. |
| Planner | Predicts ego waypoints | Makes the stack optimize toward driving behavior. |

## High-Level Takeaways

- UniAD informs whether perception, tracking, mapping, motion, occupancy, and planning should be optimized as separate products or as one planning-oriented query pipeline. The atomic interfaces are task queries: agent, map, motion, occupancy, and ego queries carry a shared scene state between modules while losses remain task-specific.
- Joint training makes upstream representations accountable to planning, but it also obscures which task and loss weight creates the gain. The missing factorial ablation freezes or removes each query interface under matched backbone, data, and latency, then measures closed-loop rather than only open-loop metrics. At 10× task or scene complexity, gradient conflict and query bandwidth dominate. UniAD's claim would fail if a modular pipeline matched closed-loop safety and progress while allowing better independent calibration and recovery.
- UniAD set the dense BEV end-to-end driving baseline that later vectorized and VLA systems compare themselves against.
- An end-to-end driving stack should arrange its intermediate tasks around planning rather than attach independent heads to a shared backbone.
