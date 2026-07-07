---
title: 'UniAD: Planning-oriented Autonomous Driving'
date: '2022-12-20T05:00:00.000Z'
section: paper-shorts
postSlug: uniad-planning-oriented-autonomous-driving
legacyPath: /paper shorts/2022/12/20/uniad-planning-oriented-autonomous-driving.html
tags:
  - Other
field: BEV
summary: UniAD connects perception, prediction, occupancy, and planning in one planning-oriented driving stack, using task queries as interfaces between modules.
---
## 2022 - UniAD

**arXiv:** [2212.10156](https://arxiv.org/abs/2212.10156)

**Code:** [OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD)

**Plain-language summary:** UniAD is a dense BEV-oriented end-to-end driving system. It does not simply train detection, mapping, forecasting, occupancy, and planning heads side by side. It arranges them so upstream tasks serve the final planning objective.

That design made UniAD a reference point for "planning-oriented" driving: perception and prediction are useful because they improve the ego vehicle's planned trajectory.

## Paper Insights

UniAD uses a BEV backbone followed by a sequence of modules: TrackFormer for object tracking, MapFormer for map elements, MotionFormer for future trajectories, OccFormer for occupancy, and a planner for ego waypoints. Queries carry task-specific state between modules, so the system can represent agents, maps, future motion, occupancy, and ego planning inside one trainable pipeline.

The important modeling claim is coordination. Modular stacks can accumulate errors across task boundaries; naive multi-task stacks can optimize tasks that do not help planning. UniAD tries to make the intermediate tasks useful for the final driving decision. The caveat is that the dense BEV pipeline is heavy, which is one reason later work such as VAD and SparseDrive pushes sparse/vectorized alternatives.

![Figure 2 from UniAD showing the planning-oriented pipeline from multi-view images to perception, prediction, occupancy, and planning](/assets/images/uniad-planning-oriented-autonomous-driving-paper-figure.png)
_Figure 2 shows UniAD's pipeline: BEV features feed tracking and mapping, those queries support motion and occupancy, and the planner consumes the resulting scene knowledge. From the [UniAD paper](https://arxiv.org/abs/2212.10156), via the arXiv PDF._

**What to look at:**
- Planning is the organizing objective, not an afterthought.
- Task queries become interfaces between perception, prediction, and planning.
- Occupancy gives the planner a scene-level safety signal.

**Evals / Benchmarks / Artifacts:**

| Module | Role | Why it matters |
| ------ | ---- | -------------- |
| TrackFormer | Tracks dynamic agents | Provides agent state for future reasoning. |
| MapFormer | Predicts map elements | Grounds motion and planning in road structure. |
| MotionFormer | Forecasts multi-agent futures | Models how other actors may move. |
| OccFormer | Predicts occupancy | Adds a dense safety-oriented future signal. |
| Planner | Predicts ego waypoints | Makes the stack optimize toward driving behavior. |

**Why it mattered:** UniAD set the dense BEV end-to-end driving baseline that later vectorized and VLA systems compare themselves against.

**Take-home message:** End-to-end driving is more than putting heads on a backbone; the intermediate tasks need to be arranged around planning.
