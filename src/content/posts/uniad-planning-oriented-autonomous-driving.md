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

## Method
UniAD is a dense BEV-oriented end-to-end driving system. It does not simply train detection, mapping, forecasting, occupancy, and planning heads side by side. It arranges them so upstream tasks serve the final planning objective.

## Summary

> That design made UniAD a reference point for "planning-oriented" driving: perception and prediction are useful because they improve the ego vehicle's planned trajectory.

## Core Insights

UniAD uses a BEV backbone followed by a sequence of modules: TrackFormer for object tracking, MapFormer for map elements, MotionFormer for future trajectories, OccFormer for occupancy, and a planner for ego waypoints. Queries carry task-specific state between modules, so the system can represent agents, maps, future motion, occupancy, and ego planning inside one trainable pipeline.

The important modeling claim is coordination. Modular stacks can accumulate errors across task boundaries; naive multi-task stacks can optimize tasks that do not help planning. UniAD tries to make the intermediate tasks useful for the final driving decision. The caveat is that the dense BEV pipeline is heavy, which is one reason later work such as VAD and SparseDrive pushes sparse/vectorized alternatives.

The query interfaces make that coordination concrete. TrackFormer produces agent queries, MapFormer produces lane, boundary, divider, and crossing queries, and MotionFormer lets agent queries attend to both agents and maps before forecasting several modes. An ego query then participates in the same interaction and feeds the planner. OccFormer adds both a scene occupancy map and instance-level occupancy, so the planner can use a learned trajectory while checking whether it enters predicted occupied space. The design is therefore a chain of reusable scene state, not five independent heads attached to one BEV tensor.

The paper’s ablation supports the chain. In its full experiment, planning average L2 is 1.004 m and average collision is 0.430%; removing occupancy while keeping tracking, mapping, and motion raises collision to 0.717%. In the dedicated planning table, adding BEV attention, a collision loss, and trajectory optimization reduces average L2 from 0.56 to 0.48 m and collision from 0.88% to 0.31%. Those numbers explain why the paper calls the system planning-oriented: the intermediate predictions become useful constraints rather than merely auxiliary scores.

The deeper design choice is that occupancy acts as a spatial veto on an otherwise plausible plan. A trajectory can imitate the logged route and still pass through a predicted occupied region; OccFormer gives the planner a scene-level and instance-level check for that failure mode. This is why the planning gain cannot be read from the waypoint regressor alone: the preceding modules change the set of trajectories that remain acceptable.

Read the pipeline from left to right. The BEV feature is only the common substrate; the important arrows are the query interfaces that turn agents and map elements into motion context, then turn that context into an ego plan. The qualitative panel below is useful for the same reason: each row lets you follow one scene through tracking, mapping, forecasting, occupancy, and the final trajectory.

![Figure 2 from UniAD showing the planning-oriented pipeline from multi-view images to perception, prediction, occupancy, and planning](/assets/images/uniad-planning-oriented-autonomous-driving-paper-figure.png)
*Fig 1: Shows UniAD's pipeline: BEV features feed tracking and mapping, those queries support motion and occupancy, and the planner consumes the resulting scene knowledge. | source: [UniAD paper](https://arxiv.org/abs/2212.10156)*

![Figure 3 from UniAD: Planning-oriented Autonomous Driving](/assets/images/uniad-planning-oriented-autonomous-driving-source-figure-3.webp)
*Fig 2: The qualitative panel aligns surround-view images with BEV outputs for tracking, mapping, motion, occupancy, and the ego plan. Reading across a row shows how the same scene queries become a trajectory decision. | source: [UniAD: Planning-oriented Autonomous Driving](https://arxiv.org/abs/2212.10156)*

![Figure 1 from UniAD: Planning-oriented Autonomous Driving](/assets/images/uniad-planning-oriented-autonomous-driving-source-figure-1.webp)
*Fig 3: The figure contrasts separate task models, a shared-backbone multi-task stack, direct end-to-end planning, and UniAD’s integrated perception-prediction-planning design. | source: [UniAD: Planning-oriented Autonomous Driving](https://arxiv.org/abs/2212.10156)*




## High-Level Takeaways

- UniAD informs whether perception, tracking, mapping, motion, occupancy, and planning should be optimized as separate products or as one planning-oriented query pipeline. The atomic interfaces are task queries: agent, map, motion, occupancy, and ego queries carry a shared scene state between modules while losses remain task-specific.
- Joint training makes upstream representations accountable to planning, but it also obscures which task and loss weight creates the gain.
- UniAD set the dense BEV end-to-end driving baseline that later vectorized and VLA systems compare themselves against.
- An end-to-end driving stack should arrange its intermediate tasks around planning rather than attach independent heads to a shared backbone.
