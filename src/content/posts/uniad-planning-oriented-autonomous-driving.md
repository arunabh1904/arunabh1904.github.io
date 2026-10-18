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

## Summary

> UniAD is a dense BEV-oriented end-to-end driving system. It does not simply train detection, mapping, forecasting, occupancy, and planning heads side by side. It arranges them so upstream tasks serve the final planning objective. That design made UniAD a reference point for "planning-oriented" driving: perception and prediction are useful because they improve the ego vehicle's planned trajectory.

## Core Insights

### Queries connect the tasks to planning

UniAD uses a BEV backbone followed by a sequence of modules: TrackFormer for object tracking, MapFormer for map elements, MotionFormer for future trajectories, OccFormer for occupancy, and a planner for ego waypoints. Queries carry task-specific state between modules, so the system can represent agents, maps, future motion, occupancy, and ego planning inside one trainable pipeline.

The important modeling claim is coordination. Modular stacks can accumulate errors across task boundaries; naive multi-task stacks can optimize tasks that do not help planning. UniAD tries to make the intermediate tasks useful for the final driving decision. The caveat is that the dense BEV pipeline is heavy, which is one reason later work such as VAD and SparseDrive pushes sparse/vectorized alternatives.

The query interfaces make that coordination concrete. TrackFormer produces agent queries, MapFormer produces lane, boundary, divider, and crossing queries, and MotionFormer lets agent queries attend to both agents and maps before forecasting several modes. An ego query then participates in the same interaction and feeds the planner. OccFormer adds both a scene occupancy map and instance-level occupancy, so the planner can use a learned trajectory while checking whether it enters predicted occupied space. The design is therefore a chain of reusable scene state, not five independent heads attached to one BEV tensor.

The paper’s ablation supports the chain. In its full experiment, planning average L2 is 1.004 m and average collision is 0.430%; removing occupancy while keeping tracking, mapping, and motion raises collision to 0.717%. The dedicated planning ablation shows a trade-off. Adding BEV attention, a collision loss, and occupancy-based trajectory optimization lowers collision rates at one, two, and three seconds from 0.56/0.88/1.64% to 0.13/0.42/1.05%, while L2 error rises from 0.44/0.99/1.71 m to 0.54/1.09/1.81 m. The safer plan does not more closely imitate the logged trajectory in every metric. These are per-horizon ablation results, separate from the main full-system comparison.

The deeper design choice is that occupancy acts as a spatial veto on an otherwise plausible plan. A trajectory can imitate the logged route and still pass through a predicted occupied region; OccFormer gives the planner a scene-level and instance-level check for that failure mode. This is why the planning gain cannot be read from the waypoint regressor alone: the preceding modules change the set of trajectories that remain acceptable.

### Perception is trained before the joint model

UniAD is not trained end to end from a blank checkpoint. Stage one loads the corresponding BEVFormer weights, freezes the image backbone, and trains tracking and online mapping for six epochs. Stage two freezes the image backbone and BEV encoder as well, then trains tracking, mapping, motion, occupancy, and planning together for 20 epochs; the full loss is the sum of those five task losses. The ablations use a shorter eight-epoch schedule and three-frame sequences, so their comparisons are not the same budget as the final model. That schedule is part of the coordination claim: perception is stabilized before later tasks are asked to shape it.

The reported comparisons run on the nuScenes validation set. Motion metrics are computed on matched true positives with a 1 m matching threshold and a 2 m miss threshold; planning is scored with trajectory L2 error and collision rate at future timestamps, while occupancy uses IoU and VPQ over near and far BEV ranges. These are logged-scene trajectory metrics. The paper does not report a closed-loop route rollout or policy-level success rate, so its strong planning numbers should not be read as evidence that an ego policy remains stable after its own actions change the scene. The appendix also identifies long-tail scenes with large trucks and trailers as recurring failure cases, while noting that specialized perception systems can still lead on individual perception metrics.

Read the pipeline from left to right. The BEV feature is only the common substrate; the important arrows are the query interfaces that turn agents and map elements into motion context, then turn that context into an ego plan.

![Figure 2 from UniAD showing the planning-oriented pipeline from multi-view images to perception, prediction, occupancy, and planning](/assets/images/uniad-planning-oriented-autonomous-driving-paper-figure-crop.png)
*Fig 1: Shows UniAD's pipeline: BEV features feed tracking and mapping, those queries support motion and occupancy, and the planner consumes the resulting scene knowledge. The image is cropped from the source figure to remove its embedded paper caption; the diagram content is unchanged. | source: [UniAD paper, Figure 2](https://arxiv.org/abs/2212.10156)*

[Open figure at full size](/assets/images/uniad-planning-oriented-autonomous-driving-paper-figure-crop.png)

The qualitative outputs let you trace those interfaces through a single scene. Tracking identifies actors, motion expands each actor into possible futures, and occupancy exposes where those futures consume space. The final ego path is the end of that chain; a plausible path alone would hide which intermediate prediction made it possible.

![Figure 3 from UniAD: Planning-oriented Autonomous Driving](/assets/images/uniad-planning-oriented-autonomous-driving-source-figure-3.webp)
*Fig 2: One driving scene is shown from six surround-view cameras at left and a BEV/top-view result at right. The BEV panel combines map, occupancy, tracks, and motion/ego paths. | source: [UniAD: Planning-oriented Autonomous Driving, Figure 3](https://arxiv.org/abs/2212.10156)*

[Open figure at full size](/assets/images/uniad-planning-oriented-autonomous-driving-source-figure-3.webp)

The comparison of task arrangements clarifies what joint optimization adds. Sharing an image backbone reduces duplication, but independent heads still need a rule for connecting their outputs. UniAD makes those connections part of the model, which also means errors and competing losses can propagate through them.

![Figure 1 from UniAD: Planning-oriented Autonomous Driving](/assets/images/uniad-planning-oriented-autonomous-driving-source-figure-1.webp)
*Fig 3: The figure contrasts separate task models, a shared-backbone multi-task stack, direct end-to-end planning, and UniAD’s integrated perception-prediction-planning design. | source: [UniAD: Planning-oriented Autonomous Driving, Figure 1](https://arxiv.org/abs/2212.10156)*


## High-Level Takeaways

- UniAD informs whether perception, tracking, mapping, motion, occupancy, and planning should be optimized as separate products or as one planning-oriented query pipeline. The atomic interfaces are task queries: agent, map, motion, occupancy, and ego queries carry a shared scene state between modules while losses remain task-specific.
- Joint training makes upstream representations accountable to planning, but it also obscures which task and loss weight creates the gain.
- UniAD set the dense BEV end-to-end driving baseline that later vectorized and VLA systems compare themselves against.
- An end-to-end driving stack should arrange its intermediate tasks around planning rather than attach independent heads to a shared backbone.
