---
title: "Top-down Traffic Scenario Generation via Joint Initial-Goal Diffusion and Trajectory Infilling"
date: '2026-08-11T00:00:00.000Z'
section: paper-shorts
postSlug: top-down-traffic-scenario-generation-via-joint-initial-goal-diffusion-and-trajectory-infilling
legacyPath: /paper shorts/2026/08/11/top-down-traffic-scenario-generation-via-joint-initial-goal-diffusion-and-trajectory-infilling.html
tags:
  - Autonomous Driving
  - Simulation
  - Diffusion Models
field: 'Motion Forecasting & Planning'
summary: "2026 – Top-down Traffic Scenario Generation via Joint Initial-Goal Diffusion and Trajectory Infilling"
---

## 2026 – Top-down Traffic Scenario Generation via Joint Initial-Goal Diffusion and Trajectory Infilling

**arXiv:** [2608.11407](https://arxiv.org/abs/2608.11407)<br />
**Code:** [TrafficDiffuser](https://github.com/CL2-UWaterloo/TrafficDiffuser)

## Summary

> TrafficDiffuser separates scenario generation from trajectory generation by modeling an agent's initial and goal states jointly. The initial-goal pair is a more interpretable high-level scenario object than an unconstrained sampled start, and fixing it turns future motion generation into trajectory infilling. On Argoverse 2, the paper reports a 55.3% reduction in speed-distribution distance and a 2.8-point reduction in off-road rate over the next-best initialization method.

## Core Insights

Many traffic simulators require initial agent states as input, which limits diversity and makes the sampled scene hard to interpret. TrafficDiffuser diffuses a set of high-level initial and goal states conditioned on map context, then uses those states as constraints for infilling trajectories. The model can condition generation on initial states, goal states, both, or neither, and can be integrated with existing trajectory generators.

The ablation exposes the value of the paired object. With both initial and goal states constrained, the reported ADE is 0.52 m and FDE and miss rate are zero in the displayed setting. Initial-only and unconstrained variants have nonzero endpoint errors, while a prior initialization method has 1.86 m ADE, 7.16 m FDE, and 43.84% miss rate. These numbers measure the chosen scenario-generation setup, not universal simulator realism.

![TrafficDiffuser overview showing joint initial-goal scenario generation followed by trajectory infilling](/assets/images/trafficdiffuser-overview-paper-figure.png)
*TrafficDiffuser first generates a high-level scenario and then infills trajectories conditioned on it. source: [TrafficDiffuser](https://arxiv.org/abs/2608.11407)*

![Figure 1 from Top-down Traffic Scenario Generation via Joint Initial-Goal Diffusion and Trajectory Infilling](/assets/images/top-down-traffic-scenario-generation-via-joint-initial-goal-diffusion-and-trajectory-infilling-source-figure-1.webp)
*Figure 1 Fig. 1: Illustration of proposed approach. (a) Generated high-level traffic scenario. (b) Infilled trajectories conditioned on the high-level scenario. source: [Top-down Traffic Scenario Generation via Joint Initial-Goal Diffusion and Trajectory Infilling](https://arxiv.org/abs/2608.11407)*

![Figure 5 from Top-down Traffic Scenario Generation via Joint Initial-Goal Diffusion and Trajectory Infilling](/assets/images/top-down-traffic-scenario-generation-via-joint-initial-goal-diffusion-and-trajectory-infilling-source-figure-5.webp)
*Figure 5 Fig. 5: Diversity in Generated High-level Traffic Scenario . Each row contains an identical map, and the number of agents, and each column shows different scenarios. Blue dots indicate the initial positions, and red stars indicate the final positions. The line connecting the two points denotes an initial-goal pair. Generated high-level scenarios show reachable pairs. More visualization is available at https://github.com/CL2-UWaterloo/TrafficDiffuser. source: [Top-down Traffic Scenario Generation via Joint Initial-Goal Diffusion and Trajectory Infilling](https://arxiv.org/abs/2608.11407)*


The decision is whether to make simulation conditions explicit before asking a model to generate motion. Joint initial-goal diffusion improves controllability, but it also transfers difficulty into goal-distribution modeling. A useful next test would measure downstream closed-loop behavior under held-out goals and compare the same trajectory generator with and without generated high-level constraints.

## High-Level Takeaways

- TrafficDiffuser informs whether scalable traffic simulation should sample interpretable initial-goal scenarios before generating detailed trajectories.
- The atomic training object is a top-down agent state pair followed by a trajectory infill sequence; map context conditions both stages.
- The high-level pair offers controllability and distribution diagnostics, but goal sampling can become the new source of bias.
- The conclusion would weaken if generated initial-goal pairs improve offline distances but reduce diversity or fail to produce useful closed-loop scenarios under held-out road layouts.
