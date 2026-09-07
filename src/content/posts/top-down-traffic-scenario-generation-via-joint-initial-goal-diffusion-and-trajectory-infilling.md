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

## Summary

> TrafficDiffuser makes a traffic scenario explicit before it generates detailed motion. A diffusion model samples each agent's initial and goal states jointly from map context; an infiller then connects those endpoints by predicting distances from a bridge between them. The paired endpoints make an initial scene interpretable and turn trajectory synthesis into a constrained infilling problem. On Argoverse 2, the authors report a 55.3% reduction in speed-distribution distance versus SceneControl and a 2.8% off-road reduction versus PD-Init, while also showing the realism tradeoffs those headline reductions hide.

## Core Insights

### Initial and goal states are a controllable scenario object

Most traffic simulators start with agent positions and ask a trajectory model to fill in behavior. TrafficDiffuser instead represents a high-level scenario as paired initial and goal states. Its TrafficGenerator denoises noisy agent states with a map summary, map polygons, temporal attention, and social attention. Because the goal is generated alongside the start, a sampled agent has an interpretable intended destination rather than an unexplained initial placement.

![TrafficDiffuser generates a high-level scenario and then infills trajectories conditioned on it](/assets/images/trafficdiffuser-overview-paper-figure.png)
*Fig 1: The framework first denoises initial-goal states and then passes the pair through an infiller that produces a kinematically feasible trajectory. | source: [Top-down Traffic Scenario Generation via Joint Initial-Goal Diffusion and Trajectory Infilling, Figure 2](https://arxiv.org/abs/2608.11407)*

The infiller constructs a bridge between the endpoints and predicts the distance from that bridge at each time step. Hard constraints can hold the initial state, goal state, both, or neither. The model can also feed generated initial states into an existing trajectory generator such as PD-Traj, so the high-level representation is an interface to downstream simulators rather than a replacement for every motion model.

![A single generated initial-goal pair becomes a set of infilled trajectories](/assets/images/top-down-traffic-scenario-generation-via-joint-initial-goal-diffusion-and-trajectory-infilling-source-figure-1.webp)
*Fig 2: The proposed approach pairs blue initial positions with red goal positions and infills the trajectories between them. | source: [Top-down Traffic Scenario Generation via Joint Initial-Goal Diffusion and Trajectory Infilling, Figure 1](https://arxiv.org/abs/2608.11407)*

### The paired endpoint makes feasibility measurable, but moves difficulty into goal sampling

On the trajectory-infilling table, constraining both initial and goal states gives ADE 0.52 m, FDE 0, and miss rate 0. Initial-only gives 0.53 m, 0.16 m, and 0.19%; goal-only gives 0.53 m, 0, and 0%; unconstrained infilling gives 0.53 m, 0.10 m, and 0.04%. The PD-Init comparison is 1.86 m ADE, 7.16 m FDE, and 43.84% miss rate. These numbers are from the displayed endpoint-conditioned setup with hard constraints, so they show that the infiller can connect a chosen pair; they do not establish that the generator samples the right pair for every traffic interaction.

The diversity and initialization table makes that boundary visible. For generated initial states, TrafficDiffuser reports collision rate 9.80%, off-road rate 5.10%, and nearest-edge distance 1.81 m. PD-Init reports 4.22%, 5.36%, and 1.64 m. For generated goals, the corresponding values are 10.6%, 7.21%, and 2.02 m; ground-truth goals are 0.50%, 2.80%, and 1.86 m. TrafficDiffuser improves some distributional distances while increasing collision rate relative to the initialization baseline. The paired representation therefore improves control and interpretation without making realism automatically monotonic.

### Diffusion supplies diversity under a fixed map, not a closed-loop simulator

The source Figure 5 holds the map and agent count fixed within each row and samples different initial-goal configurations across columns. The blue-to-red pair indicators make a useful intuition concrete: diversity is visible as different plausible endpoint assignments on the same road geometry, while the connecting line exposes whether an assignment is reachable. Guidance sampling is used for this qualitative visualization, whereas the quantitative Table I comparison is reported without guidance sampling for fairness.

![TrafficDiffuser samples diverse reachable initial-goal pairs on the same maps](/assets/images/top-down-traffic-scenario-generation-via-joint-initial-goal-diffusion-and-trajectory-infilling-source-figure-5.webp)
*Fig 3: Repeated samples on fixed maps show diversity in agent count and reachable initial-goal pairs. | source: [Top-down Traffic Scenario Generation via Joint Initial-Goal Diffusion and Trajectory Infilling, Figure 5](https://arxiv.org/abs/2608.11407)*

The authors use Argoverse 2 Motion Forecasting data, z-normalize positions and speeds, and model five categories: vehicle, bus, pedestrian, bicycle, and motorcycle. Common-sense metrics measure collision, off-road, and nearest lane-edge behavior; Jensen–Shannon divergence measures speed, lateral deviation, local density, and nearest-agent distributions. The 55.3% speed-distance reduction is relative to SceneControl's 0.16 JSD versus TrafficDiffuser's 0.068, and the off-road claim compares 5.10% with PD-Init's 5.36%. The model is not evaluated as a long-horizon closed-loop traffic world in this paper.

## High-Level Takeaways

- TrafficDiffuser makes scenario intent explicit by sampling initial and goal states together before detailed motion.
- The infiller's hard endpoint constraints explain its zero miss-rate rows; they should not be read as unconstrained driving realism.
- Initial-goal pairs expose a useful diagnostic for diversity, but goal-distribution modeling becomes the new bottleneck.
- The quantitative comparison improves speed-distribution and off-road measures while worsening collision rate relative to PD-Init, so “better scenario generation” is metric-dependent.
- The framework is most useful as a controllable scenario interface for a simulator or trajectory model; closed-loop policy behavior remains untested.
