---
title: "RoadWeaver: Large-Scale Lane-Level HD Map Generation from Scratch for Autonomous Driving Simulation"
date: '2026-08-12T00:00:00.000Z'
section: paper-shorts
postSlug: roadweaver-large-scale-lane-level-hd-map-generation-from-scratch-for-autonomous-driving-simulation
legacyPath: /paper shorts/2026/08/12/roadweaver-large-scale-lane-level-hd-map-generation-from-scratch-for-autonomous-driving-simulation.html
tags:
  - Autonomous Driving
  - HD Maps
  - Simulation
field: 'BEV Perception & Mapping'
summary: "2026 – RoadWeaver: Large-Scale Lane-Level HD Map Generation from Scratch for Autonomous Driving Simulation"
---

**arXiv:** [2608.11580](https://arxiv.org/abs/2608.11580)

## Summary

> RoadWeaver generates simulation maps by separating global road layout, local road growth, and lane-level construction. A learned discrete-token model proposes a skeleton; procedural expansion and repair turn it into a connected network; lane-building rules produce boundaries, junction connectors, and directed topology. The result is a hybrid generator whose strongest evidence concerns usable map structure: 99.8% graph reachability, 0.24 m mean lane-endpoint alignment error, and successful import of all 100 tested maps into Tactics2D. Route-planning success is measured, while closed-loop driving improvement remains a future use of the generated environments.

## Core Insights

### Generate the city's organization before deciding every lane boundary

The first stage rasterizes OSM road graphs into six channels: road probability, two orientation channels, junction and endpoint heatmaps, and a distance field. A VQ-VAE compresses this road field into discrete tokens. A conditional masked transformer then predicts token grids using a road-style code and structural controls including density, gridness, radialness, organicness, and bearing entropy.

This representation gives a local token information about the broader layout. The decoded field is reconstructed into a sparse vector skeleton, so the learned model concentrates on principal connections and road organization before detailed geometry is introduced. Training uses roughly 58,000 road-network samples from 144 cities, each approximately 2×2 km. “From scratch” describes generation without a supplied map or sensor observation; it does not mean generation without real-world training maps.

![RoadWeaver combines learned road skeletons, procedural graph expansion, and lane construction](/assets/images/roadweaver-pipeline-paper-figure.png)
*Fig 1: Read the pipeline at three scales: the learned field proposes the global skeleton, directional growth fills local roads, and geometric/topological repair creates lane-level connections. Final validity depends on all three stages. | source: [RoadWeaver, Figure 2](https://arxiv.org/abs/2608.11580)*

### Procedural repair carries part of the connectivity claim

Road growth follows a continuous direction field computed from nearby skeleton tangents. The field's major and minor eigenvectors provide locally compatible road directions; each growth front blends the nearest direction with its current heading and a style-dependent perturbation. Turn limits and snapping prevent unconstrained wandering.

Growth stops at boundaries, length limits, or an existing road. A* reconnects dangling endpoints using a cost map that favors decoded road evidence. Largest-connected-component filtering removes residual disconnected fragments, and empty regions receive additional roads. The final stage assigns lane configurations, constructs directional lanes and junction connectors, then repairs geometry and topology.

These are substantive parts of the algorithm. High final reachability cannot be attributed to the masked transformer alone when explicit reconnection and component filtering help enforce it. The paper does not provide a component ablation isolating how much validity comes from generation versus repair.

### Connectivity, route diversity, and endpoint geometry are different successes

The comparison holds approximate graph size at 35–40 nodes, counting intersections, endpoints, and turning-transition points. RoadWeaver reaches 99.9% largest-connected-component ratio and 99.8% reachability, where reachability measures whether node pairs have valid connecting paths. RoadGen reaches 100% on both, so RoadWeaver's advantage is not simply having the highest connectivity score.

| Method | Reachability | Dead-end ratio | Cycle ratio | Endpoint alignment error |
| --- | ---: | ---: | ---: | ---: |
| RoadGen | 100.0% | 12.8% | 0.0% | 4.80 m |
| HDMapGen | 56.2% | 42.9% | 46.5% | 4.32 m |
| RoadWeaver | 99.8% | 10.7% | 85.2% | 0.24 m |

Cycle ratio measures the share of nodes belonging to a cycle, giving a structural indication of alternative routes. Endpoint error measures the distance between lane endpoints intended to connect. The 94.4% reduction quoted by the paper is relative to HDMapGen's 4.32 m, not an aggregate improvement over every baseline. More cycles are useful for route variety, but the metric alone does not establish realistic urban design or challenging traffic interaction.

Density control is approximate. Two examples targeting ten nodes/km² produce 8.9 and 12.5; larger requested densities generally yield denser layouts, with saturation beyond roughly forty. Enlarging spatial extent at bounded density increases graph size without requiring a new fixed template.

### Simulator import is a necessary gate, not a driving evaluation

On the reported RTX 5090/Ryzen 9800X3D workstation, complete generation takes approximately 1.39–3.50 seconds over the tested scales. These times exclude model loading and warm-up. They measure map construction, not simulation speed or policy inference.

The deployment check generates 100 maps, imports all of them into Tactics2D without manual editing, and attempts ten routes per map. Import success is 100%; the 1,000 routing tasks succeed 98.7% of the time, with mean routing time 0.82±0.17 seconds. OSM/OpenDRIVE export and successful routing establish a useful artifact boundary: the generator produces maps a simulator can consume, rather than only plausible pictures.

The paper does not report an autonomous agent's collision rate, route completion, or learning improvement on those maps. Richer traffic semantics and roadside assets are identified as future work. The initial behavioral t-SNE also remains descriptive: separation between trajectories from different scenarios motivates topology diversity, but does not isolate its causal effect on driving behavior.

## High-Level Takeaways

- A learned global skeleton and procedural local construction divide layout diversity from geometric validity.
- Credit explicit growth, snapping, reconnection, and repair when interpreting final topology scores.
- Report connectivity, cycle structure, lane alignment, and density control separately; none alone measures simulation realism.
- Successful import and routing enable the next experiment: testing whether generated topologies reveal new closed-loop driving failures under matched traffic and policy conditions.
