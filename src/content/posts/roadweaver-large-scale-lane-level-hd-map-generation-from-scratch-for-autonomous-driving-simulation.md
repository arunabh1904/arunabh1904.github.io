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

## 2026 – RoadWeaver: Large-Scale Lane-Level HD Map Generation from Scratch for Autonomous Driving Simulation

**arXiv:** [2608.11580](https://arxiv.org/abs/2608.11580)

## Summary

> RoadWeaver generates complete simulation-ready lane maps rather than isolated local road fragments. Its coarse-to-fine pipeline samples a global road layout, expands it into a connected network, and then constructs lane geometry and topology. The reported maps reach 99.8% reachability, a 10.7% dead-end ratio, and 0.24 m endpoint alignment error, while generation takes 1.39–3.50 seconds.

## Core Insights

The paper treats a synthetic HD map as a graph-and-geometry object with a global contract. Earlier generators can create diverse roads but fail to maintain connectivity or lane-level usability at scale. RoadWeaver first establishes a global layout, then expands roads and lanes while preserving topological relations. This ordering makes downstream checks—reachability, dead ends, cycles, and endpoint alignment—part of the generation problem rather than a repair pass.

Compared with MetaDrive, RoadGen, and HDMapGen, the reported endpoint alignment error is 0.24 m, versus 8.17, 4.80, and 4.32 m respectively. Reachability is 99.8% and the dead-end ratio is 10.7%; the paper also reports an 85.2% cycle ratio. The result is a simulation asset benchmark, not evidence that traffic behavior on the generated graphs is realistic.

![RoadWeaver coarse-to-fine lane-level map generation pipeline](/assets/images/roadweaver-pipeline-paper-figure.png)
_The pipeline expands a global road layout into connected road geometry and lane topology. Source: [RoadWeaver](https://arxiv.org/abs/2608.11580)._

The expensive decision is where to put controllability. A global map generator can create large evaluation spaces, but its road-layout prior determines the scenarios available to a driving policy. The next experiment should evaluate policy rankings across generated and real map distributions, with human or rule-based validity checks for lane semantics and traffic control.

## High-Level Takeaways

- RoadWeaver informs whether simulator scale should come from generating complete lane graphs instead of stitching local road templates.
- The atomic object is a connected global road layout refined into lane-level geometry and topology; the output is directly consumable by a simulator.
- The reported speed and alignment gains concern map construction, while behavior diversity and policy validity remain separate evaluation axes.
- The conclusion would weaken if policies trained or ranked on RoadWeaver maps do not transfer to held-out real road topology and traffic-control patterns.
