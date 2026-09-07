---
title: 'Planning-Oriented End-to-End Autonomous Driving: Architectures, Evaluation, and Emerging Paradigms'
date: '2026-08-20T09:00:00.000Z'
section: paper-shorts
postSlug: planning-oriented-end-to-end-autonomous-driving-architectures-evaluation-and-emerging-paradigms
legacyPath: /paper shorts/2026/08/20/planning-oriented-end-to-end-autonomous-driving-architectures-evaluation-and-emerging-paradigms.html
tags:
  - Autonomous Driving
  - End-to-End Driving
  - Planning
  - Evaluation
field: 'Autonomous Driving: VLA & Planning'
summary: '2026 – a planning-oriented taxonomy that treats evaluation protocol as part of the method claim'
---

## 2026 – Planning-Oriented End-to-End Autonomous Driving

**arXiv:** [2608.20111](https://arxiv.org/abs/2608.20111)

## Summary

> This survey's useful claim is that “end-to-end” does not identify one architecture or one level of evidence. It classifies driving systems by the input representation, planning output, supervision, and evaluation contract that support the final plan. It then separates open-loop displacement matching, NAVSIM's non-reactive evaluation, and Bench2Drive's reactive simulation. The review is a structured claim map through June 2026, not a new experiment or a universal leaderboard.

## Core Insights

### Start with the planning contract

The survey asks a more precise question than whether a model has a visible perception head: what information is allowed to support the final plan, and how is that plan trained? Its four axes are input representation, planning output, supervision, and evaluation. Inputs may be raw images, learned BEV features, objects and maps, world-model latents, or language-conditioned tokens. Outputs may be controls, waypoints, trajectories, or actions generated through a latent or language interface. Supervision may be imitation, auxiliary perception and prediction, preference or reinforcement signals, or multimodal task mixtures.

This classification explains why “end-to-end” can describe systems with very different inductive bias. UniAD keeps structured BEV tasks because those tasks serve planning. A VLA may keep language because scene explanation and command conditioning shape the action interface. A world model may predict future consequences before selecting a trajectory. The visible module boundary is less informative than the contract between representation, objective, and action.

The survey's organizing change is therefore comparative. It asks which axis a new method changes, what control it uses, and whether the evaluation measures that change. A planner that adds a language head but reports only logged L2 has shown a different kind of evidence from a planner that improves a reactive collision score.

### Evaluation changes what a result means

Open-loop trajectory metrics compare a predicted path with the logged future under the state distribution already present in the dataset. They are useful for measuring geometric fidelity, but they do not ask where the policy goes after its first mistake. NAVSIM adds standardized planning and safety proxies while remaining non-reactive: other agents do not respond to the ego vehicle's chosen action. Bench2Drive runs routes in a reactive CARLA environment, so the policy's decisions can change future interactions, but the simulator and controller introduce their own transfer boundary. WOD-E2E and long-tail evaluations expose additional questions about coverage and preferences rather than resolving the same one.

![Open-loop errors can become different states under reactive evaluation](/assets/images/planning-oriented-end-to-end-autonomous-driving-architectures-evaluation-and-emerging-paradigms-source-figure-4.webp)
*Fig 1: The source sequence shows how a small open-loop deviation can lead to a different state, making non-reactive replay and closed-loop evaluation test different policy properties. | source: [Planning-Oriented End-to-End Autonomous Driving, Figure 4](https://arxiv.org/abs/2608.20111)*

The figure's causal path is the survey's most important evaluation intuition. A logged frame gives every method the same starting point, so displacement error can be compared cleanly. Once the ego plan changes the state, the next observation is policy-dependent. A low open-loop error can coexist with poor recovery, while a larger logged error can still lead to a safe reactive route. The metrics should therefore be read as different contracts, not placed in one ranking without qualification.

### Read foundation-model claims through action

The review treats world models and VLA systems as extensions of the same planning question. Visual realism is insufficient for a world model unless the predicted consequence is conditioned on an action and helps choose among plans. Language quality is insufficient for a VLA unless instructions, descriptions, or explanations alter an action-grounded decision under a controlled comparison. The atomic object is the future transition or action chunk, not the prettiness of a generated frame or the fluency of a caption.

That perspective also changes what a useful ablation looks like. A world-model paper should compare action-conditioned prediction against an equally trained visual-only predictor and measure downstream plan selection. A VLA paper should hold the visual input and planner fixed while removing language supervision or command conditioning. A structured planner should report whether its auxiliary map, object, or motion losses change the final planning metric under the same compute and data budget.

The survey is careful about its evidence boundary. It follows recent methods, benchmark repositories, leaderboards, code, and project pages through June 2026, but public methods and protocol-compatible results are overrepresented. It does not fit a meta-analysis, normalize every metric implementation, or establish research priority from publication order. Its value is a claim map: before comparing two numbers, identify what state distribution, controller, sensor setup, benchmark version, and safety wrapper produced them.

## High-Level Takeaways

- “End-to-end” is a training relationship; the useful comparison unit is the representation, objective, output, and evaluation contract that support the plan.
- Open-loop L2, non-reactive NAVSIM, and reactive Bench2Drive answer different questions and should not share one unqualified ranking.
- World models and VLA systems earn their planning claim only when future prediction or language conditioning changes an action-grounded evaluation.
- A useful comparison would publish controller, sensor, compute, seeds, and reactive recovery beside logged-trajectory accuracy so the four axes can be read together.
