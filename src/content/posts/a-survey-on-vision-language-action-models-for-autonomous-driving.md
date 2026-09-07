---
title: 'A Survey on Vision-Language-Action Models for Autonomous Driving'
date: '2025-06-30T00:00:00.000Z'
section: paper-shorts
postSlug: a-survey-on-vision-language-action-models-for-autonomous-driving
legacyPath: /paper shorts/2025/06/30/a-survey-on-vision-language-action-models-for-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – A Survey on Vision-Language-Action Models for Autonomous Driving"
---

**arXiv:** [2506.24044](https://arxiv.org/abs/2506.24044)

**Awesome list:** [Awesome-VLA4AD](https://github.com/JohnsonJiang1996/Awesome-VLA4AD)

## Summary

> This survey is most useful as a map of the interface between seeing, reasoning, and driving. It formalizes the recurring building blocks of VLA4AD systems, traces the field from language as an explanation layer to language-conditioned control, and places more than 20 representative models alongside their datasets and evaluation protocols. Its warning is as useful as its taxonomy: a language rationale, an open-loop trajectory, and a safe closed-loop decision are different kinds of evidence.

## Core Insights

### A driving VLA is an interface stack, not a single model type

The survey writes a VLA4AD system as three coupled pieces: multimodal inputs, a vision-language backbone, and an action head. The inputs can include multi-view images, LiDAR, radar, BEV or occupancy features, language commands, and ego state. The backbone turns those heterogeneous signals into a shared representation; the action side then emits a description, a symbolic maneuver, a trajectory, or low-level control. This decomposition is useful because two papers can both say “VLA” while placing language at very different points in the control loop.

![Comparison of conventional end-to-end driving, VLM explanation, and VLA control](/assets/images/a-survey-on-vision-language-action-models-for-autonomous-driving-source-figure-1.webp)
*Fig 1: The survey's comparison shows the missing action link in VLM-as-explainer systems and the additional decoder that connects multimodal reasoning to driving actions in VLA4AD. | source: [A Survey on Vision-Language-Action Models for Autonomous Driving, Figure 1](https://arxiv.org/abs/2506.24044)*

That interface view also clarifies why the action representation matters. A text head can expose a decision such as “slow down,” but it still needs a controller or planner to turn the token into motion. A trajectory head is closer to execution, but it has to preserve the semantic context that made the maneuver appropriate. The survey treats this boundary as a design choice rather than assuming that more language automatically means better driving.

### The field closes the loop in four stages

The paper's historical progression is from **VLM as driving explainer**, to **modular VLA**, to **end-to-end VLA**, and then to **augmented VLA**. In the first stage, a vision-language model describes a scene or justifies a proposed maneuver while a conventional stack still drives. Modular systems make language an intermediate planning signal: a model can produce a route description, waypoint sketch, or maneuver that a separate action head executes. End-to-end systems place perception, language, and action in one differentiable path. Augmented systems add memory, tools, retrieval, or chain-of-thought style reasoning for longer-horizon decisions.

![Evolution from VLM explanation to modular, end-to-end, and augmented VLA systems](/assets/images/a-survey-on-vision-language-action-models-for-autonomous-driving-source-figure-3.webp)
*Fig 2: The four columns make the architectural transition concrete: explanation has no control output, modular systems insert an intermediate representation, end-to-end systems connect multimodal input directly to action, and augmented systems add reasoning or tools before action generation. | source: [A Survey on Vision-Language-Action Models for Autonomous Driving, Figure 3](https://arxiv.org/abs/2506.24044)*

Each step moves a failure boundary. Explainers can hallucinate a plausible rationale without changing the vehicle's action. Modular pipelines can propagate an ambiguous language decision into a downstream planner. Unified models reduce hand-written interfaces but make it harder to audit which representation caused a bad maneuver. Augmented systems can reason farther ahead, while increasing latency, memory, and verification demands. The progression is therefore about where semantics meet control, not a simple ranking of model sizes.

### The evidence is split across datasets and cannot be collapsed into one score

The survey collects complementary evaluation resources. nuScenes contains about 1,000 real-world 20-second scenes from Boston and Singapore with six cameras, LiDAR, radar, and 3D annotations; it is a common open-loop substrate for perception and trajectory error. Bench2Drive supplies 220 CARLA routes across 44 scenario types for closed-loop behavior. BDD100K and its roughly 7,000-clip BDD-X subset provide in-the-wild video and human rationales. Reason2Drive supplies about 600,000 video-text pairs for chain consistency, while Impromptu VLA contributes 80,000 thirty-second corner-case clips with trajectories, captions, and time-stamped questions.

Those resources measure different contracts. Open-loop L2, ADE, FDE, collision rate, and control errors ask whether a prediction matches a recorded future. Closed-loop route completion and infractions ask whether the policy remains useful after its own actions alter the scene. Language benchmarks ask whether the explanation or instruction response is coherent. The survey's deeper point is that a VLA should eventually report all three: trajectory competence, closed-loop safety, and language fidelity. A high-quality rationale cannot compensate for a collision, and a low open-loop error does not establish robust interaction.

## High-Level Takeaways

- Use the taxonomy to locate a paper's actual interface: where language enters, what the action head emits, and whether a VLM is present during control.
- Treat “reasoning” and “acting” as separate evidence until a study measures how the intermediate representation changes closed-loop behavior.
- Compare models only after matching sensors, data, action horizon, simulator settings, latency, and route protocol; the survey's categories do not remove those confounders.
- The most valuable benchmark would score control, safety, and instruction fidelity on the same episodes, with enough temporal context to expose long-horizon failures.
