---
title: 'DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models'
date: '2024-02-19T00:00:00.000Z'
section: paper-shorts
postSlug: drivevlm-convergence-of-autonomous-driving-and-large-vision-language-models
legacyPath: /paper shorts/2024/02/01/drivevlm-convergence-of-autonomous-driving-and-large-vision-language-models.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2024 – DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models"
---

**arXiv:** [2402.12289](https://arxiv.org/abs/2402.12289) · **Project:** [DriveVLM](https://tsinghua-mars-lab.github.io/DriveVLM/)

## Summary

> DriveVLM uses a vision-language model to describe a driving scene, analyze the objects that affect the vehicle, and generate increasingly concrete plans. DriveVLM-Dual connects that slow semantic branch to conventional 3D perception and a fast trajectory planner. The central design is the interface between them: measured geometry improves object analysis, while a VLM-generated trajectory becomes guidance that a faster planner can refine. Offline ablations and an optimized vehicle demonstration support the design at different levels of evidence.

## Core Insights

### Scene analysis asks what an object changes about the plan

DriveVLM first describes weather, time, road, and lane conditions, then identifies critical objects using categories and approximate image boxes. It analyzes relevant attributes, motion, or special behavior rather than requiring the same description for every object. A truck's oversized load, a police officer's gesture, and a blocked lane matter for different reasons; the output states their potential influence on the ego vehicle.

The planning prompt combines that scene summary with route, ego pose, and velocity. It produces a sequence from 17 meta-action categories, a more specific decision description, and numerical trajectory waypoints represented as language tokens. The intermediate description identifies an action, the object or lane involved, and when or for how long to act. This makes the transition from “construction ahead” to a particular maneuver inspectable, although a plausible explanation alone does not establish its causal faithfulness.

![DriveVLM's semantic planning branch exchanges geometry and trajectory guidance with the conventional driving pipeline](/assets/images/drivevlm-convergence-of-autonomous-driving-and-large-vision-language-models-paper-figure.png)
*Fig 1: Follow both cross-branch connections: 3D perception provides object geometry to scene analysis, while the slow trajectory guides fast refinement. The lower planner can keep operating between VLM updates. | source: [DriveVLM, Figure 1](https://arxiv.org/abs/2402.12289)*

### The dual system has two concrete interfaces

The perception interface projects detected 3D boxes into the image and matches them to the VLM's critical-object boxes using category agreement and an overlap threshold. The paper's “approximate IoU” divides intersection area by the projected detector-box area, rather than by the union. Matched objects contribute measured centers, orientations, and history as language prompts. Unmatched objects remain available through image-derived tokens, so an unfamiliar object need not disappear merely because the conventional detector lacks its class.

The planning interface passes the slow branch's trajectory to a conventional planner. For an optimization-based planner it initializes the solver; for a learned planner it becomes a query combined with other features. This is more specific than asking a controller to follow prose. It gives the fast system geometric guidance it can revise using current observations. The branches operate asynchronously, and the conventional branch can selectively accept the VLM trajectory.

### The ablations distinguish semantic analysis, geometry, and refinement

On nuScenes validation, hierarchical planning without the proposed reasoning chain has mean L2 error of 0.49 meters. Adding critical-object analysis reduces it to 0.44; adding 3D perception prompts reduces it further to 0.40. The corresponding mean predicted-trajectory collision rates are 0.36%, 0.35%, and 0.27%. Geometry contributes a clearer collision reduction than object analysis alone in this ablation.

| Planner setting | Mean L2 error, meters | Mean collision rate |
| --- | ---: | ---: |
| VAD | 0.37 | 0.14% |
| DriveVLM | 0.40 | 0.27% |
| DriveVLM-Dual with VAD | 0.31 | 0.10% |

The combined system improves over either constituent's reported row. The result also transfers to other refinement choices: pairing with UniAD changes its mean L2 from 1.03 to 0.39 meters, and pairing with an MLP changes 0.44 to 0.31. The MLP and VAD dual variants have equal average L2 but different collision rates, 0.13% versus 0.10%, so the fast planner still matters even when one distance metric matches.

These are open-loop comparisons against logged futures, averaged over the reported 1-, 2-, and 3-second horizons. Their collision percentages are not on-road incident rates, and they do not measure how an error changes subsequent observations during a rollout.

### SUP-AD measures decision-relevant description on deliberately difficult clips

SUP-AD contains 1,000 clips spanning over 40 scenario categories, mined for unusual objects and changing driving maneuvers. Annotators typically choose a keyframe 0.5–1 second before the maneuver; recorded vehicle motion supplies waypoint labels. This curated distribution emphasizes situations where an explanation may matter, rather than estimating their prevalence in ordinary driving.

![SUP-AD construction-scene annotation links workers and blocked road space to a sequence of driving actions](/assets/images/drivevlm-convergence-of-autonomous-driving-and-large-vision-language-models-source-figure-2.webp)
*Fig 2: Separate the workers' effect on speed from the construction zone's effect on the available path. The annotation connects object-specific influences to a sequence of meta-actions and a more detailed decision. | source: [DriveVLM, Figure 2](https://arxiv.org/abs/2402.12289)*

The scene-description metric uses GPT-4 to match extracted facts to annotations, with partial credit and penalties for unsupported additions. Meta-actions use sequence matching with lower penalties for selected conservative actions and alternative acceptable sequences. DriveVLM scores 0.71 on description and 0.37 on meta-actions, versus 0.38 and 0.19 for GPT-4V. The comparison mixes supervised fine-tuning with GPT-4V in-context prompting, so it evaluates the configured systems rather than isolating foundation-model quality.

### Onboard latency comes from a separate optimized configuration

The main experiments use a 9.6B Qwen-VL model. Deployment instead investigates smaller language backbones, quantization, visual-token compression, cached historical features, and speculative decoding. Two OrinX processors run the fast driving system and slow VLM branch separately; the optimized VLM branch reports 410 ms average inference time. That is not a latency measurement for the original 9.6B benchmark model.

The decoding tables also separate prefill from output generation. Vocabulary restriction and speculative decoding can greatly improve token-generation throughput while leaving substantial image/prompt processing time. A vehicle demonstration shows the branches can be integrated, but the paper does not provide a matched, large-scale closed-loop intervention or collision study that attributes on-road improvements to each reasoning stage.

## High-Level Takeaways

- The useful hybrid interface carries measured object geometry upward and a reference trajectory downward; it gives each branch a concrete job.
- Semantic analysis, 3D prompts, and fast refinement contribute differently. Their separate ablations are more informative than the combined headline alone.
- SUP-AD tests selected difficult scenes with custom language and sequence metrics; nuScenes tests offline trajectories. These support distinct claims.
- The 410 ms demonstration depends on deployment-specific compression and architecture choices. It should be kept separate from the main model's accuracy results.
