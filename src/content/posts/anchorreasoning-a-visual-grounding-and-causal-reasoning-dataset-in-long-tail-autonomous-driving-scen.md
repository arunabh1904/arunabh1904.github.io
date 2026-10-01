---
title: 'AnchorReasoning: A Visual Grounding and Causal Reasoning Dataset in Long-Tail Autonomous Driving Scenarios'
date: '2026-09-23T09:00:00.000Z'
section: paper-shorts
postSlug: anchorreasoning-a-visual-grounding-and-causal-reasoning-dataset-in-long-tail-autonomous-driving-scen
legacyPath: /paper shorts/2026/09/23/anchorreasoning-a-visual-grounding-and-causal-reasoning-dataset-in-long-tail-autonomous-driving-scen.html
tags:
- Autonomous Driving
- Research
field: 'Autonomous Driving: VLMs & Evaluation'
summary: '2026 – AnchorReasoning: A Visual Grounding and Causal Reasoning Dataset in Long-Tail Autonomous Driving Scenarios'
---

## 2026 – AnchorReasoning: A Visual Grounding and Causal Reasoning Dataset in Long-Tail Autonomous Driving Scenarios

**Paper:** [arXiv:2609.28366](https://arxiv.org/abs/2609.28366) · [Full text and appendices](https://arxiv.org/html/2609.28366v1)

## Summary

> AnchorReasoning links selected visual evidence to element-level implications, a driving rationale, and a trajectory through structured supervision. It adds 416,119 annotated frames to WOD-E2E, but its evaluation uses 456 rated validation frames rather than that full annotation count. Grounding improves the tested models' trajectory predictions, including a smaller but consistent benefit over a nongrounded reasoning ablation. The evidence concerns logged-frame prediction and judged explanations; it does not establish that generated reasoning causally controls a deployed vehicle.

## Core Insights

### The annotation unit is a decision chain anchored to an image region

Each frame combines scene context, traffic events, selected decision-critical elements, their attributes and implications, an integrated rationale, longitudinal/lateral actions, and a future trajectory. The four element families are vehicles, vulnerable road users, obstacles, and traffic-control elements, with 19 finer types. Each selected element receives a mask/box, semantic attributes, an impact rank, and a statement of how it affects ego driving.

The distinction between an element's observed state and predicted intention is important. A parked vehicle and a pedestrian may both constrain a path, but only one may be about to move into it. The implication describes each constraint separately; the rationale then weighs them together. This provides a more inspectable training target than an ungrounded paragraph that merely mentions plausible road objects.

The source pipeline exposes both the spatial annotation and the reasoning chain. Its arrows describe how annotations are assembled, not proof of the policy's internal causal computation.

![Source Figure 1: AnchorReasoning annotation pipeline and visually grounded reasoning example](/assets/images/october-2609.28366-s1-f1.webp)
*Fig 1: Selected image regions are linked to attributes, implications, a combined rationale, and a driving plan. Human, model-assisted, and rule-based stages contribute different labels. | source: [AnchorReasoning, Figure 1](https://arxiv.org/html/2609.28366v1#S1.F1)*

[Open figure at full resolution](/assets/images/october-2609.28366-s1-f1.webp)

### How the annotations are created and checked

Driving-experienced annotators select and rank at most five important elements, prioritizing traffic controls, nearby or moving actors, and objects affecting future driving space. Human point prompts feed SAM3 to produce masks and boxes. Humans annotate states, intentions, traffic events, and road-topological relations; Qwen models assist with context, vehicle type, sign content, and temporary controls. Image-region rules determine relative bearing. HSV rules label red/green traffic lights, while yellow receives human annotation because automatic yellow recall is weaker.

Implications and rationales are generated from grounded elements, ego state, and navigation intent, with more complex cases routed to GPT-5.2. **The recipe is inconsistent across the source:** the main text names Qwen3-VL for scenes with at most two elements, while Appendix B.5 names Qwen3.5-122B and a narrower simple-scene rule. Both describe a tiered local/remote annotation pipeline, but the exact reproducible generator configuration needs clarification from the authors.

Quality control includes reviewing half of context/event samples, second-annotator checks of critical-element selection/ranking, and third-annotator resolution of substantial disagreements. Automatic tools are tested against human labels: context agreement varies from 76.51% for scenario type to 100% for time of day, while the traffic-light classifier reaches 97.81% agreement across 11,318 instances. These are human–automatic agreement measurements, not proof that either label source is infallible.

Generated actions are also compared with future trajectories using velocity and heading rules. For example, a “stop” label is inconsistent with continuing at normal speed, while “decelerate” can remain compatible with coming to a stop. Inconsistent cases are flagged for correction. This is a useful geometric consistency check, but it cannot validate every free-text causal explanation.

### The split makes the scale claim precise

The source preserves WOD-E2E's splits. All 415,663 decoded training frames from 2,037 segments are annotated. Validation includes **456 frames from 455 segments**, chosen because rater-feedback trajectories are available. Three rated candidate trajectories per frame give 1,368 human-rated trajectories. The decoded test split is not annotated. Thus the total 416,119 annotated frames spans a large training set and a much smaller evaluation set.

Only front-left, front, and front-right views are annotated. No explicit depth supervision is supplied, and repeated training frames are not independent scenes. The 395,379 decision-critical elements are deliberately selective: 93.2% have rank one or two. The task is to recover the driving-relevant subset, not to detect every visible object.

### Curriculum changes which tokens train the model

The model input is a single panorama stitched from three forward cameras, sixteen historical ego waypoints covering four seconds as text, and navigation intent. There is no temporal image sequence, LiDAR, or HD-map input. The evaluated backbones span Qwen, Cosmos-Reason2, Alpamayo, AutoVLA, and Impromptu-VLA. The work introduces a supervision recipe rather than a shared new projector architecture; inherited visual connectors and native action mechanisms vary by backbone.

Stage I first supervises coarse context and element presence, then grounding, rank, and attributes. The complete output format remains present, but inactive fields are masked from the loss. Stage II starts from the best Stage-I checkpoint, freezes the visual encoder, and adds implications, rationale, actions, and trajectory targets. Some samples receive trajectory-only loss, and rare states or action transitions are sampled more often.

Both stages use two B200 GPUs, full-parameter fine-tuning of trainable components, AdamW, effective batch 64, two epochs, cosine decay, 3% warmup, bf16, and ZeRO-2. Learning rates are $10^{-5}$ then $5\times10^{-6}$; the Stage-I visual rate has a 0.1 multiplier. No LoRA is used. Chain and trajectory token losses are independently normalized and combined with weight one, preventing their relative weight from being determined simply by output length. The recipe reports one seed, 42.

### Grounding, explanation, and trajectory quality have separate judges

The models output an image point for each element rather than a box. Matching is one-to-one, type-agnostic, and ordered by impact rank. A point inside a ground-truth mask has zero distance; otherwise the metric measures distance to the mask. The assignment radius is 1.5 times box diagonal, clipped to 60–300 pixels. Element recall and rank-one recall measure recovery under this tolerance; point-in-mask rate separately measures more exact localization. A generous assignment radius must not be mistaken for precise grounding.

GPT-5.5 judges element implications and frame rationales with binary cause/effect assessments. Trajectories are resampled to five seconds at 4 Hz for ADE/FDE. RFS measures agreement with human-rated trajectories on a 0–10 scale; Frame averages over frames, while Cluster gives scenario clusters equal weight. None of these is a closed-loop collision measurement.

The closest grounding ablation reduces Alpamayo-1.5's five-second ADE from 3.26 to 2.94 m and FDE from 7.97 to 7.06 m, while RFS Frame rises from 7.10 to 7.81. This isolates a more modest gain than the large improvements of general VLMs receiving the entire annotation recipe. The primary table reports average RFS gains of 1.60/1.63; the abstract and conclusion say 1.66/1.70. I use model-level rows rather than silently resolving that discrepancy.

Average token/latency reductions also hide regressions: Impromptu-VLA increases from 144 to 352 tokens and 1.91 to 4.46 seconds, while Qwen3-VL also becomes slightly slower. The note's practical implication is better structured supervision, not a blanket efficient-inference claim. Single-frame motion ambiguity, missing depth, judge dependence, and the small rated validation set remain important boundaries.

## High-Level Takeaways

- The dataset makes selected visual evidence traceable through attributes, implications, rationale, and trajectory supervision.
- Its large annotation count is mostly training data; evaluation uses 456 human-rated validation frames with the original WOD-E2E split boundary.
- Human, segmentation-model, language-model, and rule-derived labels have different error sources, and the annotation-generator recipe contains a source inconsistency.
- The curriculum explicitly controls visual training and token losses; it does not introduce one common VLM projector across all backbones.
- Grounding helps the controlled ablations, but open-loop trajectory gains, LLM-judged explanations, and average latency changes do not establish causal policy reasoning or universal efficiency.
