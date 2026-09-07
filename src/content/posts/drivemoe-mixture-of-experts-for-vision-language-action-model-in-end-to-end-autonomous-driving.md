---
title: 'DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving'
date: '2025-05-22T06:23:04.000Z'
section: paper-shorts
postSlug: drivemoe-mixture-of-experts-for-vision-language-action-model-in-end-to-end-autonomous-driving
legacyPath: /paper shorts/2025/05/22/drivemoe-mixture-of-experts-for-vision-language-action-model-in-end-to-end-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving"
---
## 2025 – DriveMoE

**arXiv:** [2505.16278](https://arxiv.org/abs/2505.16278)

**Project:** [DriveMoE](https://thinklab-sjtu.github.io/DriveMoE/)

## Summary

> DriveMoE adds sparse specialization at two different points in a driving VLA. A scene-specialized Vision MoE selects the camera view most relevant to the current route and context, while a skill-specialized Action MoE routes the trajectory decoder toward behavior experts. Built on a Drive-π0 baseline, the trajectory-level version reaches 74.22 driving score and 48.64% success on Bench2Drive, versus 55.85 and 30.00 for Drive-π0. The strongest evidence is the matched ablation: dynamic view selection and behavior routing improve closed-loop performance with only a small reported latency increase, but all evaluation is in CARLA simulation.

## Core Insights

### Route the camera evidence before routing the behavior

DriveMoE addresses two averaging problems that look similar but occur at different stages. Processing every surround camera equally creates a long, redundant visual sequence; using one policy head for every maneuver averages common lane following with rare emergency braking or overtaking. Its answer is to route **information** first and **behavior capacity** second.

![Figure 2 from DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving](/assets/images/drivemoe-mixture-of-experts-for-vision-language-action-model-in-end-to-end-autonomous-driving-paper-figure.webp)
*Fig 1: DriveMoE keeps fixed temporal views for motion estimation, adds a router-selected dynamic view, and sends the fused representation to a skill-specialized flow-matching action model. The trajectory is finally converted to steering, braking, and acceleration by a shared PID controller. | source: [DriveMoE, Figure 2](https://arxiv.org/abs/2505.16278)*

Drive-π0 uses two sequential front images, ego state, a fixed prompt, and a PaliGemma-3B VLM with a flow-matching trajectory module. DriveMoE retains that temporal front-view input and adds one top-ranked dynamic camera view. The vision router receives the GPS target and scene context, assigns logits to front, side, and rear cameras, and is trained with inexpensive view labels derived from route geometry and scenario rules. Projector layers concatenate the selected view with the fixed-view tokens. The point is not to compress every image into a smaller bag of patches; it is to preserve the spatial structure of the view that matters for the current maneuver.

![Figure 1 from DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving](/assets/images/drivemoe-mixture-of-experts-for-vision-language-action-model-in-end-to-end-autonomous-driving-source-figure-1.webp)
*Fig 2: The paper contrasts vanilla all-view encoding, query-based compression, and the Vision MoE, then contrasts one averaged trajectory distribution with skill-routed behavior distributions. | source: [DriveMoE, Figure 1](https://arxiv.org/abs/2505.16278)*

### Whole-trajectory routing preserves a shared behavior choice

The action router operates inside the flow-matching transformer. The token-level variant routes individual action tokens; the trajectory-level variant averages the token sequence before selecting experts, so one trajectory is treated as one behavioral unit. The latter is the paper’s default because routing a whole maneuver is more coherent than switching experts mid-trajectory. Skill labels cover merging, overtaking, emergency braking, give way, and traffic signs, with additional experiments adding parking-exit specialization. A shared expert carries common behavior while non-shared experts learn the residual modes; the appendix implementation uses one shared and six non-shared experts and selects the top three for trajectory generation.

![Figure 3 from DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving](/assets/images/drivemoe-mixture-of-experts-for-vision-language-action-model-in-end-to-end-autonomous-driving-source-figure-3.webp)
*Fig 3: The Vision MoE uses the GPS target to select a context-relevant camera. The selected dynamic view and fixed temporal views are projected and concatenated before entering the language model. | source: [DriveMoE, Figure 3](https://arxiv.org/abs/2505.16278)*

Training is staged to keep the routers from collapsing. In stage one, the Vision and Action MoEs select ground-truth experts while the routers learn jointly with the VLM and action experts. In stage two, selection switches to the router outputs, exposing the action model to its own mistakes. The reported setup trains for 12 epochs and then six more, with learning rate $5\times10^{-5}$, gradient accumulation to an effective batch size of 128, and flow-matching loss weight 1. Router losses are weighted 10 in stage one and 5 in stage two. The same PID controller is used for every method; the seventh waypoint sets desired speed and the tenth sets steering.

### Read the result together with its evaluation protocol

Bench2Drive contains 220 routes with one challenging corner case per route. The paper trains/evaluates on the official base set of 1,000 clips (950 train and 50 test/validation) and reports closed-loop driving score (DS), success rate (SR), efficiency, comfort, and open-loop average L2. All reported results are averaged over three runs.

| Method | DS ↑ | SR (%) ↑ | Efficiency | Comfort | Avg. L2 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Drive-π0 | 55.85 | 30.00 | 173.63 | 35.70 | 1.13 |
| DriveMoE token-level | 66.94 | 35.45 | 158.80 | 6.86 | 0.96 |
| DriveMoE trajectory-level | 74.22 | 48.64 | 175.96 | 15.31 | 1.01 |

Relative to Drive-π0, the default trajectory-level model improves DS by 18.37 points (32.9% relative) and SR by 18.64 percentage points (62.1% relative). In the five-ability evaluation, its mean ability is 47.91%, compared with 33.37% for Drive-π0; emergency-brake ability rises from 45.00% to 65.45%, and overtaking from 26.67% to 40.00%. These are meaningful closed-loop gains on rare behaviors, although the efficiency/comfort trade-off changes with the routed policy and the benchmark’s PID execution.

The vision ablation isolates the value of selecting a view. A fixed front-plus-back configuration reaches DS 63.26 and SR 31.82 at 260 ms, while an unsupervised dynamic view reaches 69.71/44.09. Adding explicit router supervision reaches 74.22/48.64 at the same 260 ms and 5,100 MB batch-one memory. The vision router accuracy is 88.85%; the action router accuracy is 65.40%. For action routing, the trajectory-level variant reaches 73.88 DS and 48.64% SR in the direct style comparison, versus 65.62 and 32.27 for token-level routing.

The efficiency comparison clarifies the scale of the extra machinery. Drive-π0 with two fixed views has 2,606M parameters, 3,400G FLOPs, and 240 ms latency. DriveMoE with two dynamic views has 3,008M parameters, 3,896G FLOPs, and 260 ms latency. Feeding six fixed views instead would cost 7,576G FLOPs and 700 ms in the paper’s measurement. Table 8’s two-view baseline scores 63.26 DS; it differs from the front-temporal baseline at 55.85 in the main results. The paper also lists 260 ms for that fixed front-plus-back view combination in Table 3 versus 240 ms in Table 8, so the small reported latency margin should not be treated as a precisely isolated hardware saving.

## High-Level Takeaways

- DriveMoE uses sparse specialization twice: choose the camera evidence that matters, then choose the behavior capacity that matches the maneuver.
- The trajectory-level router is the important choice for driving because a merge or emergency brake should be a coherent sequence, not a token-by-token mixture of skills.
- The matched ablations support both routers, while the large fixed-view baseline shows why simply adding more cameras is an inefficient substitute for selection.
- The current evidence is CARLA/Bench2Drive closed-loop simulation with heuristic router labels; real-road robustness, router failures under unseen maneuvers, and behavior under sensor dropout remain open tests.
