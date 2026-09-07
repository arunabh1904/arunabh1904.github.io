---
title: 'Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving'
date: '2026-08-31T09:00:00.000Z'
section: paper-shorts
postSlug: qwen-drive-1-0-an-initial-step-towards-a-vision-language-foundation-model-for-autonomous-driving
legacyPath: /paper shorts/2026/08/31/qwen-drive-1-0-an-initial-step-towards-a-vision-language-foundation-model-for-autonomous-driving.html
tags: [Autonomous Driving]
field: 'Autonomous Driving: VLA & Planning'
summary: '2026 – Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving'
---

## 2026 – Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving

**Paper:** [arXiv:2609.00111](https://arxiv.org/abs/2609.00111) · [Full text](https://arxiv.org/html/2609.00111v1) · [Code](https://github.com/QwenLM/Qwen-Drive-1.0) · [Model](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B)

## Summary

> Qwen-Drive-1.0 adds explicit 3D perception and a trajectory expert to Qwen3.5-4B while largely retaining its general vision-language scores. Its most useful result is the separation between benchmark improvement and interactive behavior: reinforcement learning raises NAVSIM PDMS to 90.7, but in AlpaSim it halves off-road events while reducing progress. The paper supports a shared backbone with specialized outputs; it does not establish that fluent reasoning reliably controls the resulting trajectory.

## Core Insights

### Geometry and action remain explicit

A VLM can describe a vehicle without locating it accurately enough to plan around it. Qwen-Drive addresses that gap by attaching two different readers to a shared visual-language representation. A BEV perception head produces boxes, occupancy, and map segmentation. A separate Planning Expert predicts a continuous future trajectory. Neither output has to fit through the text decoder.

The perception head reads both early vision features and later VLM features. The early features are lifted along calibrated camera rays into a voxel volume, using a learned depth distribution. The later features provide contextual information to BEV queries initialized from that geometry. Occupancy decoding retains height; a flat BEV map alone would discard the difference between free space and an obstacle above the road. This is a concrete reason to preserve two feature paths rather than ask a text-oriented representation to recover every detail afterward.

The Planning Expert is a roughly 1.1B-parameter, 32-layer diffusion transformer. It conditions on cached keys and values from the VLM's eight softmax-attention layers. One cache serves four expert layers, so trajectory refinement can reuse scene context. The output contains 50 position-and-heading waypoints over five seconds. Flow matching teaches the expert to recover a clean trajectory from an interpolation between noise and the recorded future; inference uses ten Euler steps.

### Sharing representations does not mean training everything together

The training diagram makes the gradient boundary visible. First the new perception head learns against a frozen backbone. Joint perception and VQA training then updates the backbone. Only after that does the trajectory expert learn, with the visual-language representation fixed again.

![Qwen-Drive source Figure 4, Stage 2: joint adaptation of perception and vision-language components](/assets/images/qwen-drive-source-figure-4-stage-2.png)
*Fig 1: Stage 2 updates the shared visual-language pathway using geometric and language supervision. The source figure is cropped to this training stage. | source: [Qwen-Drive, Figure 4](https://arxiv.org/abs/2609.00111)*

The distinction matters when interpreting the paper's claim of a unified model. Perception losses can change the shared features during Stage 2; planning and reinforcement rewards do not continue updating those features in Stages 3 and 4. The method demonstrates compatibility between tasks with staged adaptation. It does not demonstrate unrestricted joint optimization of perception, language, and control.

![Qwen-Drive source Figure 4, Stage 3: a trainable Planning Expert reads a frozen vision-language model](/assets/images/qwen-drive-source-figure-4-stage-3.png)
*Fig 2: Stage 3 trains trajectory generation on fixed scene representations; the optional reasoning trace is a condition, not a text-generation target. Cropped from the source training diagram. | source: [Qwen-Drive, Figure 4](https://arxiv.org/abs/2609.00111)*

Data alignment is part of the method. Two occupancy datasets can both use a $200\times200\times16$ array while assigning different physical positions to the same index. Qwen-Drive keeps their native label grids and samples predicted features onto each grid. That avoids silently treating equal tensor shapes as equal geometry. It also masks source-specific class supervision rather than treating an unannotated class as absent.

Stage 2's effective mixture is 12.7% perception, 31.0% general vision-language, and 56.3% driving vision-language examples. Stage 3 uses about 2.83M planning examples, with a reasoning condition for 24.2%. On the paper's knowledge, reasoning, and recognition aggregate, the adapted model scores 66.41 against the base model's 67.40. Retention is measured here; it is not inferred from leaving the architecture unchanged.

### Better reward scores can conceal a different driving style

The reinforcement stage changes how candidate trajectories are explored. Independent noise at every waypoint mostly creates jitter. Qwen-Drive instead perturbs six low-frequency cosine modes over the final three integration steps. A smooth basis can bend or shift a path coherently, producing alternatives worth comparing with task rewards. Group-relative rewards then train only the Planning Expert. The paper explicitly treats its restoring correction as approximate because the noise occupies a low-dimensional subspace.

The resulting gains depend on the evaluation. NAVSIM queries the planner once and replays other agents, whereas AlpaSim repeatedly replans as the vehicle's actions change its observations. The following results compare the SFT model with reasoning against the RL model, using Tables 4, 6, and 7.

| Evaluation | SFT | RL | What the change establishes |
| --- | ---: | ---: | --- |
| WOD-E2E test Rater Feedback Score, higher is better | 7.78 | 7.91 | Better alignment with rated candidate trajectories |
| NAVSIM single-trajectory PDMS, higher is better | 88.2 | 90.7 | Better performance under a non-reactive scoring protocol |
| AlpaSim off-road rate | 24% | 12% | Fewer roadway departures across 916 simulated scenarios |
| AlpaSim progress | 54% | 48% | Less scenario progress accompanies the reduction in departures |
| AlpaSim all-event close encounter rate | 38% | 41% | Not every interaction measure improves |

The headline NAVSIM result therefore needs the closed-loop rows beside it. A more conservative planner can reduce one kind of failure while completing less of the driving task. The reported 91.4 PDMS uses the highest-scoring candidate among six; it is an oracle selection result, not the ordinary single-trajectory score.

### The interesting uncertainty is whether geometry and reasoning cause better plans

The controlled Stage 2 comparison raises downstream WOD-E2E validation RFS from 7.91 with VQA adaptation to 7.96 when 3D supervision is added. The authors themselves stop short of attributing that small gain to geometric supervision. Similarly, adding reasoning moves held-out RFS from 7.76 to 7.78. A plausible rationale is not yet evidence that the action obeys it: the limitations section reports mismatches between rationale and trajectory, particularly when causes act at different time scales.

My read is that the reusable contribution is the separation of representations, output heads, and training responsibilities. The next decisive test would hold the planning data and compute fixed, intervene on a relevant geometric fact or rationale, and measure whether the closed-loop action changes correctly without sacrificing progress. More persuasive text alone would not settle that question.

## High-Level Takeaways

- A shared VLM can support language, explicit 3D outputs, and continuous trajectories while retaining much of its general capability; the specialized heads remain important parts of that result.
- Matching tensor dimensions across datasets does not align coordinates or annotation semantics. Feature resampling and selective supervision make the mixture usable.
- Smooth trajectory exploration changes plausible maneuvers instead of spending reward comparisons on waypoint jitter.
- Read the 90.7 NAVSIM score together with AlpaSim's lower off-road rate, lower progress, and higher all-event close encounter rate. The policy's behavior changed in several directions.
- The reported controls support compatibility between geometry, language, and planning. They leave the causal contribution of explicit geometry and generated reasoning unresolved.
