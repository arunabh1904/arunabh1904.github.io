---
title: >-
  AUGUstus: Autonomous Driving VLA Model with Unified Generation and
  Understanding
date: '2026-10-12T09:00:00.000Z'
section: paper-shorts
postSlug: >-
  augustus-autonomous-driving-vla-model-with-unified-generation-and-understanding
legacyPath: >-
  /paper
  shorts/2026/10/12/augustus-autonomous-driving-vla-model-with-unified-generation-and-understanding.html
tags:
  - Autonomous Driving
  - Planning
field: 'Autonomous Driving: VLA & Planning'
summary: >-
  2026 – AUGUstus: Autonomous Driving VLA Model with Unified Generation and
  Understanding
---
## 2026 – AUGUstus: Autonomous Driving VLA Model with Unified Generation and Understanding

**Paper:** [arXiv:2610.16985](https://arxiv.org/abs/2610.16985) · [Full text, v1](https://arxiv.org/html/2610.16985v1) · [Project](https://augustus-ad.github.io)

## Summary

> AUGUstus trains continuous driving actions and structured scene predictions inside one vision–language Transformer, while keeping generated text off the action's inference path. Its one-step policy reaches 94.0 PDMS on NAVSIM v1 and runs in about 73.6 ms on an H800; three steps reach 94.6. The strongest architectural result is that action-first supervision retains the reported benefit of language training without waiting roughly 4.6 seconds for a text-first rollout. These are benchmark and profiling results, with separate private-data and preliminary vehicle experiments that do not establish public-road safety.

## Core Insights

### Let language supervise action states without delaying them

Driving policies such as [SimLingo](</paper shorts/2025/03/12/simlingo-vision-only-closed-loop-autonomous-driving-with-language-action-alignment.html>) already combine language and actions; AUGUstus changes their dependency inside a shared Transformer. An autoregressive driving VLA can first narrate the scene and then predict a trajectory, but every generated word adds latency before the action becomes available. AUGUstus changes the attention pattern. Visual and textual context comes first, followed by a bidirectional block of continuous action tokens, then a causal structured prediction suffix. The language suffix can read the action states; the action states cannot read the suffix.

How can a description improve an action if the action never reads it? The source diagram's training path is the key: language loss still back-propagates through shared Transformer states, including the action representation. At inference, those learned parameters can produce the trajectory without generating the suffix.

![AUGUstus source Figure 2 comparing text-first driving with unified action-first generation and understanding](/assets/images/augustus-source-figure-2.webp)
*Fig 1: Actions precede the structured language suffix in the shared Transformer. Training couples their representations, while deployment can stop after the continuous action block. | source: [AUGUstus, Figure 2](https://arxiv.org/html/2610.16985v1#S2.F2)*

[View the figure at full resolution](/assets/images/augustus-source-figure-2.webp).

The public model uses Qwen3.5-2B. Its inputs include four frames across two seconds from three front cameras, composed into 2048-by-512 images, together with ego state, route information, and history. For NAVSIM, it predicts eight $(x,y,\text{heading})$ waypoints over four seconds; the nuScenes setup uses six waypoints over three seconds. It adds action encoding, decoding, and time-conditioned normalization modules to the shared backbone, rather than attaching a separate action Transformer or candidate scorer.

Each action waypoint is projected into the 2,048-dimensional hidden space with a sinusoidal flow-time embedding. Adaptive normalization conditions the action block on that time; context and language use time zero. The added modules total about eight million parameters. Per-dimension 1st and 99th percentiles normalize the action coordinates.

For clean trajectory $x_0$, Gaussian noise $\varepsilon$, and sampled time $t$, training constructs $x_t=(1-t)x_0+t\varepsilon$. The model predicts the clean $x_0$ with an L1 loss. At the pure-noise endpoint, a single denoising step therefore yields a full trajectory. This is continuous trajectory generation, even though the same network also supports ordinary autoregressive language.

### Joint supervision, followed by joint policy optimization

Supervised training combines action reconstruction with teacher-forced structured scene prediction. The default weights are ten for the normalized action L1 loss and one for language cross-entropy. The implementation uses bf16 and DeepSpeed ZeRO-2. Prediction targets describe objects and coarse future positions from scene annotations; they are auxiliary supervision, not evidence that a generated verbal explanation causes the chosen action.

The supervised ablation gives 89.2 PDMS without the language task and 90.6 with action-first prediction supervision. Text-first supervision also reaches 90.6, but takes roughly 4.6 seconds versus about 73.6 ms for the action-first configuration. Prediction content is more useful than the tested generic reasoning target: the latter scores 89.6. Compressing prediction into a short code-like representation yields 89.1. Thus, “add any reasoning text” is too broad an interpretation; the grounding and representation of the auxiliary task matter.

AUGUstus-GRPO then trains actions and text with the same outcome advantage. Each of 32 prompts receives eight rollouts, for 256 sequences per iteration. Rollouts share their initial flow noise within a group but vary transition noise and sampled text. A three-step stochastic action sampler supplies Gaussian transition probabilities, so the action branch has a defined likelihood ratio and KL penalty. The language branch uses ordinary clipped token-level policy optimization.

The first reward stage combines prediction quality with five times PDMS. Prediction quality combines matched-box overlap and agreement on future-position labels, under a strict output-format gate. The second stage replaces PDMS reward with endpoint accuracy and safety terms. The authors motivate this change with diminishing within-group PDMS variation and a trade-off between progress and collision-related subscores. Safety gates can zero the planning reward; the prediction reward remains separate.

The appendix specifies learning rate $10^{-6}$, clipping 0.2 for both policy heads, gradient norm one, and a frozen SFT reference throughout both stages. Freezing prefix KV during rollout is a cache behavior; it should not be interpreted as a statement that the backbone weights are frozen during training. The reward change raises reported PDMS from 94.0 after the first stage to 94.6 after the second. Removing the CoT-GRPO component gives 92.6 in the corresponding ablation, supporting joint post-training within this setup.

### The runtime claim depends on the complete execution path

| Public evaluation or profile | Reported result |
| --- | ---: |
| NAVSIM v1, one denoising step | 94.0 PDMS |
| NAVSIM v1, three steps | 94.6 PDMS |
| NAVSIM v2, one step, human-filtered / unfiltered | 90.7 / 87.1 |
| nuScenes open-loop average displacement error | 0.27 m |
| nuScenes average collision metric | 0.05% |
| H800 batch-one action-only latency | About 73.6 ms |

The filtered and unfiltered NAVSIM v2 values use different evaluation sets and should travel together. The nuScenes collision statistic comes from logged-scene trajectory evaluation, not repeated closed-loop interaction with reactive traffic.

The H800 profile spends 40.2 ms in vision processing and 33.2 ms in the language/action stage. CUDA graph replay reduces the latter from 91.83 ms to 33.19 ms; the eager end-to-end path takes 131.88 ms. The 73.6-ms result therefore depends on graph capture as well as the action-first architecture. The context contains roughly 2,257 tokens, versus only eight action tokens. Eliminating a language rollout removes a major serial cost, but does not make visual processing negligible.

The paper also reports 83 ms on its vehicle platform. Optional post-hoc explanation remains expensive: a measured 166-token suffix takes about 4.5 seconds. It can serve analysis or user interaction, but does not belong in the measured fast control path.

Private-data training and evaluation form a separate evidence track. The authors describe a run on 512 H800 GPUs for roughly two days and an internal driving test spanning about 1,000 km. Those resources and results should not be attributed to the public-only model. The internal comparison also includes a slight compliance advantage for the flow baseline, and the vehicle demonstration is preliminary and assisted. The unresolved deployment question is whether the action-first policy preserves rare-event behavior under a matched hardware budget and reactive closed-loop evaluation.

## High-Level Takeaways

- AUGUstus places continuous actions before the language suffix, allowing language loss to shape shared action representations without requiring text generation at deployment.
- Structured prediction supervision helps in the reported ablation; generic reasoning and heavily compressed targets do not provide the same gain.
- Joint GRPO supplies a shared outcome signal to Gaussian action transitions and language tokens, then changes the planning reward when PDMS becomes less informative.
- The sub-100-ms result combines one-step generation, no explanation rollout, and CUDA graph execution. Public benchmark scores, private-data scaling, and preliminary vehicle tests support different claims.
