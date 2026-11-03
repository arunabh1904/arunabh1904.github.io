---
title: 'Scene-Adaptive VLA: Efficient Autonomous Driving via Dynamic Layer Routing'
date: '2026-10-17T09:00:00.000Z'
section: paper-shorts
postSlug: scene-adaptive-vla-efficient-autonomous-driving-via-dynamic-layer-routing
legacyPath: >-
  /paper
  shorts/2026/10/17/scene-adaptive-vla-efficient-autonomous-driving-via-dynamic-layer-routing.html
tags:
  - Autonomous Driving
  - Planning
field: 'Autonomous Driving: VLA & Planning'
summary: >-
  2026 – Scene-Adaptive VLA: Efficient Autonomous Driving via Dynamic Layer
  Routing
---
## 2026 – Scene-Adaptive VLA: Efficient Autonomous Driving via Dynamic Layer Routing

**Paper:** [arXiv:2610.24590](https://arxiv.org/abs/2610.24590) · [Full text, v1](https://arxiv.org/html/2610.24590v1) · [Code](https://github.com/yang-luvu/Scene-Adaptive-VLA)

## Summary

> Scene-Adaptive VLA skips language-model layers according to visual change and disagreement between recent driving predictions. Applied to SimLingo, it reduces estimated LLM computation by 53.5% and measured decision latency from 410.79 to 214.01 ms on an RTX 4090. Bench2Drive Driving Score changes from 85.07 to 84.87, while success rate falls from 67.27% to 65.45%. The result supports conditional computation, with route-following and language-quality regressions that an average driving score can obscure.

## Core Insights

### Recent prediction disagreement becomes a compute signal

The model starts from [SimLingo](</paper shorts/2025/03/12/simlingo-vision-only-closed-loop-autonomous-driving-with-language-action-alignment.html>)'s InternVL2-1B backbone: an InternViT-300M visual encoder and Qwen2-0.5B-Instruct language model. It consumes visual scene features, a driving task prompt, navigation tokens, ego speed, and separate path- and speed-waypoint queries. The adaptation adds three scene-complexity tokens plus queries for a compute budget and a layer-routing mask.

One complexity token compares the current visual representation with an exponentially updated historical scene representation. Two others summarize disagreement between predicted paths and speed profiles from the preceding two steps. Those predictions are aligned for ego translation, yaw, and time before their difference is encoded. This avoids a second model pass at the current step. It also means the controller's uncertainty signal is historical: a newly appearing hazard can precede disagreement in cached predictions.

The source diagram shows how those signals enter the same prefix that processes ordinary driving inputs, before any later layers are skipped.

![Scene-Adaptive VLA source Figure 2 showing scene and decision deviation tokens feeding budget and layer schedulers](/assets/images/scene-adaptive-vla-source-figure-2.webp)
*Fig 1: Visual change and aligned historical waypoint disagreement inform a budget query. A second scheduler chooses which later language-model layers execute within that budget. | source: [Scene-Adaptive VLA, Figure 2](https://arxiv.org/html/2610.24590v1#S3.F2)*

[View the figure at full resolution](/assets/images/scene-adaptive-vla-source-figure-2.webp).

Four language-model layers always run; twenty later layers are routable. Halfway through the prefix, a small scheduler selects one of ten discrete budget actions, with a minimum active fraction of 0.2. At the prefix boundary, another scheduler chooses a fixed number of later layers using a straight-through Gumbel-Softmax top-k mask. Unselected layers are bypassed through residual connections. The method changes which computations execute; it does not remove those weights from memory.

Relative to the existing SimLingo policy, this is a structural change in the language backbone, not a new action representation or vision–language projector. The base waypoint and language interfaces remain. The paper accounts for 512 visual tokens in its SimLingo compute estimate, but does not independently specify a complete camera-resolution, frame-sampling, or connector-training recipe; those details remain inherited from SimLingo.

### Learn routing before learning the budget

In the first stage, supervised fine-tuning samples random budgets. The budget scheduler stays frozen while scene encoders, the layer router, budget embedding, routing query, and LoRA updates in the prefix learn useful paths through the model. Smooth L1 supervises path and speed waypoints; autoregressive cross-entropy supervises language. Diversity and balance penalties discourage collapsing onto one layer pattern. Their weights are 0.05 and 0.01, alongside unit weights for waypoint and language losses.

The second stage freezes the routing pathway and trains the budget policy with GRPO. Four sampled budgets per example receive rewards that compare driving loss with a full-budget reference while penalizing compute. Three policy updates follow each sampling step. An anchor penalty keeps trainable parameters near their first-stage values; only the first half of the prefix remains adaptable because the budget decision occurs there.

This reinforcement objective uses losses on the training examples as its driving-quality signal. It should not be confused with reinforcement learning through fresh simulator rollouts. Training uses SimLingo's joint driving and language data on two RTX 4090 GPUs: approximately 90,000 first-stage steps at learning rate $3\times10^{-5}$ and batch size 6, then about 40,000 steps at $10^{-5}$ and batch size 4. The manuscript's description leaves some optimizer and inherited dataset details to the base implementation.

### Compute, latency, memory, and success move differently

The paper averages its main closed-loop results across three seeds on 220 Bench2Drive routes. Its matched SimLingo comparison is the clearest efficiency evidence:

| Measure | SimLingo | Scene-Adaptive SimLingo |
| --- | ---: | ---: |
| Estimated LLM compute per decision | 0.59 TFLOPs | 0.27 TFLOPs |
| Measured decision latency | 410.79 ms | 214.01 ms |
| Peak memory | 2.03 GB | 2.01 GB |
| Driving Score | 85.07 | 84.87 |
| Success rate | 67.27% | 65.45% |

The FLOP estimate covers the LLM component, using active parameter count and token count. It is not a whole-system energy measurement. Latency is measured with language generation enabled on an RTX 4090, and memory barely changes because skipped layers still occupy storage. Lower active computation therefore does not imply that the model fits on a much smaller device.

The router ablations support using decision history. Removing the speed-deviation token raises average budget from 44.64% to 71.89%, while success rate falls from 65.45% to 64.09%. Random budget allocation uses about 60% capacity but reaches only 62.55% success. These comparisons favor scene-conditioned allocation over simply choosing a small fixed network.

The remaining losses deserve equal attention. Route-compliance failures rise from 13 to 19 affected routes for SimLingo, and emergency-braking ability falls from 88.33 to 80.00. Collision-affected routes decrease from 55 to 52, but categories overlap and these counts do not establish safety equivalence. The language tests also show regression: at full budget, VQA SPICE falls from 56.77 to 54.91 and commentary from 38.04 to 35.37. They assess output quality, not whether an explanation causally determines a safe action.

Applying the mechanism to ORION reduces latency from 5,440.70 to 2,971.60 ms, while Driving Score drops from 77.74 to 75.88. That supports some portability across backbones, but an average speedup can still leave a policy too slow for its intended control cadence.

The deployment question is whether saved compute can be spent on a better safety margin elsewhere without worsening rare events. A useful test would match wall-clock budgets, include sudden hazards that are absent from cached predictions, and report tail latency and route-specific failures. If historical disagreement allocates capacity too late, the mean FLOP reduction would not justify the routing policy.

## High-Level Takeaways

- Scene-Adaptive VLA uses historical visual and decision variation to choose the depth of each language-model pass.
- Supervised training first learns viable layer combinations; GRPO then learns a budget from loss-based rewards and a compute penalty.
- The measured benefit is computation and latency, with almost no memory reduction. The 53.5% figure covers estimated LLM FLOPs.
- Success, route compliance, emergency braking, and language quality expose costs hidden by a nearly unchanged mean Driving Score. Sudden-hazard and tail-latency tests are the next decision boundary.
