---
title: 'Think at 5 Hz, Act at 20 Hz: Asynchronous Fast-Slow Vision-Language-Action Inference for Closed-Loop Driving'
date: '2026-07-17T09:00:00.000Z'
section: paper-shorts
postSlug: think-at-5-hz-act-at-20-hz-asynchronous-fast-slow-vision-language-action-inference-for-closed-loop-d
legacyPath: /paper shorts/2026/07/17/think-at-5-hz-act-at-20-hz-asynchronous-fast-slow-vision-language-action-inference-for-closed-loop-d.html
tags:
  - Autonomous Driving
  - VLA
  - Efficient Inference
field: 'Autonomous Driving: VLA & Planning'
topics:
  - embodied
  - autonomy
  - multimodal
summary: '2026 – Think at 5 Hz, Act at 20 Hz: Asynchronous Fast-Slow Vision-Language-Action Inference for Closed-Loop Driving'
---

## 2026 – Think at 5 Hz, Act at 20 Hz

**arXiv:** [2607.15621](https://arxiv.org/abs/2607.15621)

The usual way to fit a slow vision-language driving policy into a fast control loop is to run it less often and replay its last action. This paper instead separates the two clocks: a frozen 7B LMDrive backbone updates its scene representation at 5 Hz, while a trainable 337M action expert reads that representation and the current observation to predict fresh waypoints at 20 Hz.

The strongest result is not the composite driving score. Running the same expert at 10 Hz and 20 Hz leaves that score within experimental spread, but fresh 20 Hz control raises route completion from 82.1% to 94.0%, cuts route deviations from 11.3 to 4.3 per kilometer, and reduces red-light violations from 10.4 to 6.9. The architecture makes that comparison feasible because its per-tick model cost stays near 32 ms instead of growing with visual history.

## Paper Insights

The slow path turns instruction text and visual history into a persistent per-layer key-value cache. Every four control ticks, it appends four tokens for the latest frame. At every tick, the action expert contributes ten tokens—current-frame features, ego state, previous predictions, and learned waypoint queries—which cross-attend to the frozen cache at all 32 layers before a small head regresses five waypoints. The split gives language and history a slower update rate without forcing the control path to consume a stale action.

![Fast-slow driving architecture with a frozen 5 Hz backbone, persistent per-layer cache, and 20 Hz action expert](/assets/images/fast-slow-vla-architecture.png)
_The slow backbone updates a cached scene representation every four ticks; the small expert reads that cache and current state every tick. Source: Figure 2 in the [paper](https://arxiv.org/abs/2607.15621)._

Asynchrony changes the training distribution. At deployment, the cache may lag the current frame by zero to three ticks, so training randomly truncates the visible prefix by a sampled delay. Previous waypoints also receive noise and dropout. This randomized-staleness expert reaches 0.031 m validation waypoint L1 under a synchronous test, compared with 0.037 m for the same expert trained only at zero delay and 0.123 m for the frozen backbone head. The gain is therefore partly robustness regularization, not only tolerance of an old cache.

Training is deliberately narrow: 27,485 LMDrive instruction clips from CARLA town05, up to 40 frames each, five epochs on one 48 GB GPU. The 7B backbone, perception encoder, Q-Former, and original heads remain frozen; only the 337M expert is optimized. The open-loop comparison is not perfectly matched because the expert receives teacher-forced previous waypoints while the backbone head does not. Closed-loop evaluation removes that privileged signal because the expert must consume its own predictions.

| Comparison | Main result | What it isolates |
| --- | --- | --- |
| Frozen head vs randomized-staleness expert | Waypoint L1: 0.123 m → 0.031 m | A small cache-reading action head can replace repeated 7B action inference, though inputs are not fully matched. |
| LMDrive 10 Hz vs fast-slow 20 Hz | Route completion: 37.0% → 94.0% | Combined effect of the new expert and fresh control. |
| Same expert at 10 Hz vs 20 Hz | Route completion: 82.1% → 94.0% | Control freshness, holding the learned expert fixed. |
| Unseen town01 / town02 | Completion: 84.3% / 94.4% vs baseline 40.5% / 30.7% | The expert transfers across short-route layouts seen only by the frozen representation stack. |

![Latency versus history length for full recomputation and the cached action-expert path](/assets/images/fast-slow-vla-latency.png)
_Full recomputation exceeds the 50 ms control budget even at short histories, while cached per-tick inference remains nearly flat around 32 ms. Source: Figure 5 (left) in the [paper](https://arxiv.org/abs/2607.15621)._

Latency accounting needs one qualification. On an RTX 3090 Ti, median model compute is 32.4 ms per tick, including amortized cache maintenance, but the measured end-to-end agent step is 58 ms after sensor formatting and harness overhead. CARLA’s synchronous simulator still receives a new command every tick; wall-clock execution is about 17 Hz, not a demonstrated real-time 20 Hz physical system.

## Decision Lens

This paper informs whether to spend inference budget on repeatedly running a large semantic model or on a small high-rate controller over cached semantic state. Its evidence favors the latter when the large model changes slowly relative to the control loop: the matched-expert ablation attributes a substantial completion gain to action freshness, while latency stays independent of history length.

The missing control is a safety-matched comparison at equal end-to-end wall-clock budget. The 20 Hz system completes more route but has a lower infraction score than its 10 Hz expert variant, and vehicle collisions rise with the extra exposure. The conclusion would weaken if an optimized small recurrent controller over frozen features matched completion and safety without per-layer access to the 7B cache, or if retuning the baseline controller erased the freshness advantage.

At larger scale, cache bandwidth and hazard coverage become the likely constraints. Every expert layer attends to a growing frozen cache, and short single-town clips do not teach long-horizon traffic negotiation. The decisive next experiment is a multi-town, long-route study with identical training data, retuned low-level controllers, several seeds, real-time sensor overhead, and safety-normalized outcomes.

**Context:** The paper reframes fast-slow VLA design as an interface problem: preserve a slow model’s semantic state, but let a smaller policy act on fresh evidence.

**Limits:** Results come from CARLA 0.9.10, with two runs for the main comparison and single runs for the frame-skip and transfer rows. On eight unseen long routes, the method completes 85.4% of the route but accumulates enough violations to score 2.96, so short-route transfer is not evidence of road readiness.

**Takeaway:** Cache slow semantic reasoning; spend the per-tick budget on a small controller trained for stale context and fresh observations.
