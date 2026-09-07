---
title: 'Think at 5 Hz, Act at 20 Hz: Asynchronous Fast-Slow Vision-Language-Action Inference for Closed-Loop Driving'
date: '2026-07-17T00:00:00.000Z'
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

**arXiv:** [2607.15621](https://arxiv.org/abs/2607.15621)

## Summary

> This paper separates the slow part of a driving VLA from the part that must react every control tick. A frozen 7B LMDrive backbone updates a per-layer key-value cache at 5 Hz, while a trainable 337M action expert reads that cache, the current frame, and ego state at 20 Hz to predict fresh waypoints. Randomized cache staleness during training makes the split usable in closed-loop CARLA, where control freshness raises route completion from 82.1% to 94.0% in the matched expert ablation.

## Core Insights

### The cache turns a slow backbone into a standing scene representation

LMDrive's 7B backbone is valuable for instruction and history, but recomputing its full visual sequence takes 89–169 ms per step as history grows. That does not fit a 50 ms control tick. The paper freezes the perception encoder, Q-Former, LLaMA-7B backbone, and original action head, then appends four visual tokens per frame to a per-layer cache every four ticks. A 337M, 32-layer expert with width 512 attends into the cached keys and values and emits five waypoints at every tick. Its ten input tokens combine the current frame, a state token containing previous predictions and ego state, and five learned waypoint queries.

The design is a scheduling decision with a modeling consequence. The backbone never attends to expert tokens, so its cache is identical whether or not the expert runs. The fast path can therefore be repeated without rebuilding the slow representation. Every 0.2 seconds, an incremental four-token append refreshes the cache; instruction changes, notices, episode boundaries, or the window cap trigger a rebuild.

![Per-step model latency versus history length](/assets/images/think-at-5-hz-act-at-20-hz-asynchronous-fast-slow-vision-language-action-inference-for-closed-loop-d-source-figure-5.webp)
*Fig 1: Full recomputation grows from 59 ms at 10 frames to 169 ms at 100 frames and crosses the 50 ms tick budget, while the cached path stays nearly flat around 32 ms of model compute. | source: [Think at 5 Hz, Act at 20 Hz, Figure 5](https://arxiv.org/abs/2607.15621)*

### Training the expert on staleness is part of the method

At deployment the cache can lag the current world by up to three ticks. The expert is trained with a sampled delay, masking the backbone prefix so that the current frame sees an older cache. Its state token also receives noisy or dropped previous waypoints, which makes it learn to recover from its own imperfect history instead of relying on teacher-forced predictions alone.

That distribution match matters even in the easier synchronous test. On held-out weather frames, the frozen backbone head reaches 0.123 m validation waypoint L1. The expert trained with randomized staleness reaches 0.031 m, while an otherwise matched expert trained only with zero delay reaches 0.037 m. The 4 mm difference is not evidence that stale context is harmless; it shows that delay augmentation acts as a useful regularizer before asynchronous execution is turned on. A cache-equivalence test also finds less than 4 mm waypoint movement when monolithic prefill is replaced by incremental appends.

The open-loop comparison needs one qualification: the expert sees teacher-forced previous waypoints and a state token that the frozen backbone head does not use. The closed-loop experiment is therefore the decisive test for whether the learned expert helps after that privileged signal disappears.

### Fresh control improves completion, but exposes a safety tradeoff

On 32 LangAuto-Short routes in CARLA Town05, the public LMDrive baseline runs at 10 Hz with replayed commands and scores 28.8 driving score and 37.0% route completion. The same fast-slow expert run at the baseline's 10 Hz cadence reaches 34.0 driving score and 82.1% completion. At 20 Hz it reaches 32.9 driving score and 94.0% completion. Holding the expert fixed across the last two rows separates the effects: freshness cuts route deviations from 11.3 to 4.3 per kilometre, timeouts from 1.3 to 0.08, and red-light violations from 10.4 to 6.9, while infraction score falls from 0.45 to 0.37 and vehicle collisions rise from 3.2 to 11.2 per kilometre. The composite driving score stays within the reported run-to-run spread.

![The four CARLA evaluation towns](/assets/images/think-at-5-hz-act-at-20-hz-asynchronous-fast-slow-vision-language-action-inference-for-closed-loop-d-source-figure-3.webp)
*Fig 2: The expert is trained on short Town05 routes, transfers to unseen Town01 and Town02, and is tested on a long-route tier in Town03. | source: [Think at 5 Hz, Act at 20 Hz, Figure 3](https://arxiv.org/abs/2607.15621)*

The zero-shot transfer is encouraging but bounded. On Town01 and Town02, route completion is 84.3% and 94.4% for the fast-slow agent versus 40.5% and 30.7% for LMDrive. On eight long Town03 routes, it completes 85.4% but earns only a 2.96 driving score because collisions and red-light violations collapse its penalty factor. The paper is a latency and control-rate study in CARLA, not evidence of safe physical-vehicle deployment. Median model compute is 32.4 ms, but sensor formatting and harness overhead make the measured end-to-end step 58 ms, about 17 Hz wall-clock execution.

## High-Level Takeaways

- Spend the slow model's budget on language and history, then spend each control tick on a small expert that can see fresh evidence.
- Staleness augmentation is a training distribution correction; without it, a cache-reading controller is asked to generalize to a delay it never saw.
- The matched 10 Hz versus 20 Hz rows attribute completion and red-light improvements to freshness, while the collision increase shows why completion alone is an unsafe objective.
- The key follow-up is multi-town, long-route evaluation with retuned low-level control, several seeds, sensor overhead, and safety-normalized outcomes.
