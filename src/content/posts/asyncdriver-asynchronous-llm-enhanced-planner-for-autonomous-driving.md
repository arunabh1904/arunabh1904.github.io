---
title: 'AsyncDriver: Asynchronous Large Language Model Enhanced Planner for Autonomous Driving'
date: '2024-06-20T00:00:00.000Z'
section: paper-shorts
postSlug: asyncdriver-asynchronous-llm-enhanced-planner-for-autonomous-driving
legacyPath: /paper shorts/2024/06/01/asyncdriver-asynchronous-llm-enhanced-planner-for-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2024 – AsyncDriver: Asynchronous Large Language Model Enhanced Planner for Autonomous Driving"
---
## 2024 – AsyncDriver

**arXiv:** [2406.14556](https://arxiv.org/abs/2406.14556)

## Summary

> AsyncDriver puts a language model beside a real-time planner instead of making it generate every trajectory. A Llama2-13B module reads vectorized agents, map context, and routing instructions, produces scene-associated features, and injects them into a transformer planner through an adaptive cross-attention block. The planner runs every frame while the LLM’s features are reused for an adjustable interval. On the paper’s 279-scenario nuPlan Hard20 split, the system scores 65.00, or 67.48 with the paper’s PDM scorer adaptation, while an interval of three planner frames cuts inference time by roughly 40% for about a 1% accuracy loss.

## Core Insights

AsyncDriver separates **what the route means** from **how often a vehicle must react**. The fast planner keeps its vector-map encoder and trajectory decoder. The LLM receives the ego state, up to 20 surrounding agents over 20 historical frames, global map data, and a sequence of route instructions such as “go straight in 9.01 m; turn right in 110.99 m.” Its output is a hidden feature rather than a waypoint string. That feature is injected into the planner, so numerical trajectory generation remains in the planner’s native representation.

![Figure 2 from AsyncDriver: Asynchronous Large Language Model Enhanced Planner for Autonomous Driving](/assets/images/asyncdriver-asynchronous-llm-enhanced-planner-for-autonomous-driving-paper-figure.png)
*Fig. 2: Vectorized scene information and routing instructions enter a Llama2-13B feature extractor. The resulting scene-associated feature crosses into the real-time planner through an Adaptive Injection Block; the LLM is used only during scheduled asynchronous updates. | source: [AsyncDriver, Figure 2](https://arxiv.org/abs/2406.14556)*

The injection block is more than concatenating a prompt embedding. The last LLM hidden state is projected into the planner dimension. Each decoder layer keeps its original attention over scene features, adds cross-attention to the LLM feature, and passes the result through a learnable gate initialized at zero. Early training therefore starts close to the pretrained planner; the language path is allowed to influence behavior as the gate learns that its advice is useful. Five auxiliary heads align the LLM with ego velocity/acceleration, adjacent-lane presence, traffic-light state, future lane change, and velocity decision. Those heads exist only during training, acting as alignment pressure rather than deployment modules.

![Figure 5 from AsyncDriver: Asynchronous Large Language Model Enhanced Planner for Autonomous Driving](/assets/images/asyncdriver-asynchronous-llm-enhanced-planner-for-autonomous-driving-source-figure-5.webp)
*Fig. 5: The prompt exposes the model to ego, agent, map, and route information and asks for future waypoints. The routing instruction is semantic context; the deployed trajectory still comes from the planner decoder. | source: [AsyncDriver, Figure 5](https://arxiv.org/abs/2406.14556)*

Figure 5 shows why the LLM is useful without making it the low-level controller. It can translate route and scene state into a semantic feature, but the final trajectory stays in the planner’s vector representation. That separation lets the system answer a changed instruction such as “stop” while preserving the planner’s collision and drivable-area objectives.

![Figure 1 from AsyncDriver: Asynchronous Large Language Model Enhanced Planner for Autonomous Driving](/assets/images/asyncdriver-asynchronous-llm-enhanced-planner-for-autonomous-driving-source-figure-1.webp)
*Fig. 1: The paper contrasts a fast planner, a fully serial LLM planner, and AsyncDriver. The third design lets the planner continue while the LLM refreshes high-level guidance. | source: [AsyncDriver, Figure 1](https://arxiv.org/abs/2406.14556)*

Figure 1 captures the two clocks: the planner can react every frame while the LLM refreshes guidance at a slower interval. The interval is therefore a controllable freshness-versus-latency knob, rather than a claim that semantic context is permanently valid.

The training data makes the two clocks possible. Planning-QA contains rule-generated conversions among high-level instructions, controls, and waypoints; Reasoning1K adds 1,000 GPT-4-generated reasoning examples. For fine-tuning, the authors sample 10,000 nuPlan scenarios, yielding 180,000 training and 20,000 validation frames. The future eight-second expert path supplies route instructions. The LLM is adapted with LoRA (rank 8, alpha 32), while the planner is initialized from a real-time planner trained on the same data. The combined objective includes the five alignment losses and the planner’s neighbor-mode negative log-likelihood plus ego-trajectory L1 loss.

## Reported evidence

The main evaluation is closed-loop reactive simulation on nuPlan’s Hard20 split. The authors randomly select 100 test scenarios for each of 14 official challenging types, then keep the 20 lowest-scoring cases per type with the 2023 PDM planner, producing 279 scenarios. Simulation runs at 10 Hz and each planner predicts an eight-second horizon. AsyncDriver scores 65.00, compared with GameFormer’s 62.05 and PDM-Closed’s 64.18. The paper’s AsyncDriver* variant uses PDM’s trajectory refinement/scoring convention and reaches 67.48; it should be read separately from the plain 65.00 result because the evaluation pipeline changes.

| Method | Score | Drivable | No collision | TTC |
| --- | ---: | ---: | ---: | ---: |
| GameFormer | 62.05 | 93.54 | 86.02 | 74.55 |
| AsyncDriver | 65.00 | 94.62 | 85.13 | 73.48 |
| AsyncDriver* | 67.48 | 96.77 | 87.63 | 76.70 |

The interval experiment tests whether semantic features age gracefully. LLM refresh intervals are 1, 9, 17, 29, 49, 79, and 149 frames. With a three-frame interval, the authors report nearly 40% lower inference time and about a 1% accuracy loss. Even one LLM inference per 149-frame scenario remains more than one score point above GameFormer, while its measured time approaches the real-time planner. The result supports the two-clock idea, but the robust average does not eliminate stale guidance: a route feature can remain semantically correct while an unexpected close obstacle requires a fresh scene update.

The instruction-following example makes the controllability claim tangible. Under ordinary routing instructions the vehicle slows slightly before a curve; when the instruction is changed to “stop,” its speed falls from 10.65 m/s to 1.06 m/s over six seconds even without an external obstacle. Component ablations show why the full stack matters: direct MLP waypoint regression scores 33.91, the real-time planner alone 62.01, and successive additions of Adaptive Injection, alignment heads, LoRA, and pretrained LoRA/Reasoning1K reach 62.84, 63.78, 64.03, and 65.00. Replacing the pretrained LLM with a 5,120-dimensional transformer and learnable instruction embeddings reaches only 63.59, versus 65.00 for the LLM feature extractor.

## High-Level Takeaways

- AsyncDriver’s useful abstraction is a semantic feature refreshed at one rate and a trajectory planner refreshed at another.
- Adaptive Injection preserves the planner’s spatial scene processing and uses the LLM for route meaning, rather than forcing the LLM to emit precise floating-point waypoints.
- Hard20 results and the interval sweep support the latency argument, but the benchmark is a paper-constructed difficult subset and the guidance-age test reports averages rather than worst-case stale-command failures.
- The next safety test should trigger refreshes on scene changes, hold the planner and sensor stream fixed, and report collision/TTC tails as a function of feature age, not only mean score and inference time.
