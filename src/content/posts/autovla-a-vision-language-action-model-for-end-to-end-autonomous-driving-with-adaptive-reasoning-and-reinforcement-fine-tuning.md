---
title: 'AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning'
date: '2025-06-16T17:58:50.000Z'
section: paper-shorts
postSlug: autovla-a-vision-language-action-model-for-end-to-end-autonomous-driving-with-adaptive-reasoning-and-reinforcement-fine-tuning
legacyPath: /paper shorts/2025/06/16/autovla-a-vision-language-action-model-for-end-to-end-autonomous-driving-with-adaptive-reasoning-and-reinforcement-fine-tuning.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning"
---
## 2025 – AutoVLA

**arXiv:** [2506.13757](https://arxiv.org/abs/2506.13757)

## Summary

> AutoVLA makes semantic reasoning and trajectory generation one autoregressive task. A Qwen2.5-VL-3B backbone reads three camera views, four-frame histories, navigation instructions, and ego state, then emits discrete physical action tokens. Supervised fine-tuning teaches a fast action-only mode and a slow chain-of-thought mode; GRPO-based reinforcement fine-tuning rewards driving quality while penalizing unnecessary reasoning. On NAVSIM, post-RFT one-shot PDMS is 89.11, and an oracle best-of-six reaches 92.12; on the closed-loop Bench2Drive test it reports a 78.84 driving score and 57.73% success rate.

## Core Insights

AutoVLA’s central choice is to make the language model’s action vocabulary physical. Each token represents a short movement \((\Delta x,\Delta y,\Delta\theta)\), and a K-disk codebook with \(K=2048\) tokens covers the observed motion patterns. At inference the model emits ten tokens, each covering 0.5 seconds, which decode to a five-second trajectory. The model therefore avoids asking a text decoder to spell precise floating-point waypoints, while keeping the planning interface inside the same autoregressive stream as the reasoning tokens.

![Figure 1 from AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](/assets/images/autovla-a-vision-language-action-model-for-end-to-end-autonomous-driving-with-adaptive-reasoning-and-reinforcement-fine-tuning-source-figure-1.webp)
*Fig. 1: AutoVLA maps camera streams, instructions, and ego state into a shared token sequence. The action codebook turns the next-token distribution into a physical trajectory; SFT supplies fast and slow responses, and GRPO trains the model to choose when reasoning is worth its cost. | source: [AutoVLA, Figure 1](https://arxiv.org/abs/2506.13757)*

The figure’s important boundary is between language and action space. A direct text waypoint such as “(4.21, 1.37)” has to preserve numerical precision through tokenization and decoding. An action codebook instead constrains every emitted action to a learned movement primitive. That buys feasibility and shorter output, but it makes codebook coverage a real modeling decision: a rare maneuver can be impossible to reconstruct if no token sequence represents it well.

The input is also deliberately small. AutoVLA uses front, front-left, and front-right RGB cameras, with four sequential frames per camera sampled at 2 Hz, plus high-level commands such as Turn Left or Go Straight and current velocity, acceleration, and historical actions. Qwen2.5-VL-72B generates structured reasoning annotations with four parts: scene description, critical objects, surrounding-agent intent, and best driving action. The resulting corpus contains about 45.6K nuPlan and 7.2K Waymo chain-of-thought examples, augmented with reformatted DriveLM data. SFT mixes action-only responses with reasoning-plus-action responses; the action-token loss is weighted separately, and CoT examples receive a sample weight of 40 in the reported setting.

The second mechanism is adaptive post-training. GRPO samples a group of candidate outputs for one scene, computes a group-relative advantage, and keeps the SFT model as a KL reference. Its reward is \(r=r_{\mathrm{Driving}}-\lambda_r r_{\mathrm{CoT}}\). NAVSIM uses PDMS, Waymo uses normalized ADE because RFS labels are limited, and the CoT term penalizes long reasoning chains with a sigmoid around a tolerance length. This gives the policy two ways to improve: choose a better trajectory and stop explaining when the scene is easy.

![Figure 4 from AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](/assets/images/autovla-a-vision-language-action-model-for-end-to-end-autonomous-driving-with-adaptive-reasoning-and-reinforcement-fine-tuning-source-figure-4.webp)
*Fig. 4: Increasing the mixed nuPlan/nuScenes training set from 10K to 185K samples improves planning metrics; the curves compare action-only and reasoning-augmented supervision on nuPlan and nuScenes. The useful question is where reasoning data begins to pay for its extra sequence length. | source: [AutoVLA, Figure 4](https://arxiv.org/abs/2506.13757)*

The scaling figure is more informative than a single leaderboard number. The paper reports that CoT supervision can lag action-only training at small data sizes, then becomes more useful as the set grows. That is a capacity/data interaction: a structured explanation is another distribution the model must learn, so it needs enough diverse scenes before the extra supervision improves planning rather than adding noise. RFT is applied after this full-data SFT stage, not as a substitute for the reasoning corpus.

## Reported evidence

The data mix spans 120 hours of OpenScene/nuPlan video, 4,021 twenty-second Waymo E2E segments, 1,000 nuScenes scenes, and more than 500,000 CARLA-Garage frames. NAVSIM evaluates collision, drivable area, direction, progress, TTC, and comfort through PDMS; nuScenes reports L2 and collision rate; Waymo reports RFS and ADE; Bench2Drive evaluates interactive CARLA routes.

| NAVSIM setting | PDMS | Collision | Area | Progress | TTC |
| --- | ---: | ---: | ---: | ---: | ---: |
| AutoVLA one-shot | 80.54 | 96.89 | 92.42 | 75.82 | 88.06 |
| AutoVLA post-RFT | 89.11 | 98.41 | 95.64 | 81.87 | 98.04 |
| AutoVLA best-of-six | 92.12 | 99.14 | 97.08 | 87.55 | 97.12 |

The best-of-six row uses an oracle scorer to select the best of six generated trajectories, so it measures candidate coverage in addition to single-sample policy quality. The post-RFT one-shot improvement is therefore the cleaner evidence for the learned adaptive policy: it raises PDMS by 8.57 points over one-shot SFT while improving every listed safety/progress component except comfort, which stays at 99.94. The paper also reports a CARLA Bench2Drive driving score of 78.84, 57.73% success, 146.93 efficiency, and 39.33 comfortness, compared with Orion’s 77.74/54.62/151.48/17.38 under the paper’s protocol.

The runtime trade-off is large before adaptation. Fast thinking averages 1.072 seconds (range 0.997–1.116), while slow thinking averages 10.518 seconds (7.607–13.706). RFT is reported to improve NAVSIM PDMS by 10.6% and reduce average runtime by 66.8% across 500 test scenarios. That reduction is a behavior result, not a hardware speedup: the policy emits shorter answers more often, while retaining long reasoning for cases where the reward supports it.

The tokenization ablation explains why the interface matters. At \(K=2048\), K-disk reconstruction reaches ADE 0.0182 m and FDE 0.0203 m with 99.42% movement coverage and 100% codebook usage; at \(K=4096\), reconstruction improves but usage falls to 91.46%. On NAVSIM, physical action tokens score PDMS 80.54, average L2 0.70 m, collision 0.31%, and runtime 3.95 s, versus 71.31, 0.89 m, 0.36%, and 7.65 s for text waypoints. The paper’s limitation remains material: even its near-real-time mode runs at about 1 Hz and is GPU-dependent.

## High-Level Takeaways

- AutoVLA unifies reasoning and control by constraining the output vocabulary to physically meaningful trajectory primitives.
- RFT’s strongest claim is selective reasoning: it improves one-shot NAVSIM behavior while shrinking the gap between the 1.072-second fast mode and the 10.518-second slow mode.
- Best-of-six performance is useful evidence that the model can produce good candidates, but oracle selection should not be confused with deployable single-trajectory control.
- The decisive follow-up would evaluate adaptive reasoning under a fixed wall-clock budget, measure tokenization errors on rare maneuvers, and test whether CoT length correlates with causal scene difficulty rather than annotation style.
