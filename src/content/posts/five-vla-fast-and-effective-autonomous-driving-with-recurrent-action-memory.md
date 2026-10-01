---
title: 'FIVE-VLA: Fast and EffectIVE Autonomous Driving with Recurrent Action Memory'
date: '2026-09-16T09:00:00.000Z'
section: paper-shorts
postSlug: five-vla-fast-and-effective-autonomous-driving-with-recurrent-action-memory
legacyPath: /paper shorts/2026/09/16/five-vla-fast-and-effective-autonomous-driving-with-recurrent-action-memory.html
tags:
- Autonomous Driving
- Research
field: 'Autonomous Driving: VLA & Planning'
summary: '2026 – FIVE-VLA: Fast and EffectIVE Autonomous Driving with Recurrent Action Memory'
---

## 2026 – FIVE-VLA: Fast and EffectIVE Autonomous Driving with Recurrent Action Memory

**Paper:** [arXiv:2609.18623](https://arxiv.org/abs/2609.18623) · [Full text and appendix](https://arxiv.org/html/2609.18623v1)

## Summary

> FIVE-VLA combines a compact visual encoder, trajectory prediction without autoregressive text generation, and recurrent memory of previous latent action tokens. Its 641M-parameter policy reaches 77.27% Bench2Drive success and 29.96 fps on an A100. The useful design lesson is where temporal information enters: a small action-memory module preserves the pretrained visual-language interface. Memory improves driving but consumes additional compute, and the real-data experiments use non-reactive replay rather than closed-loop road driving.

## Core Insights

### Compress the visual input before asking the language model to process it

FIVE-VLA starts from FastVLM: a 125M-parameter FastViTHD vision encoder coupled to a Qwen2 language model. A single front image at **448 × 896** produces **98 visual tokens**, compared with SimLingo's 512. FastViTHD progressively downsamples using strided convolutions before attention; it processes the entire image rather than encoding two independent square tiles. Four-camera experiments yield 392 visual tokens instead of 2,048.

The policy receives the image, ego velocity expressed as text, two future navigation target points, and a task instruction. Target points have their own adapter. Learnable action queries feed separate MLP decoders for path and speed waypoints. Path waypoints describe positions at one-metre intervals; speed waypoints describe positions at fixed future times, so their errors are position errors in metres, not scalar speed errors. Output counts depend on the dataset: the NVIDIA evaluation uses 30 speed waypoints over three seconds and 30 path waypoints over 30 metres.

![Source Figure 2: FIVE-VLA visual backbone, action queries, and recurrent action memory](/assets/images/october-2609.18623-s3-f2.webp)
*Fig 1: Recurrent memory modifies the action-query input while retaining the pretrained visual encoder and visual-language adapter. Separate decoders translate contextualized action tokens into path and speed waypoints. | source: [FIVE-VLA, Figure 2](https://arxiv.org/html/2609.18623v1#S3.F2)*

[Open figure at full resolution](/assets/images/october-2609.18623-s3-f2.webp)

The inherited **visual-language adapter** connects image features to the language model. It is distinct from the navigation adapter, action-decoding MLPs, and recurrent memory. The paper does not fully specify the inherited projector's layer dimensions; its token reduction comes primarily from the vision encoder, not a newly introduced query compressor. This matters when comparing against Flex or QT-Former: those baselines change the visual interface itself.

### Remove unnecessary decoding, then add memory where it helps

The efficient driving mode supplies the fixed “Waypoints:” prefix as input instead of generating its four tokens. Trajectory prediction then needs one language-model pass. This preserves the same waypoint output while avoiding four sequential decoding passes. Optional reasoning and VQA modes still generate language and incur its latency; the main efficiency claim uses the mode without chain-of-thought generation.

**Recurrent Action Memory (RAM)** stores the previous step's contextualized action tokens. A one-block self-attention encoder processes that state; a one-block cross-attention decoder lets current learnable action queries read it. The 17M-parameter module operates in latent action space, not on past executed steering commands or recorded ego positions. Its decoder's output projections are zero-initialized, so the newly inserted module initially preserves the original action queries through residual connections.

The training sequence length is only four observations. Continuously accumulating memory indefinitely performs worse than respecting this horizon. At deployment, the preferred accumulation mode maintains a second memory state, warms it while the first is active, and swaps states every half-window. Both states can be batched in one trajectory call, but they still add language-model work. On T4, throughput falls from **5.35 fps without memory to 3.89 fps with accumulation**; reset mode gives 5.15 fps. Timing only RAM's small attention blocks would understate recurrence's actual cost.

### Train independent observations first, then short recurrent streams

Closed-loop training uses SimLingo's approximately **140 hours / two million images**, collected in CARLA by the PDMLite expert at 4 Hz, with driving, reasoning, and VQA supervision. The first stage fine-tunes without memory for six epochs. The second adds RAM for four epochs using four-observation streams and gradient accumulation; the reported RAM configuration uses four accumulation steps. Only the language model, action decoders, and RAM are optimized during memory fine-tuning, and gradients stop between time steps.

The separation addresses a concrete optimization problem: at fixed batch size, longer sequences reduce the number of independent scenes in a batch. A controlled experiment at batch size 32 degrades beyond stream length two. Training without RAM for ten epochs, or giving it a matched extra fine-tuning stage, does not reproduce RAM's gains. The paper reports these schedules but does **not report a complete optimizer, learning-rate, loss-weight, and training-hardware recipe**; those missing values should not be filled in from an assumed SimLingo implementation.

For NVIDIA Physical AI AV, roughly **860 hours train / 350 hours test** provide driving supervision without language labels. No-memory models train for one epoch. RAM models initialize from the corresponding no-memory checkpoint at epoch 0.5 and train two additional epochs. Both one- and four-view settings use streams of length four. Navigation targets come from recorded future motion and receive Student's-t noise during training and evaluation to reduce direct expert-trajectory leakage; this still constitutes privileged route guidance derived from the recorded drive.

### Read the simulated and real-data evaluations separately

| Evaluation | Construction and execution | Main result |
| --- | --- | --- |
| Bench2Drive | 220 challenging CARLA scenarios; predicted path/speed waypoints drive two PID controllers; three evaluation seeds | 90.95 driving score and 77.27% success, versus SimLingo's 85.07 and 67.27% |
| Fail2Drive | 100 in-distribution and 100 generalization scenarios, including unusual hazards | Generalization success 67.0%; harmonic mean of driving score and success 71.9 versus 62.2 for SimLingo |
| NVIDIA real-data replay | Held-out recorded clips; three-second simulated ego rollouts with non-reactive recorded traffic | Single-view speed ADE 0.402 m versus 0.550 m; collision-violation observations 0.78% versus 0.87% |

The NVIDIA evaluator resets the simulated ego and controller at each recorded observation. A kinematic bicycle follows the predicted path with pure pursuit and uses a PID longitudinal controller; surrounding objects follow recorded tracks. An at-fault collision is an intersection involving an obstacle ahead or stopped, with initially overlapping tracks excluded. Violation percentages count **observations whose rollout contains a violation**, not unique crashes; overlapping horizons can measure the same event repeatedly. Missing annotations are excluded from each metric's denominator.

Its adapted PDM score omits map-based drivable-area compliance, modifies collision attribution and comfort checks, and renormalizes available components. It is therefore not the official NAVSIM score. The decrease from 0.87% to 0.78% is about a 10% *relative* reduction, only 0.09 percentage points in absolute terms. Four-view collision violations decrease from 0.82% to 0.75%. These results support better behavior under this replay protocol, not demonstrated reactive road safety.

### The strongest evidence comes from controlled memory ablations

Within FIVE-VLA, RAM raises Bench2Drive success from **73.03% to 77.27%** and driving score from 88.49 to 90.95. Giving way, overtaking, and emergency braking improve most. Explicit past ego-position tokens lower open-loop errors but hurt closed-loop success, illustrating how extrapolating expert history can look strong during replay and fail when the policy generates its own motion.

Memory is not uniformly robust: on a fixed NVIDIA test subset, replacing half the states with incompatible action memories raises trajectory ADE from 0.415 to 1.015 m, worse than bypassing memory at 0.548 m. Finally, the reported 3.89 fps T4 and 29.96 fps A100 measurements are useful hardware-specific evidence; the T4 is an edge-device proxy, not an in-vehicle deployment test.

## High-Level Takeaways

- Visual token count, unnecessary autoregressive decoding, and temporal memory are separate efficiency decisions; FIVE-VLA ablates each.
- Action-token recurrence can retain a pretrained visual-language connector while adding temporal context through two attention blocks.
- The two-stage recipe and zero initialization matter because correlated training streams and newly inserted modules can disturb a competent policy.
- RAM improves controlled closed-loop results, but accumulation has real inference cost and corrupted memory can be harmful.
- Real-data safety numbers describe a custom non-reactive evaluator with explicit attribution rules and denominators.
