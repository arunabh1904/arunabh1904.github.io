---
title: 'TurboVLA: Real-Time Vision-Language-Action Model at 32 Hz on an RTX 4090 with <1 GB VRAM'
date: '2026-07-29T09:00:00.000Z'
section: paper-shorts
postSlug: turbovla-real-time-vision-language-action-model-at-32-hz-on-an-rtx-4090-with-1-gb-vram
legacyPath: /paper shorts/2026/07/29/turbovla-real-time-vision-language-action-model-at-32-hz-on-an-rtx-4090-with-1-gb-vram.html
tags:
  - Vision-Language-Action
  - Efficient Inference
  - Robot Learning
field: 'Vision-Language-Action & Robotics'
topics:
  - embodied
  - multimodal
  - learning
summary: '2026 – TurboVLA: Real-Time Vision-Language-Action Model at 32 Hz on an RTX 4090 with <1 GB VRAM'
---

## 2026 – TurboVLA: Real-Time Vision-Language-Action Model at 32 Hz on an RTX 4090 with <1 GB VRAM

**arXiv:** [2607.27205](https://arxiv.org/abs/2607.27205)

**Project:** [TurboVLA](https://H-EmbodVis.github.io/TurboVLA) · **Code:** [H-EmbodVis/TurboVLA](https://github.com/H-EmbodVis/TurboVLA)

Most vision-language-action models route visual tokens through a large language model before producing robot actions. TurboVLA removes that execution path. DINOv3 encodes the cameras, BERT encodes the instruction, six lightweight bidirectional cross-attention blocks exchange information between the streams, and an ACT-style decoder predicts a continuous action chunk in parallel.

On LIBERO, the 0.2B-parameter configuration reaches 97.7% average success with 31.2 ms latency and 0.9 GB peak inference memory on an RTX 4090. It also reaches 60.2% across 50 RoboTwin 2.0 bimanual tasks and outperforms a matched $\pi_{0.5}$ baseline on four real-robot tasks. The result supports a sharp systems claim: concrete execution-level language conditioning does not always require a generative LLM in the control loop. It does not show that the same compact path can perform open-ended task decomposition or high-level planning.

## Paper Insights

TurboVLA preserves token-level instruction features rather than compressing language to a task ID. Each camera produces spatial visual tokens with positional and view embeddings. The interaction stack alternates visual-to-instruction and instruction-to-visual cross-attention, so scene evidence can refine the instruction representation while language selects relevant visual features. Robot state bypasses this fusion stack and enters only at the action decoder.

![TurboVLA architecture with compact vision and language encoders, bidirectional feature interaction, and parallel action-chunk decoding](/assets/images/turbovla-architecture.png)
_TurboVLA replaces the LLM-centered $V\to L\to A$ path with compact modality encoders, bidirectional cross-attention, and a continuous action-chunk decoder. Cropped from Figure 3 of the [paper](https://arxiv.org/abs/2607.27205)._

The policy is trained by behavior cloning with an $\ell_1$ loss on expert action chunks; it uses no language-modeling loss and no robot-data pretraining beyond each target benchmark. LIBERO training mixes all four suites for 80,000 steps and evaluates 50 rollouts for each of 40 tasks. RoboTwin uses one multitask model trained on the official clean demonstrations for 50 bimanual tasks. The real-world model starts from LIBERO, fine-tunes on 65 demonstrations per task, and runs 40 trials on each of four AgileX Piper tasks.

| System | Parameters | RTX 4090 latency | Inference VRAM | Main success result |
| --- | ---: | ---: | ---: | ---: |
| $\pi_{0.5}$ on LIBERO | 3.4B | 93.6 ms | 12.8 GB | 96.9% |
| VLA-Adapter on LIBERO | 1.5B | 87.3 ms | 4.3 GB | 97.3% |
| TurboVLA on LIBERO | **0.2B** | **31.2 ms** | **0.9 GB** | **97.7%** |
| $\pi_{0.5}$ on RoboTwin 2.0 | 3.4B | 95.6 ms | not reported | 57.0% |
| TurboVLA on RoboTwin 2.0 | **0.4B** | **43.4 ms** | not reported | **60.2%** |

The ablations show that efficiency does not come from ignoring language. Removing it collapses LIBERO-Goal from 97.4% to 11.6% and lowers the overall average to 70.8%. A learned task ID recovers 95.4%, while semantic instructions reach 97.7%. Directly concatenating vision and language yields 95.2%; one-way cross-attention reaches 96.1% or 96.5%; bidirectional interaction reaches 97.7%. Six fusion layers outperform two and four, while eight slip to 96.6%. The action horizon is similarly non-monotonic: 12 steps performs best among 8, 10, 12, and 15.

## Decision Lens

TurboVLA informs whether an execution policy should inherit a large language backbone or use a smaller semantic encoder plus direct feature fusion. Its training unit is a continuous action chunk, not an action token. Visual and language parameters remain separate until a compact bidirectional module, and the only optimized objective is action imitation. For closed-set manipulation instructions with benchmark-scale linguistic variation, the reported evidence favors the compact path.

The expensive decision is where to place general reasoning. Removing the LLM cuts latency and memory, but also removes a natural place for broad semantic knowledge, compositional planning, and generative intermediate reasoning. The comparisons are not matched for pretraining data, architecture, or optimization, so the paper establishes a performance-efficiency frontier on these tasks rather than isolating the LLM's causal cost. At 10× instruction and environment diversity, linguistic coverage and long-horizon task planning are likelier to fail before the action decoder does.

A decisive test would compare a compact executor, an LLM-centric policy, and a hierarchical LLM-planner-plus-compact-executor under the same robot demonstrations, visual backbone, and wall-clock budget. Instructions should include novel compositions, ambiguous references, recovery steps, and long-horizon tasks. The compact-path claim should be rejected if its latency advantage disappears once it receives the planning machinery needed to match success on those harder instructions.

**Context:** TurboVLA separates language-conditioned execution from general-purpose language generation, arguing that the former can use grounding-style cross-attention and continuous action decoding.

**Limits:** Most evidence is simulation; RoboTwin uses only the clean setting; real-world evaluation covers four tasks and 160 total trials; and the comparison mixes methods with different embodied pretraining and data. The authors explicitly scope the model to execution-level instructions rather than high-level planning.

**Takeaway:** For concrete manipulation commands, a small bidirectional vision-language interface can match large VLA policies while running at 32 Hz—but planning remains outside the claim.
