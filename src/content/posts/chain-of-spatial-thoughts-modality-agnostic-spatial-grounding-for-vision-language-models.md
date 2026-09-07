---
title: "Chain of Spatial Thoughts: Modality-Agnostic Spatial Grounding for Vision Language Models"
date: '2026-08-10T00:00:00.000Z'
section: paper-shorts
postSlug: chain-of-spatial-thoughts-modality-agnostic-spatial-grounding-for-vision-language-models
legacyPath: /paper shorts/2026/08/10/chain-of-spatial-thoughts-modality-agnostic-spatial-grounding-for-vision-language-models.html
tags:
  - Vision-Language Models
  - Spatial Reasoning
  - 3D Geometry
field: 'Vision-Language Models'
summary: "2026 – Chain of Spatial Thoughts: Modality-Agnostic Spatial Grounding for Vision Language Models"
---

## 2026 – Chain of Spatial Thoughts: Modality-Agnostic Spatial Grounding for Vision Language Models

**arXiv:** [2608.10278](https://arxiv.org/abs/2608.10278)

## Summary

> Chain of Spatial Thoughts introduces Space Tokens: continuous latent representations for scene-level 3D geometry and object-centric spatial attributes that are generated inside an ordinary VLM reasoning sequence. The model distills geometry from VGGT-Omega, learns to reconstruct cameras, depth, point maps, and 3D boxes, and then uses the tokens during supervised reasoning and GRPO refinement. On VSI-Bench, the Qwen3-VL-8B variant rises from 59.8 to 64.1 (+4.3 points), while the stronger SenseNova-SI-1.3 variant reaches 68.9 (+1.3). The gain is largest on room-size and absolute-distance questions, where global geometry is the missing evidence.

## Core Insights

Space Tokens changes the interface between geometry and language. The method reserves a subset of the vocabulary, generates those positions like ordinary autoregressive tokens, and then uses their final hidden states as continuous spatial representations. Scene tokens align to VGGT-Omega features and reconstruct camera parameters, dense depth, and 3D point maps. Object tokens predict 3D boxes whose center, dimensions, and 6D rotation encode position, size, and orientation. The auxiliary decoders are used for training and verification; they do not have to run at inference time.

![Space Tokens three-stage training pipeline for continuous spatial representations and reasoning](/assets/images/space-tokens-pipeline-paper-figure.png)
*Fig 1: The three-stage pipeline first learns spatial representations, then trains the VLM to use them in next-token reasoning, and finally refines the policy with reinforcement learning. | source: [Chain of Spatial Thoughts, Figure 2](https://arxiv.org/abs/2608.10278)*

The three stages separate “encode geometry” from “use geometry.” Stage 1 injects spatial tokens into a chain-of-thought prompt and backpropagates representation and reconstruction losses. Stage 2 freezes the projection layers and adds teacher-forced next-token prediction so the VLM must incorporate the tokens into its answer. Stage 3 uses GRPO with the VSI-Bench task metric as its reward: fuzzy matching for multiple-choice answers and mean relative accuracy for numeric answers. All three stages fine-tune pretrained VLMs with LoRA; the underlying VLM architecture is unchanged at inference.

The data protocol also explains why a compact latent can help. Training uses one epoch of VICA-322K, samples 32 frames from each video, resizes them to 448 × 448, and uses six selected frames in the reasoning prompt. The paper reports that six informative frames reach 68.6 on VSI-Bench in the prompt-frame ablation, compared with 67.6 for all 32 frames. Selecting evidence before reasoning is part of the method's effective token budget.

![Space Tokens prompt with six selected frames and decoded scene-level and object-level spatial tokens](/assets/images/chain-of-spatial-thoughts-modality-agnostic-spatial-grounding-for-vision-language-models-source-figure-1.webp)
*Fig 2: The supplementary prompt example enlarges the six frames selected for reasoning and shows the scene and object spatial tokens used to answer a room-size question. | source: [Chain of Spatial Thoughts, Figure S1](https://arxiv.org/abs/2608.10278)*

The VSI-Bench table makes the improvement specific. The pretrained Qwen3-VL-8B baseline scores 59.8 overall; the Space Tokens version scores 64.1, with gains of 15.8 points on room size and 5.1 on absolute distance. The SenseNova-SI-1.3 variant moves from 67.6 to 68.9, with a 9.5-point room-size gain. On the OOD evaluation using the stage 1–2 model, BLINK improves from 64.4 to 64.9 and NExT-QA from 74.4 to 80.0, while CV-Bench drops from 88.4 to 86.9. The gains are therefore strongest on geometry-heavy reasoning, not uniformly positive across visual tasks.

![Space Tokens qualitative reconstruction and 3D bounding-box comparison](/assets/images/chain-of-spatial-thoughts-modality-agnostic-spatial-grounding-for-vision-language-models-source-figure-4.webp)
*Fig 3: The qualitative comparison shows decoded 3D scenes and object boxes beside VGGT-Omega references; orange and green boxes distinguish predictions from ground truth. | source: [Chain of Spatial Thoughts, Figure 4](https://arxiv.org/abs/2608.10278)*

The qualitative figure is a verification tool rather than a claim of reconstruction parity. The paper explicitly notes that the lightweight decoders use limited supervision and are not expected to match a specialized 3D model perfectly. Their role is to show that the latent tokens carry decodable geometry while the reasoning model uses them.

| Design choice | What it buys | Boundary |
| --- | --- | --- |
| Reserved spatial tokens | Geometry can remain inside the normal reasoning sequence | Token semantics depend on dedicated supervision. |
| VGGT-Omega distillation | Scene-level 3D priors without a test-time teacher | Teacher quality and projection losses shape the representation. |
| Six-frame reasoning | Less irrelevant visual context than all 32 sampled frames | Frame selection can discard useful evidence. |
| GRPO refinement | Optimizes the VSI-Bench task metric with fuzzy match or mean relative accuracy | Stage-3 reward is benchmark-specific and may not transfer. |

## High-Level Takeaways

- Space Tokens makes geometry a reusable intermediate in the VLM's own reasoning stream, rather than a separate inference-time geometry encoder.
- The strongest evidence is the room-size and absolute-distance improvement; the CV-Bench decrease shows that the method does not improve every visual task.
- Decodable 3D outputs make the representation inspectable, but qualitative agreement with VGGT-Omega is a check on meaning, not a substitute for downstream evaluation.
- The six-frame result exposes a practical selection problem: the model benefits from focused evidence, yet the selector can become a new failure point.
- The fair next comparison is against equally sized learned prompts, extra visual tokens, and a geometry adapter under matched training data, memory, and inference latency.
