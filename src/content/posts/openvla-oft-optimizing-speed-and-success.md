---
title: 'Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT)'
date: '2025-02-27T00:00:00.000Z'
section: paper-shorts
postSlug: openvla-oft-optimizing-speed-and-success
legacyPath: /paper shorts/2025/02/27/openvla-oft-optimizing-speed-and-success.html
tags:
  - Robotics
  - Fine-Tuning
field: 'Vision-Language-Action & Robotics'
summary: "2025 – Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT)"
---

**arXiv:** [2502.19645](https://arxiv.org/abs/2502.19645)

**Project:** [openvla-oft.github.io](https://openvla-oft.github.io/)

## Summary

> OpenVLA-OFT shows that the fine-tuning interface can matter more than preserving a VLA's pretraining objective. It replaces autoregressive discrete action-token decoding with parallel continuous action chunks and trains them with a simple L1 regression loss.

## Core Insights

### Fine-tuning can replace the action interface

A pretrained policy need not keep the interface it learned with. The first figure separates two decisions often bundled together: whether actions are decoded one at a time, and whether those outputs are discrete tokens or continuous values. Parallel continuous prediction reuses the visual-language representation while changing both decisions.

![OpenVLA-OFT comparison of autoregressive versus parallel decoding and discrete versus continuous action prediction](/assets/images/openvla-oft-optimizing-speed-and-success-paper-figure.png)
*Fig 1: Isolates the fine-tuning choices: OpenVLA-OFT replaces sequential discrete token generation with parallel action decoding and continuous regression or diffusion objectives. | source: [OpenVLA-OFT, Figure 2](https://arxiv.org/abs/2502.19645)*

![Figure 1 from Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT)](/assets/images/openvla-oft-optimizing-speed-and-success-source-figure-1.webp)
*Fig 2: OpenVLA-OFT+ on the bimanual ALOHA robot. The optimized fine-tuning recipe adds parallel decoding, action chunking, continuous actions, and FiLM language conditioning for the real-robot setting. | source: [OpenVLA-OFT, Figure 1](https://arxiv.org/abs/2502.19645)*

![Figure 3 from Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT)](/assets/images/openvla-oft-optimizing-speed-and-success-source-figure-3.webp)
*Fig 3: LIBERO simulation benchmark task suites used to compare fine-tuning design choices. | source: [OpenVLA-OFT, Figure 3](https://arxiv.org/abs/2502.19645)*


### Parallel chunks expose the throughput tradeoff

The paper ablates three coupled decisions: serial versus parallel decoding, discrete versus continuous actions, and next-token versus regression/diffusion objectives. Its headline LIBERO comparison moves from the reported fine-tuned OpenVLA score of 76.5% to 97.1% for the full OFT recipe, but the table places those rows in different input and training-data settings; the gain is therefore a recipe result rather than a single isolated head ablation. On the same A100 measurement, parallel decoding plus an 8-step action chunk raises throughput from 4.2 to 109.7 Hz for continuous L1 actions, a 26× increase. The 71.4-Hz, 0.112-s row is the larger-input variant, so these numbers should not be conflated.

The surprising result is that a simple L1 head can match diffusion fine-tuning in the studied setting. Pretrained semantics remain useful even when the action interface and training loss change completely. OFT+ adds FiLM to the vision backbone for language grounding; in the ALOHA language-dependent tasks, removing FiLM drops following to 33%, chance level.

| Adaptation choice | OFT selection | Reason |
| --- | --- | --- |
| Decoding | Parallel action chunks | Removes sequential token latency. |
| Representation | Continuous actions | Avoids quantization error. |
| Objective | L1 regression | Fast convergence and inference in the tested tasks. |

On ALOHA, all methods use action chunk size $K=25$ except Diffusion Policy ($K=24$); the OpenVLA-OFT+ query processes three 224×224 images and a 14-dimensional robot state. Its 77.9-Hz throughput and 0.321-s latency approach the smaller RDT-1B at 84.1 Hz while retaining a 7.5B OpenVLA backbone. The comparison is a systems result: chunking amortizes the VLM pass, while the L1 head avoids iterative denoising.

## High-Level Takeaways

- OpenVLA-OFT informs which parts of a pretrained VLA should be treated as reusable semantics and which should be replaced for deployment. Its unit is an observation paired with a continuous action chunk. The VLM backbone is shared; the action head abandons the language-token interface.
- The results establish a strong speed–success recipe on LIBERO and ALOHA, not universal superiority of L1 regression. The paper's own limitation is multimodality: L1 can average valid actions when several strategies fit the same observation, while diffusion can represent alternatives at a higher inference cost.
- OpenVLA-OFT is the practical SFT baseline that later RL post-training papers improve.
- Near-saturated LIBERO success leaves little room to measure robustness and recovery; the ALOHA rollouts show why language grounding and visual feedback must be measured separately from task completion.
- Reuse the pretrained representation, not necessarily its language decoder. The useful recipe is conditional: parallel chunks and L1 help when a dominant action mode is enough, while FiLM becomes critical when the instruction changes the target object.
