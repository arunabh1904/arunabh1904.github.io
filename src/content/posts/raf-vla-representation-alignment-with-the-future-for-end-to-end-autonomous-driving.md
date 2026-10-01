---
title: 'RAF-VLA: Representation Alignment with the Future for End-to-End Autonomous Driving'
date: '2026-09-15T09:00:00.000Z'
section: paper-shorts
postSlug: raf-vla-representation-alignment-with-the-future-for-end-to-end-autonomous-driving
legacyPath: /paper shorts/2026/09/15/raf-vla-representation-alignment-with-the-future-for-end-to-end-autonomous-driving.html
tags:
- Autonomous Driving
- Research
field: 'Autonomous Driving: VLA & Planning'
summary: '2026 – RAF-VLA: Representation Alignment with the Future for End-to-End Autonomous Driving'
---

## 2026 – RAF-VLA: Representation Alignment with the Future for End-to-End Autonomous Driving

**Paper:** [arXiv:2609.17728](https://arxiv.org/abs/2609.17728) · [Full text](https://arxiv.org/html/2609.17728v1)

## Summary

> RAF-VLA uses future images to supervise a driving policy's hidden states, then removes the target encoder and alignment projector before deployment. Its controlled NAVSIM v1 comparison improves SFT PDMS from 87.92 to 89.13 with 3.8% additional training time; the benefit persists after reinforcement fine-tuning. The strongest reported 94.2 PDMS uses best-of-six selection, whereas ordinary inference scores 90.0. These are distinct evaluation settings, and representation alignment is not a deployable future-image generator.

## Core Insights

### Keep the future target in training and the useful hidden state in the policy

World-modeling driving policies often learn to generate future images, depth, or occupancy alongside actions. RAF-VLA asks whether the action policy can receive useful future supervision without learning that entire output pipeline. A frozen Cosmos world encoder extracts a target from future frames. Learnable world queries inside the VLM predict its representation through a training-only MLP. At inference, only the contextualized queries remain.

The input contract consists of a front-camera image sequence, a navigation instruction, recent ego poses, and a one-hot navigation command. The output is a four-second ego trajectory of position and heading waypoints. The source defines image history symbolically rather than providing a fully specified numerical history/resolution contract in its implementation paragraph; those values should not be inferred from the backbone name.

Qwen2.5-VL-7B-Instruct forms the vision-language expert. A second Transformer, with hidden width 1,024, forms the action expert. An MLP turns driving status into a token, and learnable action queries provide slots for a multi-anchor trajectory head. This is a **Mixture-of-Transformers**, not token routing among interchangeable sparse experts.

The attention mask is the main fusion mechanism. The action expert may read vision-language tokens, but the vision-language expert may not read action tokens. World queries can read image and instruction context; queries for the same future horizon interact bidirectionally, while later horizons are masked from earlier ones.

![Source Figure 2: RAF-VLA joint attention and token-group attention mask](/assets/images/october-2609.17728-s1-f2.webp)
*Fig 1: Separate experts retain their own transformations while exchanging context through masked joint attention. The action branch reads the vision-language context without feeding action tokens back into it. | source: [RAF-VLA, Figure 2](https://arxiv.org/html/2609.17728v1#S1.F2)*

[Open figure at full resolution](/assets/images/october-2609.17728-s1-f2.webp)

The two experts have their own query/key/value transformations with a common attention dimension. Their projected keys, queries, and values are concatenated for joint attention, then the outputs return to the appropriate expert. This allows different hidden widths without requiring the complete language backbone to become a small action network.

### The alignment projector is not the image-to-language connector

Four world queries are assigned to each of the 1 s and 4 s horizons. Their hidden states pass through a **two-layer MLP with GELU**, which maps them into the frozen future encoder's feature space. Mean-squared error aligns those projected features with the future-frame targets. The SFT objective adds this term to the action loss with weight one.

This MLP is a distillation/prediction projector. It does not provide the camera-to-language interface, and its outputs are not given to the trajectory head at inference. The action expert reads the learned query states directly. Future frames are training targets, never online policy inputs. The paper specifies the new alignment projector but does not separately redesign or fully enumerate Qwen's inherited visual connector.

### The two-stage recipe has different frozen components

| Stage | Data and objective | Optimization and frozen components |
| --- | --- | --- |
| Future-aligned SFT | NAVSIM navtrain observations, demonstrated trajectories, and future-frame representation targets; action loss + alignment MSE | 4 epochs, AdamW, batch 144, learning rate $10^{-5}$ with cosine decay; VLM vision encoder and Cosmos encoder frozen |
| Reinforcement fine-tuning | Groups of sampled trajectories scored by NAVSIM PDMS; GRPO with an SFT-reference KL penalty | 1.5 epochs, batch 288, learning rate $2\times10^{-6}$, group size 8, KL coefficient 0.04; vision encoder, world queries, and action head frozen |

Both stages use four H200 GPUs, taking approximately 11 and 7 hours respectively. The world encoder and alignment MLP are removed for reinforcement fine-tuning, so the second stage does not keep consuming future-image supervision. PDMS rewards collision avoidance, drivable-area compliance, progress, time-to-collision, and comfort. It is a driving-objective refinement of an already aligned policy.

The alignment ablation also removes the world queries themselves. Consequently, the comparison tests the complete future-alignment mechanism, not MSE alone with every token held fixed. A query-preserving, no-alignment control would isolate that narrower causal question.

### Read the benchmark setting before reading the largest score

NAVSIM is built from OpenScene observations, with navtrain used for training and navtest for evaluation. It emphasizes nontrivial driving cases. Its short-horizon simulation-based trajectory metrics are more informative than matching logged waypoints alone, but they do not establish extended, interactive real-world driving.

| Configuration, NAVSIM v1 | PDMS ↑ |
| --- | ---: |
| Ordinary SFT without alignment | 87.92 |
| Future-aligned SFT | 89.13 |
| Ordinary SFT followed by RFT | 88.85 |
| Future-aligned SFT followed by RFT | 90.03 |
| RAF-VLA with best-of-six selection, main comparison | 94.2 |

The approximately 1.2-point alignment gain survives the training-stage change. The larger best-of-six score includes candidate selection and must not be compared as ordinary single-output inference. NAVSIM v2 reports **86.2 EPDMS for the SFT model**, under an expanded metric that includes direction and traffic-light compliance, lane keeping, and comfort terms. That is neither the same model stage nor the same aggregate as the v1 RFT result.

At batch size one on an H200, trajectory inference changes from 152 to 153 ms with alignment. The **1 ms** figure is added overhead, not total latency. Controlled SFT time rises from 10.56 to 10.96 hours; cross-paper claims about fewer samples seen additionally depend on estimates from other training configurations and do not equal a matched-compute experiment.

The horizon and token ablations favor four queries at each of 1 s and 4 s. More horizons or more queries do not improve PDMS. My interpretation is that selective future targets can regularize the policy without forcing a high-bandwidth generative task. Alternative target encoders, multi-view inputs, and matched-budget projector ablations remain useful tests before generalizing that result.

## High-Level Takeaways

- The new projector maps world-query states to a future-encoder target space; it is removed before deployment, while the queries remain.
- The fusion contract is directional joint attention between a 7B VLM and a smaller action expert, with status and action tokens on the action side.
- Future-aligned SFT and GRPO have different data, objectives, and freeze policies; the future teacher is absent from the second stage.
- The controlled gain is approximately 1.2 PDMS points. The 94.2 headline additionally uses best-of-six selection, and 1 ms denotes overhead on a 153 ms pipeline.
- The main unresolved attribution is which target representation and alignment design matter once query count, backbone, data, and compute are held fixed.
