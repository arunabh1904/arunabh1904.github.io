---
title: 'InternVL3.5: Multimodal Reasoning and Efficiency'
date: '2025-08-25T00:00:00.000Z'
section: paper-shorts
postSlug: internvl3-5-reasoning-and-efficiency
legacyPath: /paper shorts/2025/08/25/internvl3-5-reasoning-and-efficiency.html
tags: [Vision-Language Models, Reinforcement Learning]
field: 'Vision-Language Models'
summary: '2025 – InternVL3.5: Multimodal Reasoning and Efficiency'
---

## 2025 – InternVL3.5: Multimodal Reasoning and Efficiency

**arXiv:** [2508.18265](https://arxiv.org/abs/2508.18265)

## Summary

> InternVL3.5 treats capability and serving cost as one design problem. Cascade RL first uses offline mixed preference optimization to produce reliable rollouts, then online GSPO refines the model's output distribution. The Flash variants add a visual resolution router that compresses easy image patches more aggressively, while Decoupled Vision-Language Deployment runs the vision and language subsystems on separate servers. The paper reports up to a 16.0% reasoning improvement and a 4.05× throughput speedup over InternVL3, but those numbers come from different ablations: reasoning comes from the Cascade RL comparison, while the 4.05× figure is a 38B deployment result at 896 resolution with DvD and ViR.

## Core Insights

InternVL3.5 is easier to understand as three coupled control loops. The first controls what the model learns, the second controls how many visual tokens an input deserves, and the third controls where computation runs. The architecture remains the familiar ViT–MLP–LLM stack. The contribution is deciding when to spend optimization, visual tokens, and hardware bandwidth.

![InternVL3.5 training recipe](/assets/images/internvl3-5-reasoning-and-efficiency-source-figure-3.webp)
*Source Figure 3. InternVL3.5 uses roughly 250B pre-training tokens, 130B SFT tokens, and a 270K-sample Cascade RL stage; InternVL3.5-Flash adds roughly 30B-token visual consistency and router training. [InternVL3.5](https://arxiv.org/abs/2508.18265)*

Cascade RL addresses a practical problem with multimodal reinforcement learning. Offline RL is efficient because rollouts can be reused, but it can plateau; online RL can improve the policy but is expensive and unstable. InternVL3.5 uses MPO as an offline warm-up, combining preference, quality, and generation losses, then uses GSPO online without a reference-model constraint. The warm-up matters mechanically: it supplies higher-quality rollouts to the online stage, so GSPO starts from a policy that already has useful answers instead of spending its budget discovering them.

The ablation is stronger than the headline. On the 8B model, the SFT checkpoint averages 53.6 over the paper's reasoning set, MPO reaches 56.3, and Cascade RL reaches 60.3. For InternVL3.5-241B-A28B, the corresponding numbers are 60.4, 62.4, and 66.9. The efficiency comparison gives the tradeoff: on the same 8B setup, MPO costs about 0.3K GPU hours for 56.3, GSPO needs about 5.5K hours for 57.3 after one episode and 11.0K for 58.2 after two, while Cascade RL reaches 60.3 at about 5.8K hours. This is not a universal cost law, but it explains why the authors put offline RL before online RL.

The second loop is ViR. In standard InternVL3.5, each image patch becomes 1,024 vision tokens and then 256 language-facing tokens after pixel shuffle. Flash adds a second compression option down to 64 tokens per patch. A patch router estimates how much loss increases under compression and chooses 64 tokens for patches whose visual information survives, or 256 for patches where detail matters. ViCO first trains the model to keep its response distribution consistent across compression rates, then freezes the main model and trains the binary router. This distinction matters: the router is not a general reasoning module; it is a learned budget allocator.

![InternVL3.5 overall capability comparison](/assets/images/internvl3-5-reasoning-and-efficiency-source-figure-1.webp)
*Source Figure 1. InternVL3.5 scores are averaged across general multimodal, reasoning, text, and agentic benchmarks; hatched bars are closed-source models. This figure is a breadth comparison, not a single-task causal ablation. [InternVL3.5](https://arxiv.org/abs/2508.18265)*

The Flash table shows the quality cost of this allocation. For 8B, Flash's overall score is 79.8 versus 80.2 for InternVL3.5; at 38B it is 83.4 versus 83.9, and at the 241B MoE scale it is 84.5 versus 85.0. On high-resolution DocVQA, 8B Flash scores 91.9 versus 92.3, while InfoVQA is 76.0 versus 76.2. The model spends full resolution where the evidence is dense, then compresses elsewhere. The 50% visual-token reduction is therefore paired with a performance comparison, not presented as a free architectural shortcut.

DvD fixes a different bottleneck. Vision encoding is parallel and bursty; language decoding is autoregressive and latency-sensitive. On one shared server, the two streams block each other. DvD places ViT/MLP/ViR on a vision server and the LLM on a language server, sending BF16 visual features over TCP or RDMA and overlapping vision processing with language prefill and decode. Table 18 measures request throughput with 16 requests per second and eight A100 GPUs for the language model. For InternVL3.5-38B at 896 resolution, throughput rises from 2.71 requests/s to 5.06 with DvD and 10.97 with DvD+ViR, a 4.05× increase. At 1344 resolution, DvD alone reaches 1.97×; the exact gain changes with resolution because the vision side becomes the larger blocker.

This coupling also sets the boundary. The paper's overall bars aggregate many benchmarks, and its reasoning gains rely on a particular data and reward stack. ViR's near-parity is measured under the authors' compression training and evaluation settings, while DvD's throughput depends on server topology, request concurrency, and communication. A deployment that changes those conditions should remeasure instead of carrying the 4.05× number forward.

## High-Level Takeaways

- Cascade RL is a sequencing decision: MPO buys reliable rollouts cheaply, then GSPO spends online RL where refinement has a stronger starting point.
- The 8B reasoning average moves 53.6 → 56.3 → 60.3 across SFT, MPO, and Cascade RL; the same table reports 0.3K, 5.5K, and 5.8K GPU-hour regimes for MPO, one-episode GSPO, and Cascade RL comparisons.
- ViR makes resolution conditional on visual content: 256 tokens remain for detail-sensitive patches, while easy patches can use 64, producing nearly matched benchmark scores with half the visual tokens.
- DvD is a serving pipeline, not a model-quality claim. At 896 resolution on 38B, throughput is 2.71 → 5.06 → 10.97 requests/s from baseline to DvD to DvD+ViR.
- InternVL3.5's headline versatility comes from stacking learning, token-budget, and serving decisions. Each layer needs its own ablation and deployment measurements.
