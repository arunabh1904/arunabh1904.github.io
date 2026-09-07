---
title: 'Kimi K3: Open Frontier Intelligence'
date: '2026-07-27T16:49:54.000Z'
section: paper-shorts
postSlug: kimi-k3-open-frontier-intelligence
legacyPath: /paper shorts/2026/07/27/kimi-k3-open-frontier-intelligence.html
tags:
  - Multimodal Models
  - Mixture of Experts
  - Agentic AI
field: 'Omni-Model Architectures'
topics:
  - multimodal
  - language-systems
  - learning
summary: '2026 – Kimi K3: Open Frontier Intelligence'
---

## 2026 – Kimi K3: Open Frontier Intelligence

**arXiv:** [2607.24653](https://arxiv.org/abs/2607.24653)<br />
**Technical blog:** [Kimi K3](https://www.kimi.com/blog/kimi-k3)<br />
**Code and report:** [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)<br />
**Weights:** [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)

## Summary

> The primary report describes Kimi K3 as a full-weight release: a native multimodal 2.78T-parameter MoE model with 104.2B activated parameters and a context window of up to one million tokens. Its bet is architectural and operational at the same time—hybrid recurrent/global attention, depth-wise attention residuals, extreme expert sparsity, long-context training, and multi-stage agentic post-training are designed as one system. The model is competitive with frontier baselines while the report explicitly says it still trails Claude Fable 5 and GPT-5.6 Sol overall.

## Core Insights

### Three selective paths carry information at different scales

Kimi K3 uses three KDA layers followed by one gated MLA layer in each block, plus a final MLA layer. KDA maintains a fixed-size recurrent state with channel-wise decay and delta-rule writes; gated MLA periodically restores unrestricted global content retrieval while using a compressed key-value representation. The resulting 69 KDA and 24 MLA layers make most sequence mixing linear-time without asking recurrence to solve every retrieval problem.

Attention Residuals apply the same selective-access idea over depth. Learned pseudo-queries attend over the embedding and earlier block outputs instead of forcing every layer into one accumulated residual. K3's Block AttnRes groups its 93 layers into eight 12-layer blocks plus a partial final block, reducing the saved depth state while retaining access to earlier representations. Stable LatentMoE then handles width: two full-width shared experts process common transformations, while each token activates 16 of 896 routed experts in a half-width latent space.

![Kimi K3 architecture across token, depth, and expert mixing](/assets/images/kimi-k3-architecture-paper-figure.png)
*Fig 1: The architecture combines KDA with periodic gated MLA, Block Attention Residuals across earlier blocks, Stable LatentMoE with shared and routed experts, and a native MoonViT-V2 vision path. | source: [Kimi K3, Figure 2](https://arxiv.org/abs/2607.24653)*

The stabilizers are part of the scaling story. RMS normalization before the routed up-projection limits variation in the aggregate latent expert output; SiTU-GLU caps both multiplicative branches while retaining SwiGLU's near-origin response; and Quantile Balancing sets expert biases from router-score quantiles without changing the mixture weights or router gradients. Those choices address activation outliers and expert-load imbalance that become visible only when 896 experts are in play.

### Native vision and context extension change the training problem

Kimi K3 trains the roughly 0.4B-parameter, 27-layer MoonViT-V2 from scratch with next-token prediction. The report compares it with a SigLIP-initialized MoonViT-3D: the from-scratch tower has lower, less spiky gradient norms and matches the initialized baseline across vision evaluations. Images and videos share the vision pathway; spatial attention is factored from temporal attention, temporal pooling compresses frames, and a 2×2 pixel shuffle reduces visual tokens before projection into the shared backbone.

Table 1 gives the scale precisely: 2.78T total parameters, 104.2B activated, 93 layers, 896 routed experts, 16 active per token, two shared experts, and 401M total ViT parameters. The model has no explicit positional embedding; KDA's recurrent decay carries positional information. Context grows from 8K to 64K during pretraining and from 256K to 1M in cooldown. Long-context data is cleaned, upsampled, and synthesized so tasks require information spread across the full context rather than only local patterns.

![Reinforcement-learning capability and assistant-step curves](/assets/images/kimi-k3-open-frontier-intelligence-source-figure-8.webp)
*Fig 2: Across coding, tool use, web development, search, workflows, and visual tasks, increasing RL FLOPs raises scores alongside the average number of assistant steps. | source: [Kimi K3, Figure 8](https://arxiv.org/abs/2607.24653)*

The report fits an approximately 2.5× improvement in overall scaling efficiency over Kimi K2, but the comparison changes model size, architecture, data, optimizer, and training recipe together. It is evidence for the K3 system recipe, not an isolated causal estimate for KDA or the expert router.

### RL turns long context into a usable capability

Post-training proceeds from supervised agent trajectories to specialist RL policies, then Multi-Teacher On-Policy Distillation (MOPD) consolidates them. The RL specialists span general tasks, general agents, and coding agents, each at low, high, and max reasoning effort, for nine expert models. Partial rollouts can pause and resume across iterations while sandbox and generation state persist, avoiding a synchronization barrier at the longest trajectories. Quantization-aware training runs through SFT and RL with MXFP4 expert weights and MXFP8 activations, so rollout and training use the deployment precision.

The reported table puts the model's strengths and limits in the same frame: GPQA Diamond 93.5, ProgramBench 77.8, Terminal-Bench 2.1 88.3, FrontierSWE 81.2, and BrowseComp 91.2. Kimi K3 is 0.5 points below GPT-5.6 Sol on Terminal-Bench and 5.4 points below Claude Fable 5 on FrontierSWE, while HLE-Full is 43.5 without tools and 56.0 with tools and CritPt is 23.4. These comparisons use max reasoning effort, varying harnesses, third-party leaderboard snapshots, fallback behavior, and internal tests, so the small gaps are not clean architecture comparisons.

## High-Level Takeaways

- Kimi K3 coordinates sparsity across sequence length, depth, and width instead of treating MoE as the only scaling lever.
- The full primary report confirms a release of model weights; the relevant question is whether the serving and training stack can support the resulting 104.2B active workload.
- Native vision, progressive context extension, persistent rollouts, and deployment-aware quantization connect the model design to long-horizon agent use.
- The 2.5× K2 scaling claim and benchmark leads are system-level results; matched-compute ablations and independently reproduced long-context evaluations are still needed.
