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

> Kimi K3 scales an open-weight, natively multimodal model along two axes at once: a 2.78-trillion-parameter sparse backbone supplies more pretrained capacity, while long-horizon reinforcement learning teaches the model to spend up to a million tokens of context on coding, research, and tool use. Only 104.2 billion parameters are active per token, but that is still a large serving workload; the report recommends supernodes with at least 64 accelerators.

## Core Insights

The paper's strongest contribution is therefore architectural and systemic rather than a single benchmark lead. Kimi Delta Attention, periodic global attention, depth-wise Attention Residuals, and a much wider expert pool are co-designed with training kernels, expert routing, persistent sandboxes, quantization-aware post-training, and prefix caching. The resulting system approaches the strongest proprietary models on several agentic tasks, but does not erase the cost of serving a 104B-active model or the comparability problems in harness-dependent evaluation.

![Kimi K3 architecture combining Kimi Delta Attention, gated MLA, Stable LatentMoE, Attention Residuals, and a MoonViT-V2 vision pathway](/assets/images/kimi-k3-architecture-paper-figure.png)
*Fig 1: Kimi K3 scales information flow along three axes: KDA and gated MLA mix tokens, Stable LatentMoE mixes channels through sparse experts, and Attention Residuals select earlier block representations across depth. MoonViT-V2 supplies the native vision path. | source: [Kimi K3 report](https://arxiv.org/abs/2607.24653)*

![Figure 8 from Kimi K3: Open Frontier Intelligence](/assets/images/kimi-k3-open-frontier-intelligence-source-figure-8.webp)
*Fig 2: Scores and the average assistant steps across a variety of public and in-house evaluations during RL. By scaling RL FLOPs, tool-call steps scale up consistently, accompanied by a comprehensive improvement in the model’s overall capability. | source: [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)*



Each backbone group contains three Kimi Delta Attention layers followed by one gated Multi-head Latent Attention layer; an extra MLA layer at the end guarantees a final global interaction. KDA maintains a fixed-size recurrent state and applies channel-wise decay plus a delta-rule write, making long sequences cheaper than full softmax attention. The periodic MLA layers preserve unrestricted content-based retrieval while compressing their KV state. This 69-KDA/24-MLA split makes most sequence mixing linear-time without asking a recurrent state to remember everything.

Attention Residuals changes the depth path. Instead of adding every previous layer into one residual stream with fixed weight, learned pseudo-queries attend over earlier block outputs. The deployed Block AttnRes variant groups the 93 layers into eight 12-layer blocks plus a partial final block, reducing retained depth state and cross-stage communication. Stable LatentMoE changes the width path: two full-width shared experts handle common transformations, while each token activates 16 of 896 routed experts in a half-width latent space. RMS normalization, bounded SiTU-GLU activations, and Quantile Balancing are training-stability mechanisms for this unusually sparse routing regime, not cosmetic feed-forward variants.

| Design axis | Kimi K3 choice | Consequence |
| --- | --- | --- |
| Capacity | 2.78T total, 104.2B active | Large expert pool without dense 2.78T compute per token |
| Sequence | 69 KDA + 24 gated MLA layers | Cheap recurrent mixing with periodic global retrieval |
| Depth | Block Attention Residuals | Selective access to earlier representations rather than uniform accumulation |
| Vision | 401M-parameter MoonViT-V2 + projector | Images and video enter the shared next-token objective from the start |
| Context | 8K → 64K in pretraining, 256K → 1M in cooldown | Concentrates expensive long-sequence training late in the schedule |
| Deployment | MXFP4 expert weights, MXFP8 expert activations | Reduces the dominant MoE memory traffic while training against quantization error |

Kimi K3 is trained natively multimodally: visual and text tokens are interleaved under one next-token objective rather than attaching a vision encoder to a finished language model. Images and videos share MoonViT-V2 parameters; temporal pooling and a 2×2 pixel shuffle reduce visual token count. The report does not disclose total pretraining tokens, modality proportions, or a reproducible data-mixture schedule, so the claimed 2.5× scaling-efficiency gain over Kimi K2 cannot be assigned cleanly to architecture, data, or optimizer changes.

Post-training begins with supervised agent trajectories, then trains nine specialist policies across three domains—general tasks, general agents, and coding agents—and three reasoning-effort levels. Multi-Teacher On-Policy Distillation consolidates those policies into one checkpoint. Long rollouts can pause across RL iterations while their sandbox and generation state persist, avoiding the utilization collapse that would follow from waiting for every million-token trajectory to finish. Quantization-aware training starts at SFT and continues through RL, so rollout behavior uses the same expert-weight precision intended for deployment.

The main evaluation uses maximum reasoning effort and a mixture of Kimi Code, Claude Code, and Codex harnesses. Kimi K3 scores 93.5 on GPQA Diamond, 77.8 on ProgramBench, 88.3 on Terminal-Bench 2.1, 81.2 on FrontierSWE, and 91.2 on BrowseComp. These are competitive frontier results: Terminal-Bench is within 0.5 points of GPT-5.6 Sol, while FrontierSWE trails Claude Fable 5 by 5.4 points. The paper also reports weaker research-level reasoning—43.5 on HLE without tools and 23.4 on CritPt—and explicitly states that Kimi K3 trails Claude Fable 5 and GPT-5.6 Sol overall. Several comparisons use different harnesses, fallback behavior, internal sets, or leaderboard snapshots, so small gaps should not be read as architecture-isolated wins.

## High-Level Takeaways

- Kimi K3 informs whether to buy long-context and agentic capability through one enormous sparse model or through a smaller model plus external orchestration. Its atomic units are text and visual tokens, but three distinct sparsity mechanisms allocate work: KDA compresses sequence history into recurrent state, AttnRes selects representations across depth, and LatentMoE activates a small expert subset across width. That coordination is the paper's real bet. A simpler backbone would be easier to serve, but would give up the three independent routes through which K3 scales capacity.
- The expensive decision is not the 2.78T headline alone; it is committing training and inference infrastructure to hybrid recurrent/global attention, 896-way routing, custom quantization, and persistent agent state. The report fits a 2.5× scaling-efficiency improvement over Kimi K2, yet changes architecture, optimizer, model size, expert count, multimodal training, data, and systems together. A matched-compute factorial study—K2 versus K3 attention, residual, MoE, and data choices under the same token budget—would show which complexity pays for itself. The claim should be rejected for a component if removing it preserves validation loss, long-context retrieval, and agent performance at lower serving cost.
- At ten times the workload, expert communication and rollout state are more likely to fail first than raw arithmetic. KDA keeps recurrent state fixed, but the global MLA layers, 104B active parameters, expert dispatch, and persistent million-token sandboxes still demand a large communication and storage domain. The model is most compelling where long trajectories and native vision justify that fixed infrastructure; it is a poor default for latency-sensitive or modest-volume deployment.
- Kimi K3 extends the Kimi Linear/KDA, Kimi K2 MoE, Kimi K2.5 agent-training, and Attention Residuals lines into one 3T-class native multimodal system.
- Pretraining token count and modality mixture are not reported; component gains are not isolated at matched compute; several agent comparisons depend on different harnesses or internal evaluations; and practical deployment assumes a 64-accelerator-class supernode.
- Kimi K3's falsifiable claim is that coordinated sparsity across sequence, depth, and experts can make a 3T-class multimodal agent worth its systems burden; matched-budget ablations and independent long-horizon evaluations now need to show which parts survive outside Moonshot's stack.
