---
title: 'DeepSeek-V4-Flash-0731: Efficient Million-Token Intelligence after Post-Training'
date: '2026-07-31T00:00:00.000Z'
section: paper-shorts
postSlug: deepseek-v4-flash-0731-efficient-million-token-intelligence
legacyPath: /paper shorts/2026/07/31/deepseek-v4-flash-0731-efficient-million-token-intelligence.html
tags:
  - Long Context
  - Mixture of Experts
  - Post-Training
field: 'Language Models'
topics:
  - language-systems
  - learning
summary: '2026 – DeepSeek-V4-Flash-0731: Efficient Million-Token Intelligence after Post-Training'
---

## 2026 – DeepSeek-V4-Flash-0731

**arXiv:** [2606.19348](https://arxiv.org/abs/2606.19348)<br />
**July 31 update:** [DeepSeek-V4-Flash Update](https://api-docs.deepseek.com/updates/)<br />
**Original release:** [DeepSeek V4 Preview](https://api-docs.deepseek.com/news/news260424/)<br />
**0731 weights and evaluation:** [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)<br />
**Fit and serving guide:** [Can DeepSeek V4 Flash 0731 Run on a 64 GB MacBook Pro?](/blog/2026/08/13/running-deepseek-v4-flash-0731-on-a-64-gb-macbook-pro.html)

## Summary

> DeepSeek-V4-Flash-0731 re-post-trains the Flash checkpoint while retaining its pretrained backbone. The architectural story is how compressed attention makes a million-token context more economical; the release story is stronger agent performance plus an attached speculative-decoding module in the public checkpoint. The model card now includes preview comparisons, so the remaining uncertainty is which post-training changes produced those gains, rather than whether a before/after table exists.

## Core Insights

### The technical report and checkpoint release answer different questions

The V4 report explains the 32T-token Flash pretraining run, hybrid attention, and original post-training pipeline. The July changelog identifies a re-post-training update; the public `0731` model card adds preview comparisons and describes the bundled DSpark speculative module. These sources support different levels of attribution. An improved checkpoint score is not an ablation of its data, reward, teacher, or rollout design.

### Compressed attention allocates detail rather than storing every past token equally

The Flash backbone has 43 Transformer layers with a 4,096-wide hidden state. Every layer uses a DeepSeekMoE feed-forward block with one shared expert and 256 routed experts; six routed experts activate per token. The first two layers use sliding-window attention. Later layers interleave Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA), which serve different memory jobs rather than one generic sparse pattern.

CSA produces one compressed KV entry per four-token stride, then selects 512 compressed entries for each query. A 128-token sliding window preserves nearby detail. The compressor uses learned, overlapping contributions from the current and preceding groups, rather than simply averaging four tokens and discarding their order.

HCA compresses much more aggressively, at 128 tokens per entry, and attends across the resulting coarse global memory. Both paths use shared-key-value multi-query attention. At roughly a million tokens, CSA selects from a pool of about a quarter-million compressed entries, while HCA retains around eight thousand coarse entries. These are approximate scale illustrations, not measured memory sizes.

![DeepSeek V4 source Figure 3 showing compressed sparse attention with a learned indexer and local window](/assets/images/deepseek-v4-source-figure-3-compressed-attention.png)
*Fig 1: The indexer selects compressed history on the right, while a separate local window supplies recent detail on the left. Compression reduces stored entries; selection reduces how many enter the main attention computation. | source: [DeepSeek V4 report, Figure 3](https://arxiv.org/abs/2606.19348)*

Follow the two routes into the top attention block. Recent tokens can arrive through the local branch without surviving the long-history selector. Older evidence must be represented in compressed memory and, for CSA, selected as relevant. A large nominal context therefore still leaves two possible failure points: losing a distinction during compression or failing to retrieve its compressed representation.

The indexer itself has a cost. Selecting 512 entries does not make the full attention system constant-cost in context length, because the candidate memory still grows. The reported efficiency comes from the complete mixture of compression, selection, shared KV storage, and precision choices.

Manifold-Constrained Hyper-Connections expand and mix residual streams through doubly stochastic mappings, aiming to add expressive paths while stabilizing signal propagation. Muon optimizes most matrices, while AdamW remains on embeddings, prediction heads, and RMSNorm weights.

| System choice | DeepSeek-V4-Flash | Reported implication |
| --- | --- | --- |
| Sparse capacity | 284B total, 13B active; 6 of 256 routed experts | High parameter capacity at a smaller active footprint than V4-Pro |
| Long-context attention | CSA top-512 after 4× compression + HCA at 128× compression | Selective detail plus a cheap global summary |
| 1M-token efficiency | 10% of V3.2 single-token FLOPs; 7% of its KV cache | Makes million-token decoding materially cheaper in the authors' estimate |
| Pretraining | 32T tokens; 4K → 16K → 64K → 1M context | Trains the target context progressively rather than extrapolating only at inference |
| Quantization | FP4 routed-expert weights; FP4 CSA indexer QK path during QAT | Reduces expert memory traffic and long-context index cost |
| July 31 delta | Same architecture, re-post-trained weights plus a DSpark draft module | Capability change cannot be attributed to a new backbone or more pretraining |

### The backbone is trained for its memory design

The 32T-token corpus extends DeepSeek-V3 data with filtered web pages, mathematics, code, multilingual material, long documents, and agentic mid-training data. Flash trains with dense attention for the first trillion tokens, introduces sparse attention at 64K context, and eventually reaches 1M. The report supplies unusually concrete optimization controls: auxiliary load-balancing loss weight 0.0001, multi-token-prediction loss weight 0.3 for most of training and 0.1 during learning-rate decay, and a 75.5M-token maximum batch. It does not report category mixture proportions or contamination audits for the July agent benchmarks.

### Specialist consolidation is reported, but the July recipe is not isolated

The original post-training recipe trains domain specialists with SFT and GRPO, then consolidates more than ten teacher models through full-vocabulary, multi-teacher on-policy distillation. It also preserves reasoning traces across user turns when tools are active, uses FP4 quantization-aware training, persists interrupted rollouts with token-level write-ahead logs, and runs agent environments in the DSec sandbox platform. These mechanisms explain how the model can learn from long, stateful trajectories. The July update does not say which of them changed.

### The current model card supplies the missing preview comparison

The public model card reports the following Flash-to-Flash comparison:

| Benchmark | Flash Preview | Flash 0731 |
| --- | ---: | ---: |
| Terminal Bench 2.1 | 61.8 | 82.7 |
| NL2Repo | 39.4 | 54.2 |
| CyberGym | 38.7 | 76.7 |
| DeepSWE | 7.3 | 54.4 |
| Toolathlon-Verified | 49.7 | 70.3 |
| Agents' Last Exam | 15.8 | 25.2 |
| AutomationBench Public | 10.8 | 25.1 |

For 0731's public code-agent evaluations, the card specifies DeepSeek Harness minimal mode, maximum effort, temperature 1.0, and top-p 0.95. It does not fully specify the preview's matching evaluation settings or provide component ablations. The comparison supports the released checkpoint's reported improvement; it cannot assign that improvement to one training intervention. Two additional DSBench results use internal tests. [Official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)

### Active parameters and cache savings describe different resources

Thirteen billion active parameters describe per-token expert computation, not the total weights a serving system must hold or access. The model still has 284B total parameters. Similarly, the report's 10% FLOP and 7% KV-cache figures compare one-million-token decoding against V3.2 under the authors' accounting; they are not a promise of tenfold wall-clock speed on any machine.

The speculative module adds another distinction. Drafting multiple tokens can improve serving throughput when enough drafts are accepted and the implementation supports it. Its presence does not explain the pretrained attention architecture or establish the cause of an agent benchmark gain. Weight capacity, active computation, KV memory, and speculative acceptance each influence a different part of deployment.

## High-Level Takeaways

- CSA combines selected compressed history with a local window; HCA provides a coarser global memory.
- Long-context usefulness depends on retaining and retrieving the relevant evidence, not just accepting a million tokens.
- Flash's 32T-token backbone and the July post-training update are separate sources of capability.
- The current model card reports preview comparisons, while training-component attribution remains unavailable.
- Total weights, active parameters, KV-cache size, and speculative decoding describe different serving costs.
