---
title: 'DeepSeek-V4-Flash-0731: Efficient Million-Token Intelligence after Post-Training'
date: '2026-04-26T00:00:00.000Z'
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
**0731 weights:** [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)<br />
**Fit and serving guide:** [Can DeepSeek V4 Flash 0731 Run on a 64 GB MacBook Pro?](/blog/2026/08/13/running-deepseek-v4-flash-0731-on-a-64-gb-macbook-pro.html)

## Summary

> DeepSeek-V4-Flash-0731 is a post-training release, not a new base model. It keeps the April preview's architecture and size—284 billion total parameters, 13 billion active per token, and a one-million-token context window—while replacing the API checkpoint with a model tuned for stronger agent behavior and native Responses API use. DeepSeek initially said the July 31 update changed only the Flash API, leaving the web/app models and V4-Pro endpoint unchanged; the exact `0731` weights were published on Hugging Face on August 1.

## Core Insights

That boundary matters because two sources support different claims. The V4 technical report explains the pretrained architecture, 32-trillion-token training run, long-context efficiency, and original specialist-to-distillation pipeline. The July 31 changelog reports new agent scores and says the checkpoint was “only re-post-trained,” but does not disclose the new data, reward design, rollout budget, teacher models, ablations, or before/after scores under the same harness.

![DeepSeek V4 architecture with hybrid compressed attention, DeepSeekMoE, manifold-constrained hyper-connections, and multi-token prediction](/assets/images/deepseek-v4-architecture-paper-figure.png)
*DeepSeek V4 retains a Transformer/MoE backbone but replaces ordinary attention with interleaved Compressed Sparse Attention and Heavily Compressed Attention, and replaces a single residual stream with manifold-constrained mixing paths. org/abs/2606.19348). source: [DeepSeek V4 report](https://arxiv.org/abs/2606.19348)*

![Figure 2 from DeepSeek-V4-Flash-0731: Efficient Million-Token Intelligence after Post-Training](/assets/images/deepseek-v4-flash-0731-efficient-million-token-intelligence-source-figure-2.webp)
*Figure 2 Overall architecture of DeepSeek-V4 series. We use hybrid CSA (Compressed Sparse Attention) and HCA (Heavily Compressed Attention) for attention layers, DeepSeekMoE for feed-forward layers, and strengthen conventional residual connections with m HC. source: [DeepSeek-V4-Flash-0731: Efficient Million-Token Intelligence after Post-Training](https://arxiv.org/abs/2606.19348)*

![Figure 1 from DeepSeek-V4-Flash-0731: Efficient Million-Token Intelligence after Post-Training](/assets/images/deepseek-v4-flash-0731-efficient-million-token-intelligence-source-figure-1.webp)
*Figure 1 Left : benchmark performance of DeepSeek-V4-Pro-Max and its counterparts. Right : inference FLOPs and KV cache size of DeepSeek-V4 series and DeepSeek-V3.2. source: [DeepSeek-V4-Flash-0731: Efficient Million-Token Intelligence after Post-Training](https://arxiv.org/abs/2606.19348)*


The Flash backbone has 43 Transformer layers with a 4,096-wide hidden state. Every layer uses a DeepSeekMoE feed-forward block with one shared expert and 256 routed experts; six routed experts activate per token. The first two layers use sliding-window attention. Later layers interleave Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA), which serve different memory jobs rather than one generic sparse pattern.

CSA first compresses each four-token KV group, then uses a learned indexer to select the top 512 compressed entries for each query; a 128-token sliding window restores local detail. HCA compresses much more aggressively—128 tokens into one entry—and supplies a cheaper global summary. Both paths use shared-key-value multi-query attention. Manifold-Constrained Hyper-Connections expand and mix residual streams through doubly stochastic mappings, aiming to gain depth-wise expressivity without unstable signal growth. Muon optimizes most matrices, while AdamW remains on embeddings, prediction heads, and RMSNorm weights.

| System choice | DeepSeek-V4-Flash | Reported implication |
| --- | --- | --- |
| Sparse capacity | 284B total, 13B active; 6 of 256 routed experts | High parameter capacity at a smaller active footprint than V4-Pro |
| Long-context attention | CSA top-512 after 4× compression + HCA at 128× compression | Selective detail plus a cheap global summary |
| 1M-token efficiency | 10% of V3.2 single-token FLOPs; 7% of its KV cache | Makes million-token decoding materially cheaper in the authors' estimate |
| Pretraining | 32T tokens; 4K → 16K → 64K → 1M context | Trains the target context progressively rather than extrapolating only at inference |
| Quantization | FP4 routed-expert weights; FP4 CSA indexer QK path during QAT | Reduces expert memory traffic and long-context index cost |
| July 31 delta | Same architecture, re-post-trained weights plus a DSpark draft module | Capability change cannot be attributed to a new backbone or more pretraining |

The 32T-token corpus extends DeepSeek-V3 data with filtered web pages, mathematics, code, multilingual material, long documents, and agentic mid-training data. Flash trains with dense attention for the first trillion tokens, introduces sparse attention at 64K context, and eventually reaches 1M. The report supplies unusually concrete optimization controls: auxiliary load-balancing loss weight 0.0001, multi-token-prediction loss weight 0.3 for most of training and 0.1 during learning-rate decay, and a 75.5M-token maximum batch. It does not report category mixture proportions or contamination audits for the July agent benchmarks.

The original post-training recipe trains domain specialists with SFT and GRPO, then consolidates more than ten teacher models through full-vocabulary, multi-teacher on-policy distillation. It also preserves reasoning traces across user turns when tools are active, uses FP4 quantization-aware training, persists interrupted rollouts with token-level write-ahead logs, and runs agent environments in the DSec sandbox platform. These mechanisms explain how the model can learn from long, stateful trajectories. The July update does not say which of them changed.

The official July evaluation reports 82.7 on Terminal-Bench 2.1, 54.2 on NL2Repo, 76.7 on CyberGym, 54.4 on DeepSWE, 70.3 on Toolathlon Verified, 25.2 on Agents' Last Exam, and 25.1 on the public AutomationBench. Code-agent tests use an unreleased “DeepSeek Harness minimal mode,” maximum effort, temperature 1.0, and top-p 0.95; two additional DSBench results are internal. The numbers establish the API checkpoint's measured level, not the causal gain from re-post-training: the changelog supplies no same-version preview baseline, and benchmark or harness versions differ from several April report tables.

## High-Level Takeaways

- DeepSeek-V4-Flash informs whether long context should be made affordable by compressing the memory hierarchy before spending more active parameters. Its atomic unit remains the token, but CSA groups tokens, selects a query-dependent subset of compressed entries, and restores a local window; HCA keeps a much coarser global trace. That design makes retrieval quality—not nominal context length—the central risk. A one-million-token window is useful only if compression and selection retain the evidence an agent needs late in a trajectory.
- The expensive commitment is to a custom serving stack for hybrid attention, sparse experts, mHC residual paths, FP4 computation, and persistent tool state. The paper reports striking modeled efficiency against V3.2, but does not isolate CSA, HCA, mHC, Muon, data, and scale under a single matched budget. The decisive experiment is a long-context retrieval-and-agent sweep that holds active parameters, training tokens, decoding budget, and hardware constant while varying compression rate and selector top-k. Reject the architecture choice if a simpler MLA/GQA baseline matches end-task accuracy and latency, or if retrieval degrades sharply outside synthetic long-context tests.
- For the 0731 checkpoint, the missing control is even simpler: evaluate the preview and updated API models with the same public harness, effort budget, samples, and judge. Without that table, the size of the post-training gain is not reported. At ten times the context or trajectory length, selector recall, accumulated compressed-state error, sandbox persistence, and cache bandwidth are likely to dominate before nominal MoE capacity does.
- DeepSeek-V4-Flash is the smaller V4 branch: it shares the V4 hybrid-attention and post-training stack, but activates 13B parameters rather than V4-Pro's 49B.
- The July checkpoint's weights are public, but its post-training recipe is not reported, its code-agent harness is not yet released, two headline benchmarks are internal, and the release provides no controlled ablation of the post-training changes.
- DeepSeek-V4-Flash-0731 is evidence that post-training can move agent performance without changing a 32T-token base; until the harness and matched before/after results are released, use it as a deployment update rather than proof of a new training method.
