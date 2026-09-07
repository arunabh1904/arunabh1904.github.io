---
title: 'Wayformer: Motion Forecasting via Simple and Efficient Attention Networks'
date: '2022-07-12T00:00:00.000Z'
section: paper-shorts
postSlug: wayformer-motion-forecasting-via-simple-and-efficient-attention-networks
legacyPath: /paper shorts/2022/07/12/wayformer-motion-forecasting-via-simple-and-efficient-attention-networks.html
tags:
  - Other
field: 'Motion Forecasting & Planning'
summary: "2022 – Wayformer: Motion Forecasting via Simple and Efficient Attention Networks"
---

**arXiv:** [2207.05844](https://arxiv.org/abs/2207.05844)

## Summary

Wayformer studies whether a motion forecaster needs a different encoder for every input modality. It projects traffic lights, road polylines, agent histories, and nearby-agent interactions into a common token width, then compares early, late, and hierarchical fusion with several attention approximations. The paper's claim is architectural simplicity: early fusion, where modalities meet in one cross-modal encoder, is competitive with more specialized designs.

The useful contribution is the controlled design space. Fusion determines when modalities can exchange information; factorized attention changes the cost of spatial-temporal interaction; latent queries reduce the number of tokens that later layers process. These are separable knobs, so quality and latency can be measured together rather than hidden behind a bespoke architecture.

## Core Insights

### Early fusion makes modality interaction the default

Wayformer represents each modality as a tensor with spatial and temporal axes, applies a learned projection and positional embedding, and concatenates the resulting tokens. Late fusion gives each modality its own encoder and delays interaction until the trajectory decoder. Hierarchical fusion splits encoder depth between modality-specific and cross-modal blocks. Early fusion keeps only modality-specific projections and lets one self-attention encoder decide which roads, agents, and signals matter.

![Wayformer's encoder-decoder architecture for multimodal scene inputs](/assets/images/wayformer-motion-forecasting-via-simple-and-efficient-attention-networks-paper-figure.png)
*Fig 1: A homogeneous scene encoder projects traffic lights, road geometry, histories, and agent interactions into a shared representation before a trajectory decoder predicts mixture components. | source: [Wayformer, Figure 1](https://arxiv.org/abs/2207.05844)*

The decoder uses learned queries and cross-attention to produce a Gaussian mixture over trajectories. The benchmark models emit 64 mixture components, then aggregate them to the six modes required by the evaluation. That distinction matters: the internal distribution is richer than the final `k=6` predictions used for minADE, minFDE, miss rate, overlap, and mAP.

### Efficiency comes from exploiting the token geometry

Joint attention over spatial and temporal tokens has cost `O(S_m² × T²)`. Factorized attention reduces this to `O(S_m²) + O(T²)`, either by processing all temporal blocks before all spatial blocks (sequential) or by alternating them (interleaved). Latent-query attention adds another control: the first block maps `L_in` input tokens to `L_out` learned latents, so the reduction ratio `L_out/L_in` directly controls later self-attention and feed-forward cost.

![Wayformer's factorized-attention comparison](/assets/images/wayformer-motion-forecasting-via-simple-and-efficient-attention-networks-source-figure-5.webp)
*Fig 2: Factorized attention traces quality and latency tradeoffs for early, late, and hierarchical fusion; the same nominal reduction can have different effects because road tokens are tiled in cross-modal encoders. | source: [Wayformer, Figure 5](https://arxiv.org/abs/2207.05844)*

The result is not simply “factorized is faster.” The authors find similar quality for sequential and interleaved attention, but latency improves most clearly for late fusion. In early and hierarchical fusion, tiling roadgraph tokens to the common temporal dimension can erase the expected savings. This is a good example of why a complexity expression must be checked against the actual tokenization.

### Latent queries buy a predictable quality-speed tradeoff

Across the fusion variants, latent queries speed models by roughly 2–16× with minimal to no quality regression in the reported ablation, while early and hierarchical fusion retain the best quality. On the WOMD benchmark, the early-fusion multi-axis model reports minFDE 1.128, minADE 0.545, miss rate 0.123, overlap 0.127, and mAP 0.419. The factorized version reports 1.126/0.545/0.123/0.127/0.412. On Argoverse, the corresponding minADE values are 0.7675 and 0.7672, with minFDE 1.1615 and 1.1625.

![Wayformer's latent-query reduction curve](/assets/images/wayformer-motion-forecasting-via-simple-and-efficient-attention-networks-source-figure-6.webp)
*Fig 3: Reducing the latent token count lowers latency while exposing the point at which compressing the scene begins to cost minADE. | source: [Wayformer, Figure 6](https://arxiv.org/abs/2207.05844)*

The benchmark setup uses 1 second of history and 8 seconds of future at 5 Hz on WOMD, and 2 seconds of history and 3 seconds of future at 10 Hz on Argoverse. Models train with AdamW for 1M steps, batch size 256, and a linear learning-rate decay. The Argoverse leaderboard result additionally ensembles 15 replicas, each with 10 decoders, before reducing the mixture to six trajectories.

## High-Level Takeaways

- Wayformer is useful when an architecture must absorb heterogeneous scene inputs without multiplying modality-specific modules. Early fusion is the simplest strong baseline in the paper's tested regimes.
- Factorized attention's real benefit depends on token layout. Roadgraph tiling can dominate the cost even when the asymptotic formula looks favorable.
- Latent queries provide an explicit latency knob, but they compress the scene before the decoder sees it. The right reduction ratio is an empirical operating point, not a universal constant.
- The reported six-mode metrics include trajectory aggregation and, for leaderboard settings, ensembling. Fusion claims should therefore be compared with matched aggregation and inference budgets.
