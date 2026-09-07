---
title: 'Scaling Laws for Native Multimodal Models'
date: '2025-04-10T00:00:00.000Z'
section: paper-shorts
postSlug: scaling-laws-for-native-multimodal-models
legacyPath: /paper shorts/2025/04/10/scaling-laws-for-native-multimodal-models.html
tags: [Multimodal AI]
field: 'Multimodal Scaling & Data Mixtures'
summary: "2025 – Scaling Laws for Native Multimodal Models"
---

**arXiv:** [2504.07951](https://arxiv.org/abs/2504.07951)  
**Conference:** ICCV 2025 (oral)

## Summary

> Across 457 native multimodal models, this study compares early fusion, late fusion, dense training, and sparse MoE training under scaling-law fits. In the tested image-text regime, early and late fusion have almost identical loss-versus-FLOP exponents, but early fusion is stronger at small sizes, uses less memory, and trains faster. Sparse early fusion improves loss at the same active-parameter cost and develops modality-specific experts without requiring hand-written modality routing.

## Core Insights

### Equal loss scaling does not imply equal systems cost

The early-fusion model linearly projects 14×14 image patches into the text width and feeds them to one transformer; the late-fusion line follows a CLIP-style vision encoder before the decoder. Both are trained from scratch on interleaved, image-caption, and text-only data, with a 1k multimodal context. The reported average validation laws are close: early fusion follows $L\propto C^{-0.0492}$ and late fusion $L\propto C^{-0.0494}$. At small model sizes, however, Figure 3 puts early fusion slightly below late fusion, and the gap narrows as parameter count grows.

The difference is in how the compute budget is spent. Table 2 gives early fusion $N_{\mathrm{opt}}\propto C^{0.526}$ and $D_{\mathrm{opt}}\propto C^{0.468}$ for the average loss, while late fusion uses approximately $N_{\mathrm{opt}}\propto C^{0.636}$ and $D_{\mathrm{opt}}\propto C^{0.462}$. These late-fusion exponents should not be read as a literal parameter/token split that must sum to one: the paper defines its compute as $C\approx6(N_vD_v+ND)$, with $N_v$ and $D_v$ for the separate vision encoder and $N,D$ for the multimodal decoder. The result is that late fusion's compute-optimal decoder is more parameter-heavy, while early fusion benefits more from tokens. Figure 4's matched 16-H100 comparison shows the practical consequence: early fusion trains faster and consumes less memory for the same compute budget.

![Scaling laws for early fusion, late fusion, and sparse early-fusion MoE models](/assets/images/scaling-laws-for-native-multimodal-models-paper-figure.png)
*Fig 1: The upper panel compares validation-loss scaling; the lower panel shows how the parameter-to-token trade-off changes with compute for early, late, and sparse early fusion. | source: [Scaling Laws for Native Multimodal Models, Figure 1](https://arxiv.org/abs/2504.07951)*

### Mixture composition changes which resource is valuable

The default early-fusion mixture is 45% image-caption, 45% interleaved, and 10% text-only data. The authors also fit 40-20-40, 30-30-40, and 20-40-40 mixtures. The compute exponents remain similar, but the balance moves: when image-caption data is increased, the fitted token exponent rises and the parameter exponent falls. The paper's explanation is mechanistic: image-caption examples contain more image tokens, so adding that domain increases the token burden; increasing interleaved and text-only data supplies relatively more text tokens.

That is why a single “multimodal Chinchilla ratio” would be misleading. The fitted average losses remain predictable, but the best way to use more compute depends on the token composition of the mixture. The paper also reports that early fusion can catch up to an LLM-initialized model with longer native training: fewer than 100B multimodal tokens match on image-caption data, while interleaved and text-only performance may require up to 1T tokens.

### Learned specialization beats a fixed modality split

Sparse early fusion uses a dropless top-1 MoE with eight experts and a 0.01 load-balancing loss. At the same active-parameter cost, sparse models beat dense early-fusion models, especially at smaller sizes; their fitted average law is $L\propto C^{-0.0474}$ with a lower multiplicative loss constant. The sparse law also favors tokens more heavily than parameters, reflecting the total expert capacity hidden behind each active path.

The routing ablation matters. Modality-aware routing sends image tokens to image experts and text tokens to text experts, but the learned modality-agnostic router performs better on both image-caption and interleaved data. The source Figure 13 visualizes why: experts in the early layer are close to unimodal, middle layers share more, and the final layers specialize again. The system discovers a useful division of labor while preserving the option to share representations.

![Expert token specialization frequency in the first MoE layer](/assets/images/scaling-laws-for-native-multimodal-models-source-figure-12.webp)
*Fig 2: The source plot shows the fraction of text and image tokens assigned to each expert in layer 0; experts are strongly modality-skewed even though the router was not given modality labels. | source: [Scaling Laws for Native Multimodal Models, Figure 13](https://arxiv.org/abs/2504.07951)*

## High-Level Takeaways

- Early and late fusion have similar loss scaling here, but their parameter-token trade-offs and system costs differ.
- Sparse early fusion can add capacity and specialization at fixed active-parameter cost.
- Learned routing outperforms a hand-written modality split in the tested image-text mixtures.
- The conclusions are bounded by the 275M–3.7B model range, image-text data families, and validation-loss proxy; broader modalities and serving costs need fresh scaling runs.
