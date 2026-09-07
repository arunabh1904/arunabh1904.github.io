---
title: 'FAST: Efficient Action Tokenization for Vision-Language-Action Models'
date: '2025-01-16T00:00:00.000Z'
section: paper-shorts
postSlug: fast-efficient-action-tokenization-for-vision-language-action-models
legacyPath: /paper shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html
tags:
  - Other
field: 'Vision-Language-Action & Robotics'
summary: "2025 – FAST: Efficient Action Tokenization for Vision-Language-Action Models"
---
**arXiv:** [2501.09747](https://arxiv.org/abs/2501.09747)

## Summary

> FAST treats action tokenization as a compression problem. It normalizes a one-second action chunk, applies a discrete cosine transform (DCT) per action dimension, quantizes the coefficients, and uses byte-pair encoding (BPE) to turn the sparse frequency sequence into model tokens. FAST+ trains that tokenizer on one million real-robot action chunks so the same interface can transfer across robots and control rates.

## Core Insights

### DCT changes what the next-token loss sees

FAST targets a subtle failure in autoregressive policies. With the usual 256-bin representation, a one-second chunk with $D$ action dimensions and $H$ timesteps produces $D H$ tokens. At higher control rates, adjacent tokens become nearly identical, so the next-token objective receives little new information and a policy can learn to copy the first action. DCT reorganizes the same trajectory into low-frequency coefficients that describe its overall shape and high-frequency coefficients that describe sharp corrections. Quantization is followed by BPE, which removes repeated zero-valued coefficients without training a separate neural compressor.

![Figure 1 from FAST: Efficient Action Tokenization for Vision-Language-Action Models](/assets/images/fast-efficient-action-tokenization-for-vision-language-action-models-paper-figure.jpg)
*Fig 1: FAST compresses action trajectories into a shorter sequence while retaining the smooth structure needed for dexterous control; the paper reports up to 5× faster VLA training in the illustrated comparison. | source: [FAST, Figure 1](https://arxiv.org/abs/2501.09747)*

![Figure 2 from FAST: Efficient Action Tokenization for Vision-Language-Action Models](/assets/images/fast-efficient-action-tokenization-for-vision-language-action-models-source-figure-2.webp)
*Fig 2: Left: FAST tokenization enables training of autoregressive Transformers for dexterous robot control via simple next-token prediction. Right: FAST outperforms binning tokenization as control frequency rises. | source: [FAST, Figure 2](https://arxiv.org/abs/2501.09747)*

![Figure 8 from FAST: Efficient Action Tokenization for Vision-Language-Action Models](/assets/images/fast-efficient-action-tokenization-for-vision-language-action-models-source-figure-8.webp)
*Fig 3: FAST+ compression ratios on robot datasets unseen during tokenizer training; the ratio compares naïve token count with FAST+ token count. | source: [FAST, Figure 8](https://arxiv.org/abs/2501.09747)*

The compression is measurable before training a policy. On one-second chunks, the average token count falls from 35 to 20 on BridgeV2 (1.75×), 105 to 29 on DROID (3.6×), 140 to 28 on 20-Hz table bussing (5×), and 700 to 53 on 50-Hz T-shirt folding (13.2×). FAST is not lossless: the DCT scale controls the reconstruction-versus-compression trade-off, so these comparisons use settings with comparable reconstruction error.

### Compression becomes a policy result

The policy experiments show why token count matters. Naïve binning makes no progress on the 20-Hz table-bussing and 50-Hz T-shirt-folding tasks, while FAST and FSQ train usable policies; FAST is generally stronger on the dexterous real-robot tasks, and FAST+ matches dataset-specific FAST. The universal tokenizer reduces token counts by about 2× across held-out robot datasets, including different morphologies, action spaces, and frequencies. On DROID, a policy trained with FAST is evaluated zero-shot in an unseen tabletop environment by language prompting, without fine-tuning.

### A shorter sequence still has a control cost

The training advantage does not automatically mean faster control. The paper reports roughly 750 ms per one-second chunk for autoregressive $π_0$-FAST on an NVIDIA 4090, because 30–60 action tokens still pass through the full 2B language model; diffusion $π_0$ uses roughly ten denoising steps and is faster at inference. BPE is therefore part of the learning signal as well as the storage format: removing it leaves many repeated zeros, hundreds of tokens, and worse rollout performance.

## High-Level Takeaways

- FAST informs a representation decision that controls both training difficulty and action fidelity: a smooth trajectory deserves a compact sequence, while a sharp correction must survive reconstruction.
- The clearest evidence is the frequency sweep: naïve tokens degrade as sampling rises, whereas DCT compression keeps the autoregressive learning problem usable. The 13.2× T-shirt-folding compression ratio is a representation fact; it is not by itself a 13.2× control-speed claim.
- FAST+ transfers the tokenizer across held-out morphologies and frequencies, but the tokenizer is trained on one million action chunks and still needs quantile normalization and a chosen DCT scale. A new robot can use it as a black box, yet its bitrate-fidelity setting remains a deployment choice.
- The paper separates training efficiency from inference efficiency. BPE and frequency ordering improve the learning signal, but autoregressive decoding through the full language model can remain slower than diffusion.
- The important follow-up is closed-loop robustness at matched reconstruction error, token budget, and control latency, especially for contact-rich corrections that DCT truncation may smooth away.
