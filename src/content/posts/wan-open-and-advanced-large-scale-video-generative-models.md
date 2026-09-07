---
title: 'Wan: Open and Advanced Large-Scale Video Generative Models'
date: '2025-03-26T00:00:00.000Z'
section: paper-shorts
postSlug: wan-open-and-advanced-large-scale-video-generative-models
legacyPath: /paper shorts/2025/03/26/wan-open-and-advanced-large-scale-video-generative-models.html
tags: [Video Generation]
field: 'Video & Interactive World Models'
summary: "2025 – Wan: Open and Advanced Large-Scale Video Generative Models"
---

**arXiv:** [2503.20314](https://arxiv.org/abs/2503.20314)  
**GitHub:** [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)  
**Conference:** Technical report

## Summary

> Wan is a systems report about making open video generation both capable and usable. A causal Wan-VAE compresses video by 4×8×8, a flow-matching DiT handles the resulting latent sequence with umT5 cross-attention, and a staged data pipeline controls quality and motion across resolutions. The release includes 1.3B and 14B models: the report gives the smaller model an 8.19 GB VRAM target and reports 83.96 VBench for Wan 1.3B versus 86.22 for Wan 14B.

## Core Insights

### The autoencoder sets the cost of every later decision

![Wan architecture with causal video VAE, flow-matching DiT blocks, timestep, and umT5 text conditioning](/assets/images/wan-open-and-advanced-large-scale-video-generative-models-paper-figure.png)
*Fig 1: Wan encodes video into a compact latent, denoises it with text-conditioned DiT blocks, and decodes the result; the text encoder and latent compression are part of the generation path. | source: [Wan, Figure 9](https://arxiv.org/abs/2503.20314)*

Wan-VAE compresses a video from [1+T, H, W, 3] to [1+T/4, H/8, W/8, 16]. The first frame is spatially compressed only, while later frames use temporal compression. The 3D causal design, RMSNorm replacement for GroupNorm, and cached convolution features preserve the rule that future frames cannot influence earlier ones. The feature cache processes at most four input frames per latent chunk and reuses the last two historical features for ordinary causal convolutions.

The training sequence is also a systems choice. Wan first trains a 2D image VAE, inflates it into a 3D causal model, trains on 128×128 five-frame videos, and then fine-tunes on high-quality videos with L1, KL, LPIPS, and a 3D GAN loss. The resulting 127M VAE is small enough to keep encoding from dominating the later DiT. On a 200-video, 25-frame, 720×720 reconstruction test, the report says Wan-VAE is 2.5× faster than Hunyuan Video at a competitive reconstruction-quality/efficiency point.

### Data provisioning is part of the generator

![Wan data provisioning pipeline across image, video, text, training, and filtering stages](/assets/images/wan-open-and-advanced-large-scale-video-generative-models-source-figure-3.webp)
*Fig 2: The report routes image, video, and text pools through resolution-specific training, visual filters, motion filters, deduplication, and SFT stages. | source: [Wan, Figure 3](https://arxiv.org/abs/2503.20314)*

Wan’s pretraining pool combines internal copyrighted sources and public data, then applies fundamental, visual, motion, and semantic filtering. Fundamental checks remove excessive text, watermarks, unsafe content, borders, overexposure, synthetic contamination, blur, and unsuitable duration or resolution. Visual quality is scored after clustering so long-tail categories are not erased. Motion quality separates useful motion from static interviews, camera-only motion, jitter, occlusion, and crowded low-quality footage. The pipeline provisions different mixtures at 192p, 480p, and 720p rather than pretending that every example has the same training value.

The DiT consumes these curated latents with a 3D (1,2,2) patchifier. A shared AdaLN MLP predicts timestep modulation parameters while each block keeps its own biases; the ablation finds that adding depth with shared AdaLN beats spending parameters on non-shared normalization. umT5 supplies 512-token bilingual text embeddings through cross-attention. This combination is what lets the model scale a long video sequence without turning the text interface into a decoder-only bottleneck.

### Flow matching and distributed inference connect quality to deployment

Wan trains with flow matching: sample noise x₀ and data latent x₁, interpolate xₜ = t x₁ + (1−t)x₀, and predict velocity x₁−x₀. The curriculum starts with 256px images and 192px, 5-second videos at 16 FPS, then moves to joint 480px and finally 720px training. Pretraining uses bf16 AdamW with weight decay 1e−3 and an initial learning rate of 1e−4 reduced when FID and CLIP metrics plateau.

![Wan-VAE reconstruction quality and efficiency compared with other video autoencoders](/assets/images/wan-open-and-advanced-large-scale-video-generative-models-source-figure-7.webp)
*Fig 3: Wan-VAE occupies a high-PSNR, high-efficiency point in the 720×720, 25-frame reconstruction comparison; circle area represents parameter count. | source: [Wan, Figure 7](https://arxiv.org/abs/2503.20314)*

The report evaluates 1,035 samples per candidate on Wan-Bench and more than 700 human tasks annotated by over 20 people. On VBench, Wan 14B reports 86.67 visual quality, 84.44 semantic consistency, and 86.22 aggregate score; Wan 1.3B reports 84.92, 80.10, and 83.96. These comparisons indicate a practical quality–resource curve, not just a parameter race. The paper also reports downstream image-to-video, editing, personalization, camera control, real-time streaming, and audio extensions, all built on the foundation model.

Inference efficiency is treated as part of the release. Two-dimensional context parallelism combines Ring Attention and Ulysses; diffusion caching reuses similar attention and classifier-free-guidance computations for a reported 1.62× speedup on Wan 14B, while FP8 GEMM adds a 1.13× DiT speedup. The remaining limit is conceptual: VBench and human preference measure appearance, motion, and prompt alignment, but they do not establish causal world consistency or action-conditioned control. The data pool and benchmark also include internal sources and model-specific filtering decisions that are difficult for an outside team to reproduce exactly.

## High-Level Takeaways

- Wan makes the video VAE a first-class architectural choice: its compression ratio and causal cache determine the sequence length, memory profile, and temporal fidelity available to the DiT.
- Data provisioning, motion filtering, flow-matching curriculum, and shared AdaLN are all part of the scaling story; quality does not come from the 14B transformer alone.
- The 1.3B/14B pair plus VBench and reconstruction plots expose a useful quality–resource tradeoff, while cache and parallelism techniques make the open release more deployable.
- Strong video generation and prompt adherence are not evidence of a controllable world model. Long-horizon physical consistency, intervention response, and reproducible data provenance remain separate questions.
