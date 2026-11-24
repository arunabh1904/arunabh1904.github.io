---
title: 'High-Resolution Image Synthesis with Latent Diffusion Models'
date: '2021-12-20T09:00:00.000Z'
section: paper-shorts
postSlug: high-resolution-image-synthesis-with-latent-diffusion-models
legacyPath: /paper shorts/2021/12/20/high-resolution-image-synthesis-with-latent-diffusion-models.html
tags: [diffusion, generative-models, representation-learning]
field: "Generative Modeling"
summary: "2021 – High-Resolution Image Synthesis with Latent Diffusion Models"
---

## 2021 – High-Resolution Image Synthesis with Latent Diffusion Models

[Paper](https://arxiv.org/abs/2112.10752) · [Code](https://github.com/CompVis/latent-diffusion)

## Summary

> Latent diffusion separates image compression from generative modeling. A pretrained autoencoder removes expensive spatial detail; a diffusion model learns to generate its smaller latent grid. The useful operating point preserves enough information for a good decoder while making repeated denoising cheaper. Compression, sampling steps, conditioning, and guidance all affect the result, so the paper's gains need to be read with those settings intact.

## Core Insights

### Move the repeated work, not just the output

Pixel-space diffusion evaluates a denoising network repeatedly on a large image grid. Here an encoder first maps an image to a spatial latent $z=\mathcal{E}(x)$. A decoder learns the reverse reconstruction. Perceptual and patch-adversarial losses favor recognizable, locally realistic images over purely pixelwise agreement. The paper tests weak KL regularization and a vector-quantized variant; neither makes the latent a lossless copy of the input.

The encoder and decoder are then fixed while a time-conditioned U-Net learns to predict noise added to the latent. During generation, the U-Net repeatedly denoises a latent sample; the decoder is used once at the end. This preserves a two-dimensional spatial interface rather than flattening the image into an autoregressive sequence.

![Original schematic of latent diffusion with compression, repeated latent denoising, and decoding.](/assets/images/latent-diffusion-original-schematic.svg)

*Fig 1: Original schematic of repeated denoising in a smaller latent grid. Training latents come from the encoder; generation begins from latent noise. | source: [Rombach et al., Figure 3](https://arxiv.org/html/2112.10752v2#S3.F3)*

[Open the diagram at full size](/assets/images/latent-diffusion-original-schematic.svg)

With downsampling factor $f$, the latent grid has $HW/f^2$ spatial locations. This is a spatial-size reduction, not an equal end-to-end speedup: channel counts, network layers, attention, sampling steps, and decoding still cost work.

### Conditioning chooses what to generate

The noise-prediction objective becomes

$$
\mathbb{E}_{x,y,\epsilon,t}\left[\left\|\epsilon-\epsilon_\theta(z_t,t,\tau_\theta(y))\right\|_2^2\right].
$$

For text, a transformer encodes the prompt. Intermediate U-Net features provide attention queries, while the conditioning representation supplies keys and values. Spatially aligned conditions, such as a low-resolution image, can instead be concatenated with the denoiser's input. These are two different interfaces, not a requirement to express every condition as text.

### Compression and guidance must remain visible in comparisons

The compression study tests factors from 1 to 32 on a single A100. Factors 4 and 8 offer a strong quality/throughput balance in the tested settings. Factor 1 leaves expensive perceptual modeling to diffusion; factor 32 discards too much information before diffusion starts. A larger reduction is therefore not automatically a better latent space.

Table 3 reports class-conditional ImageNet FID of 10.56 for LDM-4 and 3.60 with classifier-free guidance at scale 1.5, both using 250 sampling steps. Guidance also changes precision from 0.71 to 0.87 and recall from 0.62 to 0.48. The better FID comes with a distribution-coverage trade-off.

For text-conditioned MS-COCO evaluation, Table 2 reports FID 23.31 without guidance and 12.63 with guidance for the 1.45B-parameter LDM-KL-8, again at 250 steps. Those numbers describe the paper's LAION-trained system, not every later model called latent diffusion. Sampling settings and training data matter when comparing it with GLIDE or an autoregressive baseline.

### A realistic reconstruction can still be wrong

The paper demonstrates generation, inpainting, semantic synthesis, and super-resolution. These tasks do not share one definition of correctness. Super-resolution can produce convincing texture while losing pixelwise agreement with the reference. A decoder cannot reliably recover arbitrary fine detail that compression removed, and plausible completion is not evidence that hidden content was observed.

Sequential sampling also remains slower than a single-pass GAN. I would choose the compression level by checking the smallest task-relevant detail first, then compare full sampling latency and quality at a fixed guidance setting. A visually attractive sample alone does not answer either question.

## High-Level Takeaways

- Train an image compressor first, then model its spatial latent distribution.
- Run repeated denoising in the latent grid and decode the final sample once.
- Preserve compression, guidance, and sampling-step settings beside benchmark results.
- Use reconstruction fidelity as a task constraint, especially when exact details matter.
