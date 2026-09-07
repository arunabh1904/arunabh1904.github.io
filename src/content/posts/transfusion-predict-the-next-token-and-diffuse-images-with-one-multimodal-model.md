---
title: 'Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model'
date: '2024-08-20T00:00:00.000Z'
section: paper-shorts
postSlug: transfusion-predict-the-next-token-and-diffuse-images-with-one-multimodal-model
legacyPath: /paper shorts/2024/08/20/transfusion-predict-the-next-token-and-diffuse-images-with-one-multimodal-model.html
tags: [Multimodal AI]
field: 'Omni-Model Architectures'
summary: "2024 – Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model"
---
## 2024 – Transfusion

**arXiv:** [2408.11039](https://arxiv.org/abs/2408.11039)<br>
**Conference:** Technical report

## Summary

> Transfusion shares one transformer across text and images while preserving different prediction rules: text uses next-token prediction, while images use continuous VAE patches with a diffusion loss. Boundary tokens create one sequence; causal text attention and bidirectional within-image attention preserve modality structure. In a controlled Chameleon comparison, the 7B Transfusion model reaches FID 16.8 versus 29.6 at 0.5T tokens and needs an estimated 2.9% of Chameleon’s FLOPs to match FID. VAE, U-Net, and loss-weight costs remain exposed.

## Core Insights

### One transformer supports two prediction objectives

Transfusion builds a mixed sequence from discrete text tokens and continuous image patches. Text strings use a normal vocabulary. Images pass through a VAE, are arranged left-to-right and top-to-bottom as latent patches, and are wrapped with beginning-of-image (BOI) and end-of-image (EOI) tokens. The transformer sees one vector sequence, but the training losses stay modality-specific:

$$
L_{Transfusion}=L_{LM}+\lambda L_{DDPM}.
$$

The language loss is computed per text token. The diffusion loss is computed over a whole noised latent image before patchification. The paper sets $\lambda=5$ after preliminary experiments. This is a shared-parameter model with a non-shared statistical interface.

![Source Figure 1 from Transfusion: text and image blocks processed by one transformer](/assets/images/transfusion-predict-the-next-token-and-diffuse-images-with-one-multimodal-model-paper-figure.png)
*Fig 1: A text prefix emits a BOI marker, a continuous image block is denoised in parallel, and an EOI marker returns decoding to the language path; the transformer is shared while the modality heads differ. | source: [Transfusion, Figure 1](https://arxiv.org/abs/2408.11039)*

Attention follows the same separation. Every position is causally masked against later sequence elements, but patches from one image attend bidirectionally to one another. Thus a caption can condition an image, and an image can condition a later caption, while image patches do not have to predict one another left-to-right. At inference, language decoding samples token by token; after BOI, the model appends Gaussian-noise patches and runs the diffusion reverse process before emitting EOI.

### Compression comes from the VAE and the patch interface

![Source Figure 3 from Transfusion: VAE latents converted to image patches through linear or U-Net blocks](/assets/images/transfusion-predict-the-next-token-and-diffuse-images-with-one-multimodal-model-source-figure-3.webp)
*Fig 2: A VAE maps pixels to a continuous latent grid, and a linear layer or U-Net down/up path converts local latent windows to and from transformer vectors; the choice changes both sequence length and inductive bias. | source: [Transfusion, Figure 3](https://arxiv.org/abs/2408.11039)*

The canonical VAE has 86M parameters, latent dimension 8, and maps a 256×256 image to a 32×32×8 latent tensor. With 2×2 latent patching, that is 256 image elements; with 8×8 patching, one image can be represented by 16 elements. A linear encoder/decoder adds almost no parameters, while U-Net down/up blocks add 0.27B parameters across configurations and provide spatial inductive bias.

The 16-patch result is therefore a sequence-length result, not a free compression theorem. Larger patches let the model see more images per training batch and reduce attention and diffusion serving cost, but they can reduce text and image quality when the encoder is too simple. The paper’s useful question is not whether continuous patches are smaller in isolation; it is whether their retained information and learned spatial bias produce a better model at the same effective compute.

### Matched scaling favors continuous image training in this experiment

The controlled comparison trains Transfusion and Chameleon at 0.16B, 0.37B, 0.76B, 1.4B, and 7B transformer sizes on 0.5T tokens with a 1:1 text/image token ratio. Both use matched VAE data and architecture; Chameleon additionally pays for a VQ-VAE codebook and stability modifications. The largest controlled models use 2×2 latent patches, simple linear image layers, and bidirectional intra-image attention.

![Source Figure 5 from Transfusion: six-metric scaling comparison against Chameleon](/assets/images/transfusion-predict-the-next-token-and-diffuse-images-with-one-multimodal-model-source-figure-5-full.png)
*Fig 3: Figure 5 compares Transfusion (red) and Chameleon (blue) across six scaling metrics: C4 and Wikipedia perplexity, Llama 2 accuracy, MS-COCO CIDEr, FID, and CLIP. Lower is better for perplexity and FID; higher is better for accuracy, CIDEr, and CLIP. The fitted lines summarize the tested scale range, not an asymptotic law. | source: [Transfusion, Figure 5](https://arxiv.org/abs/2408.11039)*

| 7B model, 0.5T tokens | Transfusion | Chameleon | Relative FLOPs to match Chameleon |
| --- | ---: | ---: | ---: |
| C4 perplexity | 7.72 | 8.41 | 0.489 |
| Wikipedia perplexity | 4.28 | 4.69 | 0.526 |
| Llama evaluation accuracy | 61.5 | 59.1 | 0.600 |
| MS-COCO CIDEr | 27.2 | 18.0 | 0.218 |
| MS-COCO FID | 16.8 | 29.6 | 0.029 |
| MS-COCO CLIP | 25.5 | 24.3 | 0.319 |

The FID parity ratio is the striking number: the fitted comparison estimates that Transfusion needs about 2.9% of Chameleon’s FLOPs to reach the 7B Chameleon result. The table does not mean an image is 34 times cheaper in every deployment. It uses a controlled proxy $6ND$, and continuous representations shorten the sequence, so the authors calculate theoretical FLOPs to remove that particular sequence-length confounder.

The large 7B Transfusion model adds U-Net image blocks and trains on 2T multimodal tokens: 1T text tokens and about 3.5B image-caption pairs. The paper reports 66.1 Llama evaluation accuracy, MS-COCO FID 6.78, and GenEval 0.63, compared with SDXL’s GenEval 0.55 and DeepFloyd’s 0.61. The cited SD3 result reaches 0.68 using synthetic captions, so the comparison also changes the caption data. Transfusion’s distinctive evidence is that one model also generates text, not that it dominates every image-only system.

### The ablations locate the source of the gain

Intra-image bidirectional attention is decisive for the simple linear image path: at 0.76B, FID falls from 61.3 with causal attention to 20.3 with bidirectional attention, and CIDEr rises from 12.7 to 16.0. The U-Net already has internal bidirectional structure, so its FID changes only 16.8 to 16.7. This is a mechanistic result: continuous image patches need to exchange spatial information inside the transformer unless the modality-specific blocks provide it.

At 0.76B with U-Net encoding, reducing each image from 256 to 64 or 16 patches gives FID 16.7, 16.0, and 16.1, while Llama accuracy falls from 51.9 to 50.7 and 49.2. The paper’s interpretation is that larger patches expose more distinct images and diffusion noise during training but force the transformer to learn more visual content per vector. Limiting diffusion noise to $t\le500$ when images precede captions improves 7B CIDEr from 33.7 to 35.2, showing that noisy image conditioning can specifically harm captioning.

## High-Level Takeaways

- Transfusion shares transformer parameters while preserving next-token prediction for text and diffusion for continuous images; the loss boundary is the core design decision.
- Continuous VAE patches remove a discrete image-token bottleneck, but VAE quality, patch size, U-Net parameters, and diffusion steps determine the actual cost.
- Intra-image bidirectional attention and U-Net spatial bias explain much of the image advantage in the ablations, so the headline comparison is not only about continuous versus discrete values.
- Shortening the image sequence shifts work into the image interface rather than removing it. The patch-size ablation shows why spatial inductive bias matters: a U-Net can preserve image quality under compression even while text accuracy falls.
