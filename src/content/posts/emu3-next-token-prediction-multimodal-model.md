---
title: 'Emu3: Next-Token Prediction Is All You Need'
date: '2024-09-28T00:00:00.000Z'
section: paper-shorts
postSlug: emu3-next-token-prediction-multimodal-model
legacyPath: /paper shorts/2024/09/28/emu3-next-token-prediction-multimodal-model.html
tags: [Omni-Model Architectures, Multimodal Generation]
field: 'Omni-Model Architectures'
summary: '2024 – Emu3: Next-Token Prediction Is All You Need'
---

## 2024 – Emu3: Next-Token Prediction Is All You Need

**arXiv:** [2409.18869](https://arxiv.org/abs/2409.18869)

### Method and reported result

Emu3 represents text, images, and video as discrete tokens and trains one transformer from scratch with next-token prediction. The paper reports competitive image generation, visual understanding, and video generation against specialist systems, including diffusion-based image and video models and CLIP-connected vision-language models.

## Summary

> Emu3 treats a visual model as a language model with a better visual tokenizer. That makes the training objective unusually legible: learn the next symbol whether it belongs to a sentence, a frame, or an image. The trade is equally legible: visual compression decides what the symbol sequence can preserve.

## Core Insights

![Emu3 architecture: one transformer predicts interleaved text, image, and video tokens](/assets/images/emu3-paper-figure-1.png)
*Fig 1: Emu3 tokenizes video, images, and text, then trains a single transformer to predict the next token in the mixed sequence. | source: [Emu3, Figure 1](https://arxiv.org/abs/2409.18869)*

Emu3's conceptual move is smaller than a new multimodal loss and more consequential than it first appears. Once a tokenizer maps pixels and frames into discrete symbols, the transformer no longer needs separate objectives for “understanding” and “generation.” It can condition on image tokens when answering a question, continue a video token stream, or emit image tokens after text. The model is trained from scratch on a mixture of language, image, and video data, so the shared interface is present throughout training rather than added after a language-only model has already formed its abstractions.

![Emu3 original and reconstructed video and image samples](/assets/images/emu3-next-token-prediction-multimodal-model-source-figure-2.webp)
*Fig 2: Reconstruction samples compare original and reconstructed 540 × 960 videos, sampled at eight frames and 30 FPS, alongside 512 × 512 images. | source: [Emu3, Figure 3](https://arxiv.org/abs/2409.18869)*

The reconstruction figure is the most useful intuition for the architecture. A next-token model cannot recover detail that the visual tokenizer discarded. When the reconstructions preserve the scene, color, and object layout, the transformer has a workable alphabet for generation; when small text, texture, or temporal changes disappear, no amount of language-side reasoning can restore them reliably. The paper's image and video results therefore test two coupled systems: the autoregressive model and the codec that defines its visual vocabulary.

![Emu3 text-to-image generation samples across varied prompts and styles](/assets/images/emu3-next-token-prediction-multimodal-model-source-figure-3.webp)
*Fig 3: Qualitative text-to-image samples show Emu3 generating varied scenes, objects, people, and styles through causal visual-token prediction. | source: [Emu3, Figure 4](https://arxiv.org/abs/2409.18869)*

The reported comparison in Figure 2 aggregates human evaluation for image generation, twelve visual-understanding benchmarks, and VBench for video generation. Emu3 beats the listed SDXL, LLaVA-1.6, and OpenSora baselines in those comparisons, but the result is not a proof that one objective dominates every specialist. The systems differ in data, tokenizers, parameterization, and inference procedure. Emu3 does make a narrower point convincingly: a single causal model can be a serious multimodal baseline without a diffusion decoder or a CLIP-style contrastive bridge. The paper also applies direct preference optimization to autoregressive vision generation, showing that preference alignment can sit on the same token interface.

| Question | Emu3's answer | What remains coupled |
| --- | --- | --- |
| What is shared? | One causal transformer and one next-token objective | Visual and textual token statistics still differ. |
| What is specialized? | Tokenizers and detokenizers for each modality | Codec fidelity controls the usable information. |
| Where does scale go? | Longer mixed sequences and larger token streams | Video makes context and sampling cost grow quickly. |

## High-Level Takeaways

- Emu3 is a strong test of whether next-token prediction can cover perception and generation with one causal interface.
- The codec is the hidden model boundary: reconstruction quality determines which visual facts are even available to the transformer.
- Video generation is especially demanding because temporal consistency becomes a long-range token-prediction problem rather than a separate denoising process.
- The reported specialist comparisons are encouraging, but they are not compute-matched evidence against diffusion or contrastive systems.
- If building a unified model, start by measuring token-rate, reconstruction fidelity, and context cost before celebrating objective simplicity.
