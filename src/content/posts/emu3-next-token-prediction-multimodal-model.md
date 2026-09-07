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

## Summary

> Emu3 treats a visual model as a language model with a visual tokenizer. Text, images, and video become discrete symbols, and one transformer predicts the next symbol. The reported comparison is unusually concrete: human evaluation on 100 English prompts with three voters per prompt, a twelve-benchmark vision average, and VBench for video generation, all against specialist baselines.

## Core Insights

### Discrete visual tokens extend next-token prediction

Once a tokenizer maps pixels and frames into discrete symbols, the transformer no longer needs separate objectives for “understanding” and “generation.” It can condition on image tokens while answering a question, continue a video token stream, or emit image tokens after text. Emu3 is trained from scratch on a mixture of language, image, and video data, so the shared interface is present throughout training rather than added after a language-only model has already formed its abstractions.

![Emu3 architecture: one transformer predicts interleaved text, image, and video tokens](/assets/images/emu3-paper-figure-1.png)
*Fig 1: Emu3 tokenizes video, images, and text, then trains a single transformer to predict the next token in the mixed sequence. | source: [Emu3, Figure 1](https://arxiv.org/abs/2409.18869)*

The codec is the real model boundary. A next-token model cannot recover detail that the visual tokenizer discarded, so reconstruction quality determines what the transformer can later generate or reason about.

![Emu3 original and reconstructed video and image samples](/assets/images/emu3-next-token-prediction-multimodal-model-source-figure-2.webp)
*Fig 2: Reconstruction samples compare original and reconstructed 540 × 960 videos, sampled at eight frames and 30 FPS, alongside 512 × 512 images. | source: [Emu3, Figure 3](https://arxiv.org/abs/2409.18869)*

The reconstruction figure gives the right intuition for the architecture. When the reconstructions preserve scene layout, color, and object identity, the transformer has a workable visual alphabet. If small text, texture, or temporal changes disappear in the codec, language-side reasoning cannot restore them reliably. The generation result is therefore a property of both the causal model and the discrete representation.

![Emu3 text-to-image generation samples across varied prompts and styles](/assets/images/emu3-next-token-prediction-multimodal-model-source-figure-3.webp)
*Fig 3: Qualitative text-to-image samples show Emu3 generating varied scenes, objects, people, and styles through causal visual-token prediction. | source: [Emu3, Figure 4](https://arxiv.org/abs/2409.18869)*

The paper's image-generation protocol uses 100 diverse user prompts judged by three independent voters for visual quality and prompt following. Emu3 beats SDXL in the overall human score and is reported as comparable to DALL-E 3 and MJ-v5.2. On the automated DPG-Bench text-to-image evaluation, Emu3-DPO reaches 81.6 overall, above SDXL's 74.65 and PixArt-alpha's 71.11 and near DALL-E 3's 83.50. Those are separate claims: the human comparison concerns base model generation, while the 81.6 score is the DPO-aligned model on a benchmark.

The source Figure 2 comparison also places Emu3 against LLaVA-1.6-7B on the average of twelve visual-understanding benchmarks and against OpenSora-1.2 on VBench. The model beats the listed baselines in those plots, but those systems differ in data, tokenizers, parameterization, and inference procedure. The narrower result is stronger: a causal model can be a serious multimodal baseline without a diffusion decoder or a CLIP-style contrastive bridge. The paper also applies direct preference optimization to autoregressive vision generation, showing that preference alignment can remain on the same token interface.

| Question | Emu3's answer | What remains coupled |
| --- | --- | --- |
| What is shared? | One causal transformer and one next-token objective | Visual and textual token statistics still differ. |
| What is specialized? | Tokenizers and detokenizers for each modality | Codec fidelity controls usable information. |
| Where does scale go? | Longer mixed sequences and larger token streams | Video makes context and sampling cost grow quickly. |

## High-Level Takeaways

- Emu3 is a serious test of whether next-token prediction can cover perception and generation with one causal interface.
- The reconstruction figure exposes the hidden boundary: the codec decides which visual facts are available before the transformer starts reasoning.
- The 100-prompt, three-voter human protocol and the DPG-Bench numbers make the generation claim checkable, while the specialist comparisons still have unequal systems.
- Video generation turns temporal consistency into a long-range token-prediction problem rather than a separate denoising process.
- Objective simplicity does not remove the need to measure visual token rate, reconstruction fidelity, and context cost.
