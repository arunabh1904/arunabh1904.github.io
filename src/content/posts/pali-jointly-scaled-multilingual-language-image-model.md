---
title: 'PaLI: A Jointly-Scaled Multilingual Language-Image Model'
date: '2022-09-14T00:00:00.000Z'
section: paper-shorts
postSlug: pali-jointly-scaled-multilingual-language-image-model
legacyPath: /paper shorts/2022/09/14/pali-jointly-scaled-multilingual-language-image-model.html
tags: [Vision-Language Models, Multimodal Scaling]
field: 'Multimodal Scaling & Data Mixtures'
summary: '2022 – PaLI: A Jointly-Scaled Multilingual Language-Image Model'
---

**arXiv:** [2209.06794](https://arxiv.org/abs/2209.06794)

## Summary

> PaLI is an encoder-decoder model in which a large ViT turns an image into patch features and an mT5-style text stack turns the combined image-and-prompt sequence into text. The 17B configuration combines a 13B language component with a roughly 4B ViT, and is trained on a filtered multilingual WebLI mixture. The paper's scaling result is unusually concrete: language capacity, visual capacity, and a later high-resolution phase each add measurable gains, but the evidence is tied to this pretrained model family and data recipe.

## Core Insights

### A single text interface makes the visual path easy to scale

PaLI keeps the task interface deliberately uniform. The image is passed through a ViT without pooling; its patch features are placed into the encoder alongside text, and the decoder generates the answer, caption, or translated text. Prompts identify the task, so captioning, VQA, OCR-heavy questions, and multilingual transfer use the same text-generation machinery rather than separate prediction heads.

The family makes the scaling choice visible. PaLI-3B pairs an approximately 1B mT5-Large with a 1.8B ViT-G, PaLI-15B pairs a 13B mT5-XXL with ViT-G, and PaLI-17B replaces that visual tower with the roughly 4B ViT-e. In the largest model the visual encoder is about a quarter of the parameter count, which is large enough for visual representation quality to become a first-class bottleneck.

![PaLI architecture: image and prompt features enter a shared encoder-decoder interface](/assets/images/pali-jointly-scaled-multilingual-language-image-model-source-figure-1.webp)
*Fig 1: PaLI feeds ViT patch features and the text prompt into an encoder, then generates the answer with a text decoder; the image and query therefore share one task interface. | source: [PaLI, Figure 1](https://arxiv.org/abs/2209.06794)*

### Joint scaling is a matched comparison, not a parameter slogan

The paper's most useful comparison is Figure 2. Moving from mT5-Large to mT5-XXL adds about 12B language parameters and improves the average over seven reported tasks by 3.1 points. Holding that language component fixed and moving from ViT-G to ViT-e adds roughly 2B visual parameters, only about 13% of the total model, yet improves the same average by 3.2 points. A later high-resolution phase adds another 2.0 points to that average.

That pattern explains why a modest-looking vision increase matters. In Table 5's 224×224 zero-shot image classification, PaLI-17B with ViT-e scores 72.11 top-1 on ImageNet versus 70.27 for PaLI-15B with ViT-G; these are generative zero-shot class scores, not a separately fine-tuned vision classifier. The paired model moves COCO captioning from 146.2 to 149.1 CIDEr and VQAv2 from 82.9 to 83.4. Visual capacity is paying off where the model must ground generated language in an image, even when a vision-only score barely moves.

![PaLI scaling comparison across language, vision, and high-resolution stages](/assets/images/pali-jointly-scaled-multilingual-language-image-model-source-figure-2.webp)
*Fig 2: The bars compare PaLI-3B, PaLI-15B, PaLI-17B, and the additional high-resolution phase across captioning and VQA tasks; the empty bars isolate the high-resolution contribution. | source: [PaLI, Figure 2](https://arxiv.org/abs/2209.06794)*

### WebLI and initialization are part of the result

WebLI is described as 10B images with 12B alt-texts in 109 languages plus 29B image-OCR pairs. PaLI does not train on all of that raw collection: the main filtered image-text subset is about 1B examples, and the full pretraining mixture is about 1.566B examples spanning text-only, alt-text, OCR, multilingual captioning, VQA, visual question generation, object-aware data, and detection. The first phase uses 224×224 images with the ViT frozen while the language side adapts; PaLI-17B then receives 10,000 high-resolution steps over about 10M examples at 588×588 with all parameters updated.

The ablations show why these details cannot be treated as implementation trivia. Initializing from mT5 and ViT-G gives PaLI-3B 141.4 CIDEr on COCO, 93.8 English and 42.5 six-language Crossmodal-3600 scores, and 41.6 TextVQA. Training the same-sized model from scratch drops those numbers to 72.8, 22.1/10.1, and 12.8. A multilingual WebLI mixture raises Crossmodal-3600 from 8.2 to 30.0 on the six-language average even when English changes much less. The reported gains therefore combine joint scaling with strong unimodal initialization and a carefully composed multilingual corpus.

## High-Level Takeaways

- PaLI makes visual capacity an explicit scaling axis inside a text-generation system.
- The reported gains separate language scaling, vision scaling, and high-resolution adaptation well enough to guide a matched ablation plan.
- WebLI's language coverage and the pretrained mT5/ViT initialization are necessary context for interpreting the 17B results.
- The paper supports this design in its tested encoder-decoder family; it does not show that every multimodal architecture should allocate a quarter of its parameters to vision.
