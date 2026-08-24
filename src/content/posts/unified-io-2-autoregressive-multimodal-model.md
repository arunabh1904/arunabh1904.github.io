---
title: 'Unified-IO 2: Scaling Autoregressive Multimodal Models'
date: '2023-12-28T00:00:00.000Z'
section: paper-shorts
postSlug: unified-io-2-autoregressive-multimodal-model
legacyPath: /paper shorts/2023/12/28/unified-io-2-autoregressive-multimodal-model.html
tags: [Omni-Model Architectures, Multimodal Generation]
field: 'Omni-Model Architectures'
summary: '2023 – Unified-IO 2: Scaling Autoregressive Multimodal Models'
---

## 2023 – Unified-IO 2: Scaling Autoregressive Multimodal Models

**arXiv:** [2312.17172](https://arxiv.org/abs/2312.17172)

## Summary

> Unified-IO 2 expresses image, text, audio, boxes, and actions as token sequences handled by one encoder-decoder transformer. The same autoregressive objective supports both understanding and generation across more than 100 tasks. The paper tests whether a common token interface can replace a collection of task-specific heads.

## Core Insights

![Unified-IO 2 architecture encoding and decoding text, images, audio, and actions](/assets/images/unified-io-2-paper-figure-2.png)
*Modality-specific tokenizers feed one transformer, and modality-specific decoders recover the requested output. source: [Unified-IO 2](https://arxiv.org/abs/2312.17172)*

![Figure 2 from Unified-IO 2: Scaling Autoregressive Multimodal Models](/assets/images/unified-io-2-autoregressive-multimodal-model-source-figure-2.webp)
*Figure 2 Unified-IO 2 architecture. Input text, images, audio, or image/audio history are encoded into sequences of embeddings which are concatenated and used as input to an encoder-decoder transformer model. The transformer outputs discrete tokens that can be decoded into text, an image, or an audio clip. source: [Unified-IO 2: Scaling Autoregressive Multimodal Models](https://arxiv.org/abs/2312.17172)*

![Figure 3 from Unified-IO 2: Scaling Autoregressive Multimodal Models](/assets/images/unified-io-2-autoregressive-multimodal-model-source-figure-3.webp)
*Figure 3 Left : Training loss (a) and gradient norms (b) on different modality mixtures. Right : Training loss (c) and next token prediction accuracy (d) of UIO-2 on all modalities. Results were obtained before applying the proposed architectural improvements. source: [Unified-IO 2: Scaling Autoregressive Multimodal Models](https://arxiv.org/abs/2312.17172)*


Unification happens at the sequence interface, not at the raw signal. Images, audio, text, and actions still need different tokenizers and output decoders. The shared transformer then models their relationships through one next-token training objective.

This makes task composition simple, but tokenization determines the cost and fidelity of every modality. A shared sequence can hide conflicts between semantic abstraction, pixel reconstruction, temporal precision, and action control.

## High-Level Takeaways

- Unified-IO 2 puts many modalities and tasks behind one autoregressive interface.
- Shared sequence modeling does not remove modality-specific compression choices.
- The clean objective trades task-specific heads for longer contexts and possible representation interference.
