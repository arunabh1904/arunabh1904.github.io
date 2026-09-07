---
title: 'Perceiver IO: A General Architecture for Structured Inputs & Outputs'
date: '2021-07-30T00:00:00.000Z'
section: paper-shorts
postSlug: perceiver-io-a-general-architecture-for-structured-inputs-and-outputs
legacyPath: /paper shorts/2021/07/30/perceiver-io-a-general-architecture-for-structured-inputs-and-outputs.html
tags:
  - Other
field: 'Omni-Model Architectures'
summary: "2021 – Perceiver IO: A General Architecture for Structured Inputs & Outputs"
---
## 2021 – Perceiver IO

**arXiv:** [2107.14795](https://arxiv.org/abs/2107.14795)

**Code:** [google-deepmind/deepmind-research/perceiver](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver)

## Summary

> Perceiver IO separates data shape from model shape. It reads an arbitrary input array into a fixed latent workspace, processes that workspace with self-attention, and decodes arbitrary outputs through queries that describe the requested positions, tasks, or modalities. The resulting complexity is linear in input and output size, while the expensive processing depth depends on the latent array.

## Core Insights

### A fixed latent workspace separates the interface from the computation

Let the input be $x\in\mathbb{R}^{M\times C}$, the latent array be $z\in\mathbb{R}^{N\times D}$, and the output be $y\in\mathbb{R}^{O\times E}$. Perceiver IO performs three operations:

1. Encode input elements into the latent array with cross-attention.
2. Apply a stack of self-attention and MLP blocks to the $N$ latent elements.
3. Decode output elements by using an $O$-element query array to cross-attend to the latents.

The central question is whether one latent workspace can serve many output structures. The source architecture diagram makes the data flow explicit:

![Perceiver IO encode, process, and decode path](/assets/images/perceiver-io-a-general-architecture-for-structured-inputs-and-outputs-source-figure-2.png)
*Fig 1: Arbitrary input elements are compressed into latents, latent self-attention performs the expensive processing, and output queries read the information needed for each output element. | source: [Perceiver IO: A General Architecture for Structured Inputs & Outputs, Figure 2](https://arxiv.org/abs/2107.14795)*

A standard Transformer repeatedly builds queries and keys over the full input, which makes each self-attention layer quadratic in sequence length. Perceiver IO uses attention non-homogeneously: input-to-latent cross-attention sees $M$ inputs once, latent processing sees only $N$ elements, and latent-to-output cross-attention sees $O$ queries. The model does not require the latent array to preserve the input’s grid or sequence layout.

### Queries carry the semantics of the requested output

The latent array is deliberately task agnostic. Output semantics enter through the queries. A classification task can use one learned query. A sequence or dense spatial output can attach a position embedding to each query. A multimodal output can combine modality and position embeddings. For optical flow, the query can also include the input feature at the location being decoded; for StarCraft II, it can include the unit representation.

The source’s domain overview shows the same interface spanning unlike data shapes:

![Perceiver IO across language, vision, multimodal autoencoding, and symbolic game outputs](/assets/images/perceiver-io-a-general-architecture-for-structured-inputs-and-outputs-source-figure-1.webp)
*Fig 2: The same encode–process–decode template is applied to multitask language, dense optical flow, video+audio+label autoencoding, and StarCraft II outputs. | source: [Perceiver IO: A General Architecture for Structured Inputs & Outputs, Figure 1](https://arxiv.org/abs/2107.14795)*

This query mechanism is why outputs can be computed in parallel: each output point depends on its query and the shared latent array. For very large outputs, the authors subsample output points during training and decode the full array in batches at test time. The output interface is flexible, but query design remains part of the model: a poor query does not tell the latent representation what to retrieve.

### The bottleneck buys scaling and creates a capacity decision

With feature size $F$ and $L$ latent processing blocks, the paper gives the attention complexity as

$$
O\big((M+O+LN)NF\big).
$$

The encoder and decoder are linear in input and output index sizes, while latent self-attention is independent of $M$ and $O$. This is the computational reason to use a bottleneck. It is also the representational risk: if $N$ or $D$ is too small, the latent array must discard information before the output query reveals which details matter.

The multimodal autoencoding experiment is an unusually clear stress test. The model serializes 50,000 video patches, 30,000 raw audio samples, and one 700-dimensional class label into a common input array. It uses 512-channel latents and reports a 88× compression setting with 784 latents:

![Multimodal audio-video-label autoencoding at 88× compression](/assets/images/perceiver-io-a-general-architecture-for-structured-inputs-and-outputs-source-figure-4.webp)
*Fig 3: Inputs are on the left and reconstructions on the right for the 88× compression setting; audio, video, and labels share the latent workspace. | source: [Perceiver IO: A General Architecture for Structured Inputs & Outputs, Figure 4](https://arxiv.org/abs/2107.14795)*

The table makes the tradeoff visible: at 88× compression, audio PSNR is 26.97, video PSNR is 24.37, and top-1 classification accuracy is 10.2%. At 176× and 352×, reconstruction quality changes as the latent bottleneck tightens. The authors also show that increasing the classification-loss weight can reach 45% top-1 accuracy while retaining 20.7 video PSNR, which is evidence that the shared latent can support competing modalities when the loss balance is chosen deliberately.

### The architecture is competitive where its interface matters

On GLUE, a SentencePiece Perceiver IO model reaches 81.2 average versus 81.1 for BERT Base at a comparable FLOPs budget. The byte-level version avoids tokenization and reaches 81.8 with a larger compute budget. On optical flow, Perceiver IO obtains 1.81 EPE on Sintel clean and 2.42 on Sintel final, compared with RAFT’s 1.95 and 2.57 in the reported AutoFlow comparison. The model uses no cost volumes or explicit warping, and its latent representation does not maintain a 2D layout.

These results establish that a general interface can be competitive, not that input structure is irrelevant. The optical-flow experiment still uses patch extraction, positional features, and RAFT’s AutoFlow augmentation parameters. Perceiver IO moves the burden from a fixed architecture to the input features, output queries, latent capacity, and loss weighting.

## High-Level Takeaways

- Use Perceiver IO when input size, output size, and output semantics vary independently. Encode into a fixed latent workspace, process there, and let queries specify what should be read out.
- The compute advantage is explicit: $M$ and $O$ enter linearly, while latent depth is controlled by $N$. The price is a bottleneck-capacity decision that can erase detail before the task-specific query sees it.
- The paper’s broad results are strongest as an interface demonstration. GLUE, Sintel, multimodal reconstruction, and StarCraft II use task-specific encodings, queries, and training recipes; the latent template does not remove that design work.
- A useful capacity check is to sweep latent count and width at matched FLOPs while measuring both task quality and reconstruction or fine-detail failure. The paper’s own latent-count ablation frames this as a real tradeoff, not a fixed universal setting.
