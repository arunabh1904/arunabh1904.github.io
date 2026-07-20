---
title: CS336 Lecture 3 — LLM Architecture and Training Stability
date: '2025-05-22T04:00:00.000Z'
section: revision-notes
postSlug: cs336-lecture-3
legacyPath: /revision notes/2025/05/22/cs336-lecture-3.html
tags:
  - Other
summary: CS336 Lecture 3 — LLM Architecture & Training Stability
---
# CS336 Lecture 3 — LLM Architecture and Training Stability

*Revision notes updated May 24, 2025.*

## Quick overview

This lecture surveys architecture choices in modern large language models and the tricks that keep training stable. The useful lens is comparative: what changed from earlier Transformers, and which choices survived in recent LLMs?

## Normalization techniques

- RMSNorm has largely replaced LayerNorm because it uses less memory while delivering similar performance. It omits mean adjustment and bias terms.
- Applying normalization before each block improves gradient flow and helps prevent loss spikes. Some models add an extra norm after the block as well.
- Large networks often drop bias terms from linear layers because the simpler parameterization trains reliably.

## Activation functions and MLPs

- Modern models favor gated linear units such as SwiGLU or GeGLU over ReLU or GELU.
- GLUs gate the first linear layer's output elementwise, which consistently improves over non-gated activations.

## Serial versus parallel layers

- Most architectures keep the classic attention-then-MLP order. A few compute both branches in parallel for GPU efficiency, but serial layers remain more expressive.

## Position embeddings

- Rotary Position Embeddings (RoPE) have become standard. They rotate queries and keys inside attention so relative position is built into the dot product.

## Hyperparameter guidelines

- Feed-forward dimensions scale to four times the model width for non-GLU MLPs or roughly eight-thirds times width with GLUs.
- Set the head dimension so that model dimension equals head dimension times the number of heads.
- A hidden-to-layer ratio around 128 provides a good balance of width and depth.
- Vocabulary sizes have grown to the 100k–250k range, especially for multilingual models.
- Weight decay remains useful in pre-training, whereas dropout is less common.

## Training stability tricks

- Z-loss penalizes deviations of the softmax normalizer from one to avoid numerical issues.
- Applying LayerNorm to queries and keys (QK Norm) bounds softmax inputs, preventing gradient spikes.

## Attention variants for inference

- Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) reduce key/value cache size by sharing keys and values across heads, which speeds up generation.

## Long-context handling

- Models mix full self-attention without positions for periodic global context and sliding-window attention with RoPE for local detail.

## Source

Percy Liang, **CS336: Language Modeling from Scratch**, Stanford University, Lecture 3: *Architecture* (Spring 2025).

## Slides and recording

- [Recording](https://www.youtube.com/watch?v=ptFiH_bHnJw)
- [Slides](https://stanford-cs336.github.io/spring2025-lectures/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf)
