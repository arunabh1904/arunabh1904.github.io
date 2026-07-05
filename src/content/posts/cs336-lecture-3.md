---
title: Lecture 3
date: '2025-05-22T04:00:00.000Z'
section: revision-notes
postSlug: cs336-lecture-3
legacyPath: /revision notes/2025/05/22/cs336-lecture-3.html
tags:
  - Other
summary: CS336 Lecture 3 — LLM Architecture & Training Stability
---
# CS336 Lecture 3 — LLM Architecture & Training Stability

*Revision notes updated May 24 2025*

---

## Quick Overview

This lecture surveys architecture choices in modern large language models and the tricks that keep training stable. The useful lens is comparative: what changed from earlier Transformers, and which choices survived in recent LLMs?

---

## Normalization Techniques

- RMSNorm has largely replaced LayerNorm because it uses less memory while delivering similar performance. It omits mean adjustment and bias terms.
- Applying normalization before each block improves gradient flow and helps prevent loss spikes. Some models add an extra norm after the block as well.
- Large networks often drop bias terms from linear layers because the simpler parameterization trains reliably.

## Activation Functions and MLPs

- Modern models favor gated linear units such as SwiGLU or GeGLU over ReLU or GELU.
- GLUs gate the first linear layer's output elementwise, which consistently improves over non-gated activations.

## Serial vs. Parallel Layers

- Most architectures keep the classic attention-then-MLP order. A few compute both branches in parallel for GPU efficiency, but serial layers remain more expressive.

## Position Embeddings

- Rotary Position Embeddings (RoPE) have become standard. They rotate queries and keys inside attention so relative position is built into the dot product.

## Hyperparameter Guidelines

- Feed-forward dimensions scale to four times the model width for non-GLU MLPs or roughly eight-thirds times width with GLUs.
- Set the head dimension so that model dimension equals head dimension times the number of heads.
- A hidden-to-layer ratio around 128 provides a good balance of width and depth.
- Vocabulary sizes have grown to the 100k–250k range, especially for multilingual models.
- Weight decay remains useful in pre-training, whereas dropout is less common.

## Training Stability Tricks

- Z-loss penalizes deviations of the softmax normalizer from one to avoid numerical issues.
- Applying LayerNorm to queries and keys (QK Norm) bounds softmax inputs, preventing gradient spikes.

## Attention Variations for Inference

- Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) reduce key/value cache size by sharing keys and values across heads, which speeds up generation.

## Long Context Handling

- Models mix full self-attention without positions for periodic global context and sliding-window attention with RoPE for local detail.

## Source

Percy Liang, **CS336 — Large Language Models**, Stanford University, Lecture 3: *Architecture* (Winter 2025).

## Slides & Recording

- [Recording](https://www.youtube.com/watch?v=ptFiH_bHnJw)
- [Slides](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%203%20-%20architecture.pdf)
