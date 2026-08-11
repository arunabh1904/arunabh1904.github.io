---
title: PyTorch Tensor Mechanics
date: '2026-08-10T04:00:00.000Z'
section: revision-notes
postSlug: pytorch-tensor-mechanics
legacyPath: /revision notes/2026/08/10/pytorch-tensor-mechanics.html
tags:
  - PyTorch
summary: A focused route through tensor construction, metadata, views, indexing, broadcasting, and layout decisions.
---

# PyTorch Tensor Mechanics

## Quick overview

The fastest way to debug tensor code is to predict the tensor's shape, dtype, device, and aliasing before reading its values. This note follows one object through construction, storage metadata, indexing, broadcasting, and layout changes.

This is the first part of the [PyTorch Revision Notes](/revision%20notes/2026/08/09/pytorch-tensor-revision-notes.html). Continue with [Autograd and training](/revision%20notes/2026/08/10/pytorch-autograd-and-training.html) once the tensor contract is clear.

## Construction is a copy or sharing decision

`torch.tensor(data)` copies its input and creates a fresh leaf by default. `torch.as_tensor` avoids a copy where possible, `torch.from_numpy` intentionally shares CPU storage with a NumPy array, and `torch.frombuffer` exposes a Python buffer. The choice is part of the tensor's correctness contract because mutation, dtype conversion, and device placement can change what is shared.

Factory functions make allocation intent explicit. The `*_like` family starts from an existing tensor's shape and, by default, its dtype, layout, and device; pass overrides when those properties should differ. `torch.empty` allocates without initializing values, so every element must be overwritten before it is read.

## Metadata maps indices to storage

A strided tensor is a logical array described by shape, strides, and storage offset. For index $(i_0, \ldots, i_{n-1})$, its storage position is

$$
\text{storage\_offset} + \sum_{k=0}^{n-1} i_k\,\text{stride}_k.
$$

This is why a slice or transpose can be cheap: it can change metadata without moving values. The same mechanism creates non-contiguity and aliasing. `view` requires compatible strides; `reshape` may return a view or make a copy; `clone` copies while preserving gradient connectivity; and `detach` removes the autograd edge while normally sharing storage.

## Indexing and broadcasting change the logical view

Integer indexing removes a dimension, slicing preserves it, `None` inserts one, and advanced indexing copies selected values. Assignment through either basic or advanced indexing mutates the destination. Broadcasting aligns dimensions from the right; dimensions are compatible when equal or when one is `1`. `expand` represents the larger logical shape with zero strides, so it should not be treated as independent writable storage.

## Layout is an API contract

Dense `torch.strided` tensors are the default, but channels-last, sparse, quantized, nested, and jagged layouts change operator support and memory behavior. Inspect layout and contiguity at boundaries where a downstream operator depends on them. Use a specialized layout only when its supported operator sequence and measured workload justify the additional invariants.

## Retrieval hook

Before changing tensor code, write down: `shape → dtype → device → layout → strides → aliasing`. That sequence exposes most silent copies, invalid broadcasts, and in-place view bugs.

## Source links

- [PyTorch 2.13 tensors](https://docs.pytorch.org/docs/2.13/tensors.html)
- [Tensor attributes](https://docs.pytorch.org/docs/2.13/tensor_attributes.html)
- [Tensor views](https://docs.pytorch.org/docs/2.13/tensor_view.html)
- [PyTorch Interview Preparation](https://github.com/rohanmistry231/PyTorch-Interview-Preparation)
