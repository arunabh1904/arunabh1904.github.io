---
title: PyTorch Autograd and Training
date: '2026-08-10T04:00:00.000Z'
section: revision-notes
postSlug: pytorch-autograd-and-training
legacyPath: /revision notes/2026/08/10/pytorch-autograd-and-training.html
tags:
  - PyTorch
summary: A focused route through autograd graphs, gradient accumulation, modules, optimization, evaluation, and checkpointing.
---

# PyTorch Autograd and Training

## Quick overview

Training is a controlled sequence: construct registered state, run a forward computation, reduce to a loss, differentiate, process gradients, update parameters, and evaluate under a separate module mode. This note makes the boundaries between those steps explicit.

This is the second part of the [PyTorch Revision Notes](/revision%20notes/2026/08/09/pytorch-tensor-revision-notes.html). Start with [Tensor mechanics](/revision%20notes/2026/08/10/pytorch-tensor-mechanics.html) if shape, storage, or device behavior is unclear.

## Backward is a vector-Jacobian product

PyTorch records operations when grad mode is enabled and at least one input requires gradients. A scalar output can call `.backward()` directly. A non-scalar output needs an upstream gradient, because backward computes a vector-Jacobian product rather than materializing a full Jacobian.

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2
y.backward(torch.ones_like(y))
print(x.grad)  # tensor([2., 4.])
```

Only leaf tensors accumulate into `.grad` by default. Non-leaf gradients require `retain_grad()`, and gradients accumulate across backward calls until they are cleared. `torch.autograd.grad` is preferable when gradients should be returned as values rather than accumulated as state.

## Gradient modes do different jobs

`detach()` removes a tensor from the graph but normally shares storage. `torch.no_grad()` temporarily prevents graph construction and is useful for updates or preprocessing. `torch.inference_mode()` removes more autograd tracking for isolated inference. `model.eval()` changes mode-dependent module behavior such as Dropout and BatchNorm; it does not disable gradients.

## The training invariant

The ordinary update is `zero_grad → forward → loss → backward → optional clipping or scaling → optimizer.step`. Use `optimizer.zero_grad(set_to_none=True)` when the distinction between “no gradient” and “zero gradient” is useful. Construct the optimizer after parameters are materialized and placed on the intended device.

AMP changes the order around gradient processing: autocast covers forward and loss, `GradScaler` scales the loss for float16, and gradients must be unscaled before clipping or inspection. Evaluation sets `model.eval()` and disables autograd separately.

## State and checkpoints

`nn.Module` registers parameters, buffers, and child modules into a state tree. Use `ModuleList` and `ModuleDict` instead of ordinary Python containers when state must move, serialize, or reach the optimizer. Save `state_dict()` rather than pickling an entire module; resumable training may also require optimizer, scheduler, scaler, step, configuration, RNG, and sampler state.

## Retrieval hook

When a training result is wrong, inspect the boundary in order: did the model receive the intended batch, did the loss depend on the parameters, did gradients arrive, were they scaled or clipped correctly, and did the optimizer update the same registered parameters?

## Source links

- [Autograd mechanics](https://docs.pytorch.org/docs/2.13/notes/autograd.html)
- [PyTorch Interview Preparation: autograd basics](https://github.com/rohanmistry231/PyTorch-Interview-Preparation/blob/main/PyTorch%20Fundamentals/01%20Core%20PyTorch%20Foundations/02%20Autograd/autograd_basics.py)
- [PyTorch Interview Preparation](https://github.com/rohanmistry231/PyTorch-Interview-Preparation)
