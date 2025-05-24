---
layout: content
title: "Auto-Encoding Variational Bayes"
date: 2014-04-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Variational Inference
---

## 2014 – Auto-Encoding Variational Bayes

**arXiv:** [1312.6114](https://arxiv.org/abs/1312.6114)

**GitHub:** [pyro-ppl/vae](https://github.com/pyro-ppl/vae) (example implementation)

**Project page:** n/a

**Conference:** ICLR 2014

**Summary (abstract in plain English):**
Kingma and Welling propose training deep generative models with continuous latent variables by
optimising a reparameterised evidence lower bound (ELBO) using standard stochastic gradient descent.
An encoder \(q_\phi(z \mid x)\) approximates the true posterior, while a decoder \(p_\theta(x \mid z)\)
reconstructs observations. The reparameterisation trick makes the stochastic nodes differentiable,
enabling efficient gradient learning.

**Novel insights:**
- Reparameterisation turns Monte Carlo gradients into low-variance, back-propagation-friendly estimates.
- Recognition networks fuse amortised inference with deep learning, making variational Bayes practical
  at scale.
- Established VAEs as a general-purpose unsupervised representation learner, inspiring variants such as
  \(\beta\)-VAE, conditional VAEs, and flow-based models.

**Evals / Latency benchmarks:**

| Dataset | Latent dim | \(-\log p(x)\) ↓ (nats) | Notes |
| ------- | ---------- | ---------------------- | ----- |
| Binarised MNIST | 30 | ≈ 88 nats (ELBO estimate) | Competed closely with deep latent-Gaussian models |
| Frey Faces | 2 | Smooth latent manifold | Visually coherent reconstructions |

Training cost: minutes per epoch on MNIST using a single GPU in 2013; linear in dataset size thanks to
SGD mini-batches. Optimisation is a single-objective loop, so wall-clock time is dominated by forward
and backward passes.

**Critiques & limitations:**
- **What works well:** Stable likelihood-based training with no mode collapse. Latent space enables
  interpolation and arithmetic.
- **Limitations:** Gaussian priors and posteriors can under-fit complex data; ELBO is only a lower bound,
  so the gap to true likelihood may be large.

**Take-home message:**
Auto-Encoding Variational Bayes introduced the reparameterisation trick and amortised variational
inference, providing a scalable recipe for deep generative modelling that remains a cornerstone of
probabilistic machine learning.
