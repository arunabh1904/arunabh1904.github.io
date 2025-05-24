---
layout: content
title: "Generative Adversarial Networks"
date: 2014-06-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Generative Models
---

## 2014 – Generative Adversarial Networks

**arXiv:** [1406.2661](https://arxiv.org/abs/1406.2661)

**GitHub:** [goodfeli/adversarial](https://github.com/goodfeli/adversarial)

**Project page:** n/a

**Conference:** NIPS 2014

**Summary (abstract in plain English):**
GANs formulate generative modelling as a game between two neural networks.
A generator \(G\) transforms random noise into synthetic samples, while a
 discriminator \(D\) learns to distinguish real data from fakes.
Training alternates between improving \(D\) to classify correctly and
adjusting \(G\) to fool \(D\).
At equilibrium, \(G\) replicates the true data distribution and \(D\) outputs
0.5 everywhere.
Initial experiments with multilayer perceptrons produced realistic MNIST
 digits without needing explicit likelihoods.

**Novel insights:**
- Adversarial training recasts density estimation as a differentiable,
  game-theoretic contest rather than maximising likelihood.
- With infinite capacity and optimal play, the generator converges to the
  data distribution and the discriminator becomes maximally uncertain.
- Direct pixel-space feedback avoids the blurriness common in VAEs of the time.

**Evals / Latency benchmarks:**
- Only qualitative results on MNIST and small CIFAR-like images were
  reported. No likelihood metrics appeared in the paper.
- Training a small model on a single GPU (K20Xm) took minutes; larger
  datasets like ImageNet were tackled in later work.
- The paper eventually motivated metrics such as Inception Score and FID,
  though they were developed later.

**Critiques & limitations:**
- **What works well:** Elegant formulation that opened a broad research area
  across image synthesis, style transfer and more.
- **Limitations:** Optimisation is fragile and can diverge. Generators often
  collapse to limited modes, and discriminators can saturate, providing
  poor gradients. Many variants such as Wasserstein-GAN and gradient
  penalties were invented to address these issues.

**Take-home message:**
Generative Adversarial Networks introduced adversarial training as a powerful,
but unstable, approach to generative modelling, inspiring a decade of
research into more robust GAN objectives and architectures.
