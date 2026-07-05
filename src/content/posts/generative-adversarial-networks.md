---
title: Generative Adversarial Networks
date: '2014-06-01T04:00:00.000Z'
section: paper-shorts
postSlug: generative-adversarial-networks
legacyPath: /paper shorts/2014/06/01/generative-adversarial-networks.html
tags:
  - Other
field: Generative Models
summary: 2014 – Generative Adversarial Networks
---
## 2014 – Generative Adversarial Networks

**arXiv:** [1406.2661](https://arxiv.org/abs/1406.2661)

**GitHub:** [goodfeli/adversarial](https://github.com/goodfeli/adversarial)

**Project page:** n/a

**Conference:** NIPS 2014

**Summary:** GANs turn generative modelling into a contest. A generator $G$ maps random noise into synthetic samples, while a discriminator $D$ learns to tell real data from generated data. Training alternates between making $D$ better at the classification problem and making $G$ better at fooling $D$.

The paper's central promise is elegant: with enough capacity and ideal optimization, the generator recovers the true data distribution and the discriminator becomes maximally uncertain, outputting 0.5 everywhere. That framing recasts density estimation as a differentiable game rather than an explicit likelihood problem. It also gives the generator direct feedback in pixel space, which helped avoid some of the blurry samples associated with early likelihood-based models.

**Why it mattered:** The first experiments were small, mostly multilayer perceptrons on MNIST-like data, but the idea opened a much larger path. GANs made high-quality sample generation feel less like approximate density estimation and more like learned counterfeiting.

**Evals / Latency benchmarks:** The original paper reports qualitative results on MNIST and small CIFAR-like images, not likelihood metrics. A small model on a single K20Xm GPU trained in minutes; ImageNet-scale synthesis came later. The evaluation gap is part of the legacy: GANs eventually pushed the community toward sample-quality metrics such as Inception Score and FID.

**Critiques & limitations:** The formulation is beautiful, but the optimization is fragile. Training can diverge, generators can collapse to a small set of modes, and discriminators can saturate until the generator receives weak gradients. Much of the later GAN literature, including Wasserstein GANs and gradient penalties, is really a response to those failure modes.

**Take-home message:** GANs introduced adversarial training as a powerful but unstable route to generative modelling. The paper's impact comes from both sides of that sentence: the samples were exciting, and the instability became an entire research program.
