---
title: Auto-Encoding Variational Bayes
date: '2014-04-01T04:00:00.000Z'
section: paper-shorts
postSlug: auto-encoding-variational-bayes
legacyPath: /paper shorts/2014/04/01/auto-encoding-variational-bayes.html
tags:
  - Other
field: Variational Inference
summary: VAEs made latent-variable models trainable with SGD by turning stochastic sampling into a differentiable reparameterized path.
---
## 2014 – Auto-Encoding Variational Bayes

**arXiv:** [1312.6114](https://arxiv.org/abs/1312.6114)

**GitHub:** [pyro-ppl/vae](https://github.com/pyro-ppl/vae) (example implementation)

**Project page:** n/a

**Conference:** ICLR 2014

## Paper map

The VAE paper solves inference for latent-variable models whose posterior is intractable but differentiable. It introduces the reparameterization trick: sample noise from a fixed distribution, transform it through encoder outputs, and backpropagate through the stochastic latent variable. The objective is the evidence lower bound, combining reconstruction likelihood with a KL term that keeps the approximate posterior near the prior. An encoder, or recognition model, amortizes inference across datapoints instead of optimizing a separate variational distribution for each one. The experiments show generative modeling and semi-supervised learning can be trained with stochastic gradient descent. The caveat is the usual VAE tradeoff: simple likelihoods and strong KL pressure can produce blurry samples or underused latents.

![Figure from Auto-Encoding Variational Bayes: AEVB improves the variational lower bound over wake-sleep](/assets/images/vae-paper-figure-1.png)
_Figure from the [AEVB paper](https://arxiv.org/abs/1312.6114), via ar5iv._

**Summary:** Kingma and Welling made variational inference feel like ordinary neural-network training. Their variational autoencoder pairs an encoder $q_\phi(z \mid x)$, which approximates the posterior over latent variables, with a decoder $p_\theta(x \mid z)$, which reconstructs observations from those latents. The key move is the reparameterisation trick: instead of sampling $z$ in a way that blocks gradients, sample fixed noise and transform it through differentiable parameters.

That trick turns Monte Carlo estimates of the evidence lower bound (ELBO) into low-variance gradients that work with standard stochastic gradient descent. The paper also made amortised inference practical at scale: the encoder learns to predict posterior parameters directly, rather than solving a separate inference problem for every datapoint.

**Why it mattered:** VAEs gave deep generative modelling a stable likelihood-based recipe. They did not produce the sharpest samples, but they made latent-variable models trainable, inspectable, and useful for representation learning. Later work such as $\beta$-VAE, conditional VAEs, and flow-based models all build on this basic encoder-decoder view of probabilistic inference.

**Evals / Latency benchmarks:**

| Dataset | Latent dim | $-\log p(x)$ ↓ (nats) | Notes |
| ------- | ---------- | ---------------------- | ----- |
| Binarised MNIST | 30 | ≈ 88 nats (ELBO estimate) | Competed closely with deep latent-Gaussian models |
| Frey Faces | 2 | Smooth latent manifold | Visually coherent reconstructions |

Training cost was modest for the original experiments: minutes per epoch on MNIST using a single GPU in 2013. Because the method uses SGD mini-batches and one objective, wall-clock time mostly comes down to ordinary forward and backward passes.

**Critiques & limitations:** The strength of VAEs is stability: training has a clear objective, avoids GAN-style mode collapse, and produces a latent space that supports interpolation. The tradeoff is expressiveness. Simple Gaussian priors and posteriors can underfit complex data, and the ELBO is only a lower bound, so a good training objective does not guarantee a tight estimate of true likelihood.

**Take-home message:** Auto-Encoding Variational Bayes turned variational inference into a scalable deep-learning procedure. The reparameterisation trick is the small mathematical hinge that made the whole recipe practical.
