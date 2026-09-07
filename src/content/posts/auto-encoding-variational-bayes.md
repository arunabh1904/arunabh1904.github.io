---
title: Auto-Encoding Variational Bayes
date: '2013-12-20T00:00:00.000Z'
section: paper-shorts
postSlug: auto-encoding-variational-bayes
legacyPath: /paper shorts/2014/04/01/auto-encoding-variational-bayes.html
tags:
  - Other
field: 'Generative Modeling'
summary: "2014 – Auto-Encoding Variational Bayes"
---
## 2014 – Auto-Encoding Variational Bayes

**arXiv:** [1312.6114](https://arxiv.org/abs/1312.6114)<br>
**GitHub:** [pyro-ppl/vae](https://github.com/pyro-ppl/vae) (example implementation)<br>
**Project page:** n/a<br>
**Conference:** ICLR 2014

## Summary

> AEVB makes posterior inference in a nonlinear latent-variable model trainable with ordinary minibatch gradients. It rewrites the variational lower bound around a recognition network and samples as $z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon$, so the randomness is independent of the encoder parameters. On MNIST and Frey Faces, the method reaches a better variational bound faster than wake-sleep across latent dimensions and remains usable on large minibatches. The evidence is an optimization and inference result on small image datasets; it does not show that a diagonal Gaussian posterior is adequate for every generative model.

## Core Insights

### The lower bound turns an intractable posterior into a trainable target

AEVB starts with a directed model $p_\theta(z)p_\theta(x\mid z)$. The marginal likelihood

$$
\log p_\theta(x)=\log\int p_\theta(z)p_\theta(x\mid z)\,dz
$$

is hard to evaluate when the decoder is a nonlinear neural network, and the posterior $p_\theta(z\mid x)$ is hard to sample from. The paper introduces a recognition model $q_\phi(z\mid x)$ and writes

$$
\log p_\theta(x)=D_{KL}\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big)+\mathcal L(\theta,\phi;x).
$$

The second term is the evidence lower bound:

$$
\mathcal L=-D_{KL}\big(q_\phi(z\mid x)\,\|\,p_\theta(z)\big)+\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)].
$$

This decomposition makes the trade-off explicit. The decoder must explain the observation, while the encoder is regularized toward the prior. The gap is posterior mismatch, so maximizing the bound is useful even when exact marginal likelihood remains unavailable.

![Source Figure 2 from Auto-Encoding Variational Bayes: lower-bound optimization against wake-sleep](/assets/images/auto-encoding-variational-bayes-source-figure-2.webp)
*Fig 1: Across MNIST and Frey Face latent dimensions, the AEVB curves rise to a better variational bound faster than wake-sleep; the plot is evidence about optimization, not a sample-quality ranking. | source: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)*

The plot makes the amortization claim concrete. One encoder is updated for every minibatch, rather than running a separate iterative inference procedure for each example. On MNIST the comparison spans $N_z=3,5,10,20,200$; on Frey Faces it spans $N_z=2,5,10,20$. The extra latent coordinates do not visibly cause the bound to overfit in these experiments, which the paper attributes to the KL regularizer.

### Reparameterization removes the high-variance gradient estimator

A direct Monte Carlo gradient through a draw from $q_\phi(z\mid x)$ uses a score-function estimator. The paper notes that this estimator has high variance. For a diagonal Gaussian recognition model, it instead draws fixed noise and transforms it:

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\qquad \epsilon\sim\mathcal N(0,I).
$$

The expectation over $q_\phi$ becomes an expectation over $\epsilon$, so the sampled path is differentiable with respect to both $\mu_\phi$ and $\sigma_\phi$. In the VAE example, the prior is $\mathcal N(0,I)$, the encoder is an MLP that predicts the mean and standard deviation, and the decoder uses a Bernoulli likelihood for binary data or a Gaussian likelihood for real-valued data. When the KL term is analytic, only the reconstruction expectation needs Monte Carlo samples.

The experimental algorithm uses minibatches of $M=100$ points and $L=1$ latent sample per point. A single sample is sufficient in the reported setting because the minibatch averages the estimator noise. That choice is part of the result: reparameterization is valuable because it makes a noisy objective cheap enough to optimize repeatedly, not because it removes stochasticity.

### The marginal-likelihood comparison exposes the scale advantage

![Source Figure 3 from Auto-Encoding Variational Bayes: estimated marginal likelihood on two MNIST training-set sizes](/assets/images/auto-encoding-variational-bayes-source-figure-3.webp)
*Fig 2: With three latent variables, AEVB and wake-sleep improve the estimated marginal likelihood online, while MCEM is much slower on the larger MNIST training set. | source: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)*

For the marginal-likelihood experiment, the authors use 100-hidden-unit encoder and decoder networks, three latent variables, the first 1,000 training and test points, and 50 posterior samples per point for the estimator. The small and large panels use $N_{train}=1{,}000$ and $50{,}000$. MCEM uses a Hybrid Monte Carlo sampler and is not an online algorithm; the paper therefore treats its behavior on the full MNIST set as a scalability boundary rather than as a cleaner baseline.

The paper also gives a useful representation check. In two dimensions, it maps a grid through the inverse Gaussian CDF and decodes each latent point. The result is a smooth manifold of faces or digits, but smooth interpolation is a property of this prior, decoder, and training setup. It is not evidence that every semantic factor has become an independent latent coordinate.

![Source Figure 4 from Auto-Encoding Variational Bayes: learned two-dimensional Frey Face and MNIST manifolds](/assets/images/auto-encoding-variational-bayes-source-figure-4.png)
*Fig 3: Decoding a grid in the Gaussian latent prior yields continuous face and digit manifolds, showing how the learned decoder organizes nearby latent points. | source: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)*

The closest comparison, wake-sleep, also uses a recognition model and has the same per-datapoint complexity, but its wake and sleep objectives do not jointly optimize a marginal-likelihood bound. Its advantage is support for discrete latent variables. AEVB’s gain is narrower and more consequential: for continuous latents, the encoder and decoder can be trained together with one differentiable bound.

## High-Level Takeaways

- AEVB’s durable mechanism is the reparameterized stochastic path, which turns encoder inference into ordinary backpropagation through a sampled latent.
- The KL term is both a tractable regularizer and the source of the bound’s gap from exact evidence; a better reconstruction score alone does not close that gap.
- The reported $M=100$, $L=1$ minibatch regime demonstrates scalable optimization, while MCEM exposes why per-example iterative inference does not scale as cleanly.
- The diagonal Gaussian posterior and simple likelihoods are deliberate tractability choices. A matched test with richer posteriors should check whether they are limiting coverage or only simplifying the first demonstration.
