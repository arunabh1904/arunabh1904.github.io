---
title: Generative Adversarial Networks
date: '2014-06-10T00:00:00.000Z'
section: paper-shorts
postSlug: generative-adversarial-networks
legacyPath: /paper shorts/2014/06/01/generative-adversarial-networks.html
tags:
  - Other
field: 'Generative Modeling'
summary: "2014 – Generative Adversarial Networks"
---
## 2014 – Generative Adversarial Networks

**arXiv:** [1406.2661](https://arxiv.org/abs/1406.2661)<br>
**GitHub:** [goodfeli/adversarial](https://github.com/goodfeli/adversarial)<br>
**Project page:** n/a<br>
**Conference:** NIPS 2014

## Summary

> GANs replace explicit likelihood optimization with a learned density-ratio game. A generator maps noise to samples, a discriminator estimates whether they came from data, and alternating backpropagation updates both. The nonparametric analysis reduces the game to Jensen–Shannon divergence, with global solution $p_g=p_{data}$. On MNIST and the Toronto Face Database, experiments show direct samples and smooth latent interpolation; Parzen estimates are a noisy likelihood proxy. Finite networks can collapse modes or destabilize training, and the model has no explicit likelihood.

## Core Insights

### The discriminator supplies a moving density ratio

The generator $G(z;\theta_g)$ maps noise $z\sim p_z$ to a sample, defining an implicit distribution $p_g$. The discriminator $D(x;\theta_d)$ estimates the probability that $x$ came from the data distribution. Their original objective is

$$
\min_G\max_D\;V(D,G)=\mathbb E_{x\sim p_{data}}[\log D(x)]+\mathbb E_{z\sim p_z}[\log(1-D(G(z)))].
$$

For a fixed generator, the optimal discriminator is

$$
D_G^*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}.
$$

Substituting it into the game gives $C(G)=-\log 4+2\,\mathrm{JSD}(p_{data}\|p_g)$. The ideal equilibrium is therefore precise: $p_g=p_{data}$ and $D(x)=1/2$. The discriminator is not merely a critic with an arbitrary reward; in the idealized derivation it is the data-versus-generator posterior, whose odds recover the density ratio $p_{data}(x)/p_g(x)$.

![Source Figure 1 from Generative Adversarial Nets: the generator and discriminator approach distributional equilibrium](/assets/images/generative-adversarial-networks-source-figure-1.png)
*Fig 1: The four panels move from separated data and generator densities to an equilibrium where the discriminator is flat and $p_g$ matches $p_{data}$; arrows show how the latent map changes the generated density. | source: [Generative Adversarial Nets, Figure 1](https://arxiv.org/abs/1406.2661)*

The panel sequence also shows what the theorem assumes away. It treats the discriminator as reaching its optimum while the generator changes slowly. The actual algorithm alternates $k$ discriminator updates with one generator update; the paper uses $k=1$, minibatches, and momentum because solving the inner game to completion would be expensive and could overfit a finite dataset.

### The practical generator objective is a gradient choice

The minimax generator term $\log(1-D(G(z)))$ can saturate when an untrained generator produces samples that the discriminator rejects confidently. The paper therefore recommends maximizing $\log D(G(z))$ for the generator in practice. This non-saturating objective has the same fixed point but gives a stronger early gradient. That small change separates the ideal game from the optimization rule used to reach it.

The original models are multilayer perceptrons. Generators use rectifier and sigmoid activations; discriminators use maxout units and dropout. Noise enters only the bottom layer of the generator. This is enough to show the mechanism, but it leaves several engineering decisions implicit in later GAN work: how to stabilize a discriminator, how to keep rare modes alive, and how to evaluate an implicit distribution without $p_g(x)$.

### The experiments test sampling and a weak likelihood proxy

![Source Figure 2 from Generative Adversarial Nets: random samples and nearest training examples](/assets/images/generative-adversarial-networks-source-figure-2.png)
*Fig 2: The panels show random MNIST, face, and CIFAR-10 samples; the yellow rightmost column is the nearest training example for each neighboring sample, a visual check against simple memorization. | source: [Generative Adversarial Nets, Figure 2](https://arxiv.org/abs/1406.2661)*

The paper evaluates MNIST, the Toronto Face Database, and CIFAR-10. For a quantitative signal it fits a Gaussian Parzen window to generated samples and reports test-set log-likelihood estimates. On MNIST, adversarial nets report $225\pm2$, compared with $214\pm1.1$ for Deep GSN, $138\pm2$ for DBN, and $121\pm1.6$ for stacked CAE. On TFD, the result is $2057\pm26$, below stacked CAE’s $2110\pm50$ but above DBN’s $1909\pm66$ and Deep GSN’s $1890\pm29$.

Those numbers require careful reading. The MNIST comparison uses real-valued rather than binary images. On TFD, the bandwidth is cross-validated per fold and the reported error is across folds. The authors explicitly warn that Parzen estimates have high variance and behave poorly in high dimensions. The table is evidence that an implicit model can produce competitive samples under a then-available proxy, not a general likelihood victory.

![Source Figure 3 from Generative Adversarial Nets: linear interpolation in the generator’s latent space](/assets/images/generative-adversarial-networks-source-figure-3.png)
*Fig 3: Linear paths through latent space produce smooth digit changes. This tests continuity of the learned mapping; smooth interpolation alone does not establish mode coverage or rule out memorization. | source: [Generative Adversarial Nets, Figure 3](https://arxiv.org/abs/1406.2661)*

The samples are direct forward passes and do not depend on Markov-chain mixing. That is a meaningful systems advantage over contemporaneous models. The same property does not guarantee coverage: the generator can map many noise values to the same output, the paper’s “Helvetica scenario,” while the discriminator remains poorly synchronized.

### The theorem leaves the difficult part in the optimizer

The nonparametric proof does not cover finite MLP parameterizations, imperfect discriminator updates, or the geometry of the generator’s parameter space. The paper itself notes multiple critical points in that space and offers no guarantee that alternating updates reach the global equilibrium. It also does not define an explicit tractable density, so exact likelihood and calibrated uncertainty remain outside the training objective.

That boundary explains the later research line. Wasserstein objectives, gradient penalties, architectural constraints, and precision/recall diagnostics all address pieces of the same gap. They preserve the adversarial idea while changing the metric, the optimization geometry, or the evidence used to detect missing modes.

## High-Level Takeaways

- GANs make a density-ratio classifier the training interface for an implicit generator, removing MCMC and explicit likelihood from the sampling path.
- The global $p_g=p_{data}$ result depends on an optimal discriminator and sufficient capacity; finite alternating updates create the actual research problem.
- The original Parzen scores and nearest-example panels support feasibility, while mode coverage and high-dimensional likelihood remain unmeasured or weakly measured.
- Direct sampling avoids a Markov chain, but a cheap forward pass can still miss modes. The theorem concerns matching distributions; attractive individual samples are only part of the evidence.
