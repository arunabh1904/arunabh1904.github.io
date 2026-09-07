---
title: Improved Denoising Diffusion Probabilistic Models
date: '2021-02-18T00:00:00.000Z'
section: paper-shorts
postSlug: improved-denoising-diffusion-probabilistic-models
legacyPath: /paper shorts/2021/06/01/improved-denoising-diffusion-probabilistic-models.html
tags:
  - Other
field: 'Generative Modeling'
summary: "2021 – Improved Denoising Diffusion Probabilistic Models"
---
## 2021 – Improved Denoising Diffusion Probabilistic Models (Improved DDPM)

**arXiv:** [2102.09672](https://arxiv.org/abs/2102.09672)<br>
**GitHub:** [openai/improved-diffusion](https://github.com/openai/improved-diffusion)<br>
**Conference:** ICML 2021

## Summary

> Improved DDPM keeps the DDPM reverse chain and targets the two places where its practical story was weakest: likelihood and sampling cost. It learns reverse variances, combines the noise-prediction loss with a lightly weighted variational term, and replaces the linear schedule with a cosine schedule that preserves useful signal longer at 32×32 and 64×64 resolutions. The paper reports better likelihood/FID trade-offs and near-optimal sample quality with about 100 sampling steps for fully trained models. The evidence is a carefully separated set of image-scale ablations; sequential denoising remains the serving cost.

## Core Insights

### Reverse variance is the interface between likelihood and fast sampling

Original DDPM fixes the reverse variance, even though the variational bound is sensitive to the earliest reverse transitions. Improved DDPM lets the network output an interpolation in log variance:

$$
\Sigma_\theta(x_t,t)=\exp\big(v\log\beta_t+(1-v)\log\tilde\beta_t\big),
$$

where $\beta_t$ and $\tilde\beta_t$ are the two posterior-variance endpoints. The network is not asked to predict an unconstrained variance from scratch; it chooses a point in a narrow, source-motivated range. The paper observes that the two endpoints are nearly equal for much of the chain, but differ where the model is reconstructing imperceptible detail. Those early terms contribute disproportionately to the variational bound.

![Source Figure 1 from Improved DDPM: posterior-variance endpoints across diffusion lengths](/assets/images/improved-denoising-diffusion-probabilistic-models-source-figure-1.webp)
*Fig 1: The ratio between the posterior variance and the forward variance stays close to one for most timesteps, clarifying why learned variance mainly matters near the low-noise end of the chain. | source: [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)*

To keep that likelihood signal from destabilizing the sample-quality objective, the paper uses

$$
L_{hybrid}=L_{simple}+\lambda L_{vlb},\qquad \lambda=0.001,
$$

and stops the gradient from the variational term through $\mu_\theta$. In effect, $L_{simple}$ remains responsible for the reverse mean while $L_{vlb}$ can teach the variance. This is why the correct description is a hybrid objective with a stop-gradient design, rather than a generic KL-plus-MSE loss.

### The cosine schedule changes where information disappears

For low-resolution images, the original linear schedule drives $\bar\alpha_t$ toward zero too early. Many late forward states are already nearly pure noise, so reverse steps in that region contribute little useful learning. Improved DDPM defines

$$
\bar\alpha_t=\frac{f(t)}{f(0)},\qquad f(t)=\cos^2\left(\frac{t/T+s}{1+s}\frac{\pi}{2}\right),\qquad s=0.008,
$$

then derives $\beta_t=1-\bar\alpha_t/\bar\alpha_{t-1}$ and clips $\beta_t$ at 0.999.

![Source Figure 3 from Improved DDPM: latent states under linear and cosine schedules](/assets/images/improved-ddpm-paper-figure-3-noise-schedule.png)
*Fig 2: At equally spaced forward times, the linear schedule has become almost pure noise during its last quarter, while the cosine schedule retains recognizable structure for longer. | source: [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)*

![Source Figure 5 from Improved DDPM: cumulative signal under the two schedules](/assets/images/improved-denoising-diffusion-probabilistic-models-source-figure-5.webp)
*Fig 3: The cosine schedule makes $\bar\alpha_t$ fall gradually through the middle of diffusion instead of discarding signal rapidly near the start; the change is a redistribution of denoising difficulty. | source: [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)*

The paper also reduces gradient noise when optimizing the full variational objective by sampling timesteps with probability proportional to $\sqrt{\mathbb E[L_t^2]}$, estimated from a history of the previous ten loss terms. That resampling matters for likelihood training, but the authors report that it is not helpful for the less noisy hybrid objective.

### The ablations separate likelihood from perceptual quality

| Dataset and recipe | NLL (bits/dim) | FID |
| --- | ---: | ---: |
| ImageNet-64, 200K, 1K linear $L_{simple}$ | 3.99 | 32.5 |
| ImageNet-64, 200K, 4K cosine $L_{hybrid}$ | 3.62 | 28.0 |
| ImageNet-64, 1.5M, 4K cosine $L_{hybrid}$ | 3.57 | 19.2 |
| ImageNet-64, 1.5M, 4K cosine $L_{vlb}$ | 3.53 | 40.1 |
| CIFAR-10, 500K, 4K cosine $L_{hybrid}$ | 3.17 | 3.19 |
| CIFAR-10, 500K, 4K cosine $L_{vlb}$ | 2.94 | 11.47 |

The $L_{vlb}$ rows make the trade-off visible: direct likelihood improves bits per dimension but can hurt FID sharply. The hybrid objective is the paper’s compromise, not a claim that one scalar dominates both metrics. On class-conditional ImageNet-64 with 250 sampling steps, the large Improved Diffusion model reports FID 2.92, precision 0.82, and recall 0.71; BigGAN-deep reports FID 4.06, precision 0.86, and recall 0.59. The lower recall of BigGAN in this matched comparison is the evidence for a coverage advantage, while the precision difference shows the trade is not one-sided.

### Learned variance makes shorter chains viable

All models in the speed study are trained with 4,000 diffusion steps. The learned-variance hybrid model retains near-optimal FID with 100 sampling steps, whereas fixed-variance $L_{simple}$ models degrade more when the chain is strided. This is the source of the often-repeated “10× fewer steps” result: it compares a 4,000-step training schedule to a shorter inference subsequence, not a single-step generator.

The scaling experiment varies ImageNet-64 U-Net capacity from 30M to 270M parameters. FID follows an approximately linear trend on a log-log compute plot, while NLL fits a power law less cleanly. That difference reinforces the central boundary: visual quality and likelihood can improve with the same compute, but they are not interchangeable objectives.

## High-Level Takeaways

- Learned reverse variance gives the sampler freedom to absorb uncertainty without asking the network to predict an arbitrary scale; the hybrid loss keeps that freedom from overwhelming mean prediction.
- The cosine schedule is useful because low-resolution images need signal spread across the chain, not because cosine is a universal replacement for every noise schedule.
- Direct $L_{vlb}$ improves likelihood in the reported ablations while degrading FID, so the paper’s practical choice is an explicit quality/likelihood compromise.
- The matched speed result is near-optimal FID at about 100 sampling steps after 4,000-step training; high-resolution serving still pays for sequential network evaluations.
