---
title: Improved Denoising Diffusion Probabilistic Models
date: '2021-06-01T04:00:00.000Z'
section: paper-shorts
postSlug: improved-denoising-diffusion-probabilistic-models
legacyPath: >-
  /paper
  shorts/2021/06/01/improved-denoising-diffusion-probabilistic-models.html
tags:
  - Other
field: Generative Models
summary: Improved DDPM tightened diffusion likelihoods and made sampling faster by learning reverse-process variance.
---
## 2021 – Improved Denoising Diffusion Probabilistic Models (ID DPM)

**arXiv:** [2102.09672](https://arxiv.org/abs/2102.09672)

**GitHub:** [openai/improved-diffusion](https://github.com/openai/improved-diffusion)

**Conference:** ICML 2021

## Paper Insights

Improved DDPM keeps the diffusion framework but fixes practical weaknesses in likelihood and sampling speed. The paper learns reverse-process variances instead of using a fixed variance schedule, studies hybrid objectives that balance sample quality and likelihood, and introduces a cosine noise schedule that allocates denoising difficulty more smoothly. It also shows that fewer sampling steps can preserve quality better than expected. The evidence includes ImageNet and other image-generation experiments comparing FID, negative log-likelihood, and precision/recall. The caveat is that diffusion still requires sequential denoising, so faster sampling is an improvement rather than a full solution. The paper is important because it turned DDPM from a promising sampler into a more scalable generative modeling recipe.

![Figure 3 from Improved DDPM: linear and cosine noise schedules preserve signal at different rates](/assets/images/improved-ddpm-paper-figure-3-noise-schedule.png)
_Figure 3 from the [Improved DDPM paper](https://arxiv.org/abs/2102.09672), via ar5iv._

**Plain-language summary:** Nichol and Dhariwal made DDPMs faster and stronger without changing the basic denoising story. They learn the reverse-process variance $\Sigma_\theta$ instead of keeping it fixed, optimize a hybrid objective that mixes ELBO terms with the simple noise-prediction loss, and use a cosine noise schedule with importance-weighted terms for stabler gradients.

Those changes let the sampler take larger steps through noise space. Denoising drops from 1000 steps to roughly 50-250 with little FID loss, while log-likelihoods reach parity with autoregressive models on ImageNet-64. The paper also makes a useful empirical claim: bigger UNets and more compute improve bits-per-dim and FID in a predictable scaling-law-like way.

**Key results**

| Dataset / setting | Steps | FID ↓ | IS ↑ | NLL (bits/dim) |
| ----------------- | ----- | ---- | ---- | --------------- |
| CIFAR-10 32² | 250 | 2.92 | 9.89 | 3.40 |
| CIFAR-10 32² (orig. DDPM) | 1000 | 3.17 | 9.46 | 3.69 |
| ImageNet 64² (class-cond.) | 250 | 2.92 | — | 3.57 |
| ImageNet 64² (BigGAN-deep, 100M params) | 1 | 4.06 | — | — |

The key result is speed without giving up quality: ID DPM matches or beats GANs while requiring about ten times fewer network evaluations than vanilla DDPM.

**Tiny PyTorch snippet – hybrid loss and learned variance**
```python
def iddpm_loss(model, x0, timesteps, betas, logvar_schedule):
    """Hybrid loss from Nichol & Dhariwal (2021)."""
    noise = torch.randn_like(x0)
    sqrt_alphas_cum = torch.sqrt(torch.cumprod(1 - betas, 0))
    sqrt_one_minus = torch.sqrt(1 - torch.cumprod(1 - betas, 0))
    x_t = sqrt_alphas_cum[timesteps] * x0 + sqrt_one_minus[timesteps] * noise

    eps_hat, logvar_hat = model(x_t, timesteps)

    mse = F.mse_loss(eps_hat, noise)
    kl = 0.5 * (torch.exp(-logvar_hat) * (noise ** 2) + logvar_hat).mean()
    return 0.5 * (mse + kl)
```
Switching to the cosine $\beta_t$ schedule from the appendix further sharpens FID at low step counts.

**Critiques:** The upgrades are attractive because they are almost drop-in: learn variance, adjust the objective, improve the schedule. They became part of the practical diffusion toolbox used by later systems. Sampling still needs dozens of UNet passes, large models remain memory-heavy, and classifier guidance can introduce bias, which later classifier-free methods address more cleanly.

**Take-home message:** ID DPM turned diffusion from a slow curiosity into a practical generator, paving the way for fast samplers and classifier-free guidance.
