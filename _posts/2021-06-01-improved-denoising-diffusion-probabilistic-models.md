---
layout: content
title: "Improved Denoising Diffusion Probabilistic Models"
date: 2021-06-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Generative Models
---

## 2021 – Improved Denoising Diffusion Probabilistic Models (ID DPM)

**arXiv:** [2102.09672](https://arxiv.org/abs/2102.09672)

**GitHub:** [openai/improved-diffusion](https://github.com/openai/improved-diffusion)

**Conference:** ICML 2021

**Plain-language summary**
Nichol and Dhariwal revisit DDPMs with three upgrades that lift both likelihood and sample quality while
reducing sampling cost.

- Learn the reverse-process variance \(\Sigma_\theta\) instead of keeping it fixed.
- Optimise a hybrid objective: half ELBO, half simple noise-prediction loss.
- Use a cosine noise schedule with importance-weighted loss terms for stable gradients.

These tweaks cut denoising from 1000 steps to around 50–250 with minimal FID drop and push
log-likelihoods to parity with autoregressive models on ImageNet-64.

**Novel insights**
- Variance learning enables larger strides through noise space for faster sampling.
- The hybrid loss reduces gradient noise and improves NLL and FID.
- Diffusion models show higher recall than GANs at similar fidelity.
- Scaling laws hold: bigger UNets and more compute steadily improve bits-per-dim and FID.

**Key results**

| Dataset / setting | Steps | FID ↓ | IS ↑ | NLL (bits/dim) |
| ----------------- | ----- | ---- | ---- | --------------- |
| CIFAR-10 32² | 250 | 2.92 | 9.89 | 3.40 |
| CIFAR-10 32² (orig. DDPM) | 1000 | 3.17 | 9.46 | 3.69 |
| ImageNet 64² (class-cond.) | 250 | 2.92 | — | 3.57 |
| ImageNet 64² (BigGAN-deep, 100M params) | 1 | 4.06 | — | — |

Take-away: ID DPM matches or beats GANs while requiring ten times fewer network evaluations than vanilla
DDPM.

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
Switching to the cosine \(\beta_t\) schedule from the appendix further sharpens FID at low step counts.

**Critiques**
- **What shines:** Simple, drop-in upgrades that became the default for Stable Diffusion and DALLE‑2.
- **Caveats:** Sampling still needs over 50 UNet passes and large models remain memory heavy.
- Classifier guidance can introduce bias; classifier‑free approaches address this later.

ID DPM turned diffusion from a slow curiosity into a practical generator, paving the way for fast samplers and
classifier-free guidance.

