---
layout: content
title: "Denoising Diffusion Probabilistic Models"
date: 2020-06-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Generative Models
---

## 2020 – Denoising Diffusion Probabilistic Models (DDPM)

**arXiv:** [2006.11239](https://arxiv.org/abs/2006.11239)

**Improved DDPM:** [2102.09672](https://arxiv.org/abs/2102.09672)

**GitHub:** [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)

**OpenAI PyTorch port:** [openai/improved-diffusion](https://github.com/openai/improved-diffusion)

**Conference:** NeurIPS 2020

**Plain-language abstract**  
DDPMs define a forward Markov chain that gradually corrupts real data into near-isotropic Gaussian noise and a
learned reverse chain (parameterised by a UNet) that denoises step-by-step. Training minimises a weighted MSE
between the network’s noise prediction ε̂ and the actual noise ε. With 1000 denoising steps, DDPM achieved FID 3.17
and IS 9.46 on CIFAR-10, matching StyleGAN-v2 without adversarial training.

**Deep-dive insights**
- Noise-conditioning as score matching links DDPMs to score-based SDE models.
- Learning the reverse-process variance (“Improved DDPM”) cuts sampling to around 50 steps.
- Classifier or classifier-free guidance steers generation for class or text conditioning.
- Optimising a tractable ELBO yields good likelihoods alongside crisp samples.

**Benchmarks**

| Dataset & Steps | FID ↓ | IS ↑ |
| --------------- | ----- | ---- |
| CIFAR-10, 1000 steps | 3.17 | 9.46 |
| CIFAR-10, 250 steps (Improved) | 2.92 | 9.89 |
| LSUN-Bedroom 256² | 11.9 | — |
| ImageNet 128², 250 steps | 12.26 | — |

**Minimal PyTorch-style snippet – loss and sampler**

```python
def q_sample(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    return (sqrt_alphas_cumprod[t] * x0 +
            sqrt_one_minus_alphas_cumprod[t] * noise), noise

def p_losses(model, x0, t, noise, sqrt_alphas_cumprod,
             sqrt_one_minus_alphas_cumprod):
    x_t, noise = q_sample(x0, t, sqrt_alphas_cumprod,
                          sqrt_one_minus_alphas_cumprod, noise)
    eps_hat = model(x_t, t)
    return F.mse_loss(eps_hat, noise)
```

```python
@torch.no_grad()
def p_sample_loop(model, shape, betas, sqrt_recip_alphas,
                  sqrt_recipm1_alphas, posterior_variance):
    x_t = torch.randn(shape).to(device)
    for t in reversed(range(len(betas))):
        eps_hat = model(x_t, t)
        x0_hat = (x_t - sqrt_recipm1_alphas[t] * eps_hat) * sqrt_recip_alphas[t]
        noise = torch.randn_like(x_t) if t else 0
        x_t = x0_hat + torch.sqrt(posterior_variance[t]) * noise
    return x_t
```

**Critiques & open questions**
- **Strengths:** Stable training, no mode collapse, tractable likelihood.
- **Limitations:** Sampling requires many UNet passes; training is compute heavy.

**Take-home message**  
DDPM reframed generative modelling around gradual noise erosion and learned denoising.
Variance learning and improved samplers brought diffusion models from curiosity to the engines powering
modern text-to-image systems.

