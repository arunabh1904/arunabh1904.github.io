---
title: Denoising Diffusion Probabilistic Models
date: '2020-06-01T04:00:00.000Z'
section: paper-shorts
postSlug: denoising-diffusion-probabilistic-models
legacyPath: /paper shorts/2020/06/01/denoising-diffusion-probabilistic-models.html
tags:
  - Other
field: Generative Models
summary: DDPMs turned generation into learned denoising, trading slow sampling for stable training and strong image quality.
---
## 2020 – Denoising Diffusion Probabilistic Models (DDPM)

**arXiv:** [2006.11239](https://arxiv.org/abs/2006.11239)

**Improved DDPM:** [2102.09672](https://arxiv.org/abs/2102.09672)

**GitHub:** [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)

**OpenAI PyTorch port:** [openai/improved-diffusion](https://github.com/openai/improved-diffusion)

**Conference:** NeurIPS 2020

## Paper Insights

DDPM frames generation as learning to reverse a gradual noising process. The forward chain adds Gaussian noise until data becomes nearly pure noise; the reverse model learns denoising transitions that reconstruct samples step by step. The training objective can be written as a variational bound, but the practical loss predicts noise and connects to denoising score matching. The paper's evidence is high-quality image synthesis on CIFAR-10 and LSUN, with strong FID and a useful progressive decompression interpretation. The main cost is sampling: generation requires many sequential denoising steps, making early DDPMs slower than GANs. The lasting idea is that likelihood-based latent-variable models can produce strong samples when trained as iterative denoisers.

![Figure 2 from DDPM: the directed graphical model for forward noising and learned reverse denoising](/assets/images/ddpm-paper-figure-2-graphical-model.png)
_Figure 2 from the [DDPM paper](https://arxiv.org/abs/2006.11239), via ar5iv._

**Plain-language abstract:** DDPMs generate data by learning to reverse noise. The forward process gradually corrupts real examples into near-isotropic Gaussian noise. The learned reverse process, usually parameterized by a UNet, removes that noise step by step. Training asks the model to predict the noise $\hat{\epsilon}$ that was added at each timestep, then minimizes a weighted MSE against the true noise $\epsilon$.

The payoff is stability. With 1000 denoising steps, DDPM reached FID 3.17 and IS 9.46 on CIFAR-10, matching StyleGAN-v2 without adversarial training. The framework also connects noise-conditioned denoising to score matching and score-based SDE models, giving diffusion a strong probabilistic interpretation rather than just a good sample generator.

**Why it mattered:** Diffusion models made high-quality generation feel less fragile than GAN training. Later improvements, including learned reverse-process variance, faster samplers, and classifier-free guidance, turned the original slow denoising loop into the foundation for modern text-to-image systems.

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

**Critiques & open questions:** DDPMs train stably, avoid mode collapse, and provide a tractable likelihood story. The cost is sampling speed. A naive sampler needs many UNet passes, and training remains compute-heavy even when optimization is well behaved.

**Take-home message:** DDPM reframed generative modelling around gradual noise erosion and learned denoising. Once variance learning and faster samplers arrived, diffusion moved from elegant curiosity to practical engine.
