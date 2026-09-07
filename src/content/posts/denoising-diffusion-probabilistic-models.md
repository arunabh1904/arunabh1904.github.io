---
title: Denoising Diffusion Probabilistic Models
date: '2020-06-19T00:00:00.000Z'
section: paper-shorts
postSlug: denoising-diffusion-probabilistic-models
legacyPath: /paper shorts/2020/06/01/denoising-diffusion-probabilistic-models.html
tags:
  - Other
field: 'Generative Modeling'
summary: "2020 – Denoising Diffusion Probabilistic Models"
---
## 2020 – Denoising Diffusion Probabilistic Models (DDPM)

**arXiv:** [2006.11239](https://arxiv.org/abs/2006.11239)<br>
**GitHub:** [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)<br>
**OpenAI PyTorch port:** [openai/improved-diffusion](https://github.com/openai/improved-diffusion)<br>
**Conference:** NeurIPS 2020

## Summary

> DDPM turns generation into the reverse of a fixed Gaussian corruption chain. A forward process makes each image noisy at a known timestep; a network predicts the noise needed to undo it. The variational bound gives a likelihood interpretation, while the simplified loss connects to denoising score matching. On CIFAR-10, the paper reports FID 3.17 and Inception Score 9.46 with 1,000 reverse steps, plus progressive generation and compression behavior. The recipe is stable and high quality, but the original sampler spends one network evaluation on every denoising step.

## Core Insights

### The forward chain gives the reverse model a supervised target at every noise level

DDPM defines a fixed Markov process

$$
q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1}),\qquad q(x_t\mid x_{t-1})=\mathcal N\big(\sqrt{1-\beta_t}\,x_{t-1},\beta_t I\big).
$$

Writing $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$ gives a one-shot training sample:

$$
q(x_t\mid x_0)=\mathcal N\big(\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)I\big).
$$

The model does not have to simulate all earlier corruptions to train on timestep $t$. It can draw $t$, draw $\epsilon\sim\mathcal N(0,I)$, and construct $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$.

The learned reverse chain starts from $p(x_T)=\mathcal N(0,I)$ and predicts $p_\theta(x_{t-1}\mid x_t)$. With fixed reverse variance, the network predicts the noise $\epsilon_\theta(x_t,t)$ and uses

$$
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right).
$$

![Source Figure 2 from DDPM: forward noising and learned reverse denoising](/assets/images/ddpm-paper-figure-2-graphical-model.png)
*Fig 1: The forward chain moves $x_0$ toward Gaussian noise, while the learned reverse transitions move from $x_T$ back toward a data sample; the two directions share the same timestep structure but only the reverse path is learned. | source: [Denoising Diffusion Probabilistic Models, Figure 2](https://arxiv.org/abs/2006.11239)*

This graphical model is the useful mental split. Corruption is known, cheap, and differentiable in closed form. Learning concentrates on the conditional reverse steps, where the target noise is available because the corruption process generated it.

### Noise prediction is a better optimization target than a raw mean

The exact variational objective decomposes into endpoint and per-timestep KL terms. DDPM also evaluates a simplified objective:

$$
L_{simple}=\mathbb E_{x_0,\epsilon,t}\left[\left\|\epsilon-\epsilon_\theta\left(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t\right)\right\|^2\right],
$$

with $t$ sampled uniformly. The model is therefore trained on a family of denoising problems, from almost-clean images to near-pure noise, rather than on a single reconstruction task. The paper connects this objective to denoising score matching and Langevin dynamics, which explains why the same prediction field can support both sampling and a probabilistic interpretation.

The main CIFAR-10 model uses $T=1000$, a linear $\beta_t$ schedule from $10^{-4}$ to $0.02$, and a U-Net with group normalization, sinusoidal timestep embeddings, and self-attention at 16×16 resolution. The reported CIFAR model has 35.7M parameters. The source describes larger 114M-parameter models for high-resolution datasets, so the comparison is not a claim that one small network solves every image scale.

### The reported samples are strong, but the protocol matters

![Source Figure 6 from DDPM: progressive CIFAR-10 generation across reverse time](/assets/images/denoising-diffusion-probabilistic-models-source-figure-6.webp)
*Fig 2: Each row shows the model's estimate of the clean CIFAR-10 image at successive reverse times; broad structure appears before the final texture and edges. | source: [Denoising Diffusion Probabilistic Models, Figure 6](https://arxiv.org/abs/2006.11239)*

The progressive panel is more informative than a final sample grid because it shows where the chain spends its representational work. Early reverse states recover large-scale shape and color; later states add local detail. That ordering is also why the chain can be read as progressive lossy decompression: a noisy intermediate preserves some coarse information and discards fine detail.

| Evaluation | Reported result | Protocol boundary |
| --- | ---: | --- |
| CIFAR-10 | FID 3.17; IS 9.46 | 1,000-step $L_{simple}$ model |
| CIFAR-10 test reference | FID 5.24 | The 3.17 FID is against the training set |
| LSUN Church 256² | FID 7.89 | Unconditional samples |
| LSUN Bedroom 256² | FID 4.90 | Unconditional samples |
| CIFAR-10 compression | 1.78 bits/dim; RMSE 0.95 [0,255] pixel units | Best-quality progressive-compression model |

![Source Figure 1 from DDPM: four unconditional CelebA-HQ face samples (retained crop)](/assets/images/denoising-diffusion-probabilistic-models-source-figure-1.webp)
*Fig 3: This retained crop shows four unconditional CelebA-HQ faces from source Figure 1; the original source figure also contains CIFAR-10 samples, which are omitted here for legibility. | source: [Denoising Diffusion Probabilistic Models, Figure 1](https://arxiv.org/abs/2006.11239)*

The training-set versus test-set FID distinction is material. It prevents the headline 3.17 from being read as a universal estimate of generalization, and it makes the later sampler and evaluation improvements easier to compare honestly. The original paper’s strongest contribution is the combination of a simple corruption process, a stable noise target, and a reverse chain that produced competitive samples without adversarial training.

### The deployment boundary is sequential compute

Sampling requires the model to evaluate a reverse transition for each selected timestep. The original 1,000-step chain is easy to state and parallelizes over a batch, but it does not parallelize over time. Later work learns reverse variances, changes schedules, or uses accelerated samplers; those methods address the cost of the chain rather than invalidate the forward/reverse construction.

## High-Level Takeaways

- DDPM supplies a known corruption path and a noise target at every timestep, turning generation into a sequence of supervised denoising problems.
- The simplified noise-prediction loss gives strong samples while the variational decomposition keeps a likelihood and compression interpretation available.
- CIFAR-10 FID 3.17 is a training-set reference result; the reported test FID 5.24 and the 1,000 reverse steps define the practical boundary.
- Training can jump directly to any noise level, while the original sampler must visit every reverse step. That asymmetry is why sampling speed becomes a separate research problem after the denoising objective works.
