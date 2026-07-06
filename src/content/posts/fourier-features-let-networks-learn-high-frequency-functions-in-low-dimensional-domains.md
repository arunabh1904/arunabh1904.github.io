---
title: 'Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains'
date: '2020-06-18T04:00:00.000Z'
section: paper-shorts
postSlug: fourier-features-let-networks-learn-high-frequency-functions-in-low-dimensional-domains
legacyPath: /paper shorts/2020/06/18/fourier-features-let-networks-learn-high-frequency-functions-in-low-dimensional-domains.html
tags:
  - Other
field: BEV
summary: Fourier features turn low-dimensional coordinates into sinusoidal embeddings so MLPs can fit high-frequency geometry, images, and scene signals.
---
## 2020 - Fourier Features

**arXiv:** [2006.10739](https://arxiv.org/abs/2006.10739)

**Project:** [Fourier Feature Networks](https://bmild.github.io/fourfeat/)

**Code:** [tancik/fourier-feature-networks](https://github.com/tancik/fourier-feature-networks)

**Plain-language summary:** A plain MLP has a spectral bias: it learns smooth, low-frequency functions much more easily than sharp boundaries or fine texture. This paper fixes that failure mode by mapping each coordinate through sinusoidal Fourier features before the MLP sees it.

That small input change matters for BEV and mapping work because many spatial tasks are low-dimensional coordinate regression problems with high-frequency structure. Lane boundaries, occupancy edges, radiance fields, and implicit maps all ask a network to represent sharp geometry from coordinates.

## Paper Insights

The paper studies why coordinate MLPs struggle when the target function contains high-frequency detail. The method samples or chooses Fourier bases, maps an input coordinate through sine and cosine features, and feeds the expanded representation into a standard MLP. Using neural tangent kernel analysis, the authors show that this transformation changes the effective kernel into a stationary kernel whose bandwidth can be tuned by the Fourier feature scale.

The empirical evidence comes from image regression and low-dimensional vision and graphics tasks. The important caveat is that the frequency scale is a real modeling choice. Too little bandwidth leaves the MLP smooth; too much bandwidth can overfit or make optimization brittle.

![Figure 2 from Fourier Features showing how Fourier mappings change the MLP neural tangent kernel](/assets/images/fourier-features-let-networks-learn-high-frequency-functions-in-low-dimensional-domains-paper-figure.png)
_Figure 2 shows why the input mapping matters: Fourier features reshape the effective NTK into a frequency-aware kernel. From the [Fourier Features paper](https://arxiv.org/abs/2006.10739), via ar5iv._

**What to look at:**
- The contribution is an input representation, not a new network family.
- Fourier features let a coordinate network expose high frequencies early instead of asking hidden layers to discover them slowly.
- The bandwidth parameter controls the smoothness/detail tradeoff.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Core method | Sinusoidal coordinate mapping before an MLP | A small representation change unlocks high-frequency regression. |
| Theory | NTK analysis of the transformed MLP | Explains the frequency bias instead of only showing examples. |
| Tasks | Low-dimensional vision and graphics regression | Matches the coordinate-heavy structure of implicit scene and map models. |

**Compact result slice:**

| Input mapping | 2D natural image PSNR | 3D shape IoU |
| ------------- | --------------------- | ------------ |
| No mapping | 19.32 | 0.864 |
| Gaussian Fourier features | 25.57 | 0.973 |

**Why it mattered:** Fourier features became one of the standard ways to make coordinate networks useful for detailed spatial signals.

**Take-home message:** If an MLP is asked to learn geometry from raw coordinates, give it a frequency basis first; otherwise the model starts with the wrong smoothness prior.
