---
title: 'Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains'
date: '2020-06-18T00:00:00.000Z'
section: paper-shorts
postSlug: fourier-features-let-networks-learn-high-frequency-functions-in-low-dimensional-domains
legacyPath: /paper shorts/2020/06/18/fourier-features-let-networks-learn-high-frequency-functions-in-low-dimensional-domains.html
tags:
  - Other
field: 'Vision Foundations'
summary: "2020 – Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
---
## 2020 – Fourier Features

**arXiv:** [2006.10739](https://arxiv.org/abs/2006.10739)

**Project:** [Fourier Feature Networks](https://bmild.github.io/fourfeat/)

**Code:** [tancik/fourier-feature-networks](https://github.com/tancik/fourier-feature-networks)

## Summary

> Coordinate MLPs learn smooth structure first, which makes sharp image detail, shape boundaries, and radiance variation slow to fit. Fourier features change the coordinates before the MLP sees them: each input is projected onto sinusoidal bases, exposing a tunable range of frequencies. The paper explains this through the neural tangent kernel and shows gains on direct and indirect low-dimensional regression, including image, shape, CT, MRI, and NeRF-style tasks. The useful rule is a bandwidth choice, not a universal encoding: low scales underfit and high scales overfit.

## Core Insights

### Mechanism

For a coordinate v, the mapping is gamma(v) = [cos(2π Bv), sin(2π Bv)], with rows of B sampled from a frequency distribution. A plain MLP's NTK is a dot-product kernel that is not translation-invariant over a dense Euclidean coordinate domain. The sinusoidal mapping makes the composed kernel a function of coordinate differences, so it is stationary. The standard deviation of the sampled frequencies then controls how much high-frequency power the effective kernel carries.

![The paper compares the NTK of a raw coordinate MLP with the stationary, tunable kernels produced by Fourier mappings.](/assets/images/fourier-features-let-networks-learn-high-frequency-functions-in-low-dimensional-domains-paper-figure.png)
*Fig 1: A Fourier mapping makes the composed NTK more stationary and widens its spectrum as the frequency schedule changes. | source: [Fourier Features, Figure 2](https://arxiv.org/abs/2006.10739)*

The distinction between scale and distribution shape is central. In a controlled one-dimensional task, Gaussian, uniform, log-uniform, and Laplacian frequency samples trace nearly the same error curve when compared at the same empirical standard deviation. A low standard deviation leaves the kernel narrow and the reconstruction smooth. A high standard deviation supplies detail but can fit frequencies absent from the held-out signal.

![Different random frequency distributions follow a shared underfitting-to-overfitting curve when plotted against sampled-frequency scale.](/assets/images/fourier-features-let-networks-learn-high-frequency-functions-in-low-dimensional-domains-source-figure-4.webp)
*Fig 2: Sparse random Fourier features match dense features over a useful range, with underfitting at low frequency scale and overfitting at high scale. | source: [Fourier Features, Figure 4](https://arxiv.org/abs/2006.10739)*

### Evidence

The experiments use four-layer, 256-channel ReLU MLPs for most tasks and 256 frequencies; the shape experiment uses an eight-layer network. For 2D image regression, 512×512 images are split into a 256×256 training grid and an offset 256×256 test grid. Scales are tuned on held-out images, then evaluated on the remaining images.

| Mapping | Natural image PSNR | Text image PSNR | 3D shape boundary IoU |
| --- | ---: | ---: | ---: |
| No mapping | 19.32 ± 2.48 | 18.40 ± 2.23 | 0.864 ± 0.014 |
| Basic Fourier | 21.71 ± 2.71 | 20.48 ± 1.96 | 0.892 ± 0.017 |
| Positional encoding | 24.95 ± 3.72 | 27.57 ± 3.07 | 0.960 ± 0.011 |
| Gaussian Fourier | 25.57 ± 4.19 | 30.47 ± 2.11 | 0.973 ± 0.010 |

The same pattern extends to indirect supervision. In Table 1, Gaussian features reach 28.33 PSNR on 2D CT, 19.88 on 3D MRI, and 25.48 on the simplified NeRF task, compared with 16.75, 15.44, and 22.41 with no mapping. Those tasks supervise the network through integral projections, Fourier coefficients, or volume rendering rather than a label at each coordinate, so the improvement is not just a pixel-interpolation trick.

![Fourier features improve coordinate MLPs on both direct and forward-model-supervised regression tasks.](/assets/images/fourier-features-let-networks-learn-high-frequency-functions-in-low-dimensional-domains-source-figure-1.webp)
*Fig 3: The same input mapping helps image and shape regression as well as indirect CT, MRI, and inverse-rendering supervision. | source: [Fourier Features, Figure 1](https://arxiv.org/abs/2006.10739)*

### Boundary

The frequency scale is tuned separately for each dataset: the paper uses sigma 10 for Gaussian features on Natural images, sigma 14 for Text, sigma 5 for MRI, and sigma 6.05 for its NeRF scene. Those values are evidence that the encoding is a controllable prior, not a plug-in constant. Jointly optimizing the feature frequencies with the MLP did not improve the 2D task, and axis-aligned positional encoding performs worse on off-axis sinusoidal signals than isotropic Gaussian features. The experiments are small coordinate-regression problems, often fitting one image or one mesh per network; they do not establish robustness to noisy coordinates or a single bandwidth across scenes.

## High-Level Takeaways

- Fourier features expose high frequencies before the MLP has to discover them through slow optimization.
- The NTK view explains both benefits and failure modes: the frequency scale widens the learnable spectrum, while too much bandwidth produces noisy interpolation.
- Gaussian features improve every task in the paper's Table 1, including indirect CT, MRI, and NeRF-style supervision.
- Bandwidth must be selected with the target signal and sampling pattern; the paper does not justify one encoding scale for all spatial tasks.
