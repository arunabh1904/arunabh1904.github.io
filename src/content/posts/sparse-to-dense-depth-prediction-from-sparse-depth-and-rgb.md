---
title: 'Sparse-to-Dense: Depth Prediction from Sparse Depth and RGB'
date: '2017-09-21T04:00:00.000Z'
section: paper-shorts
postSlug: sparse-to-dense-depth-prediction-from-sparse-depth-and-rgb
legacyPath: /paper shorts/2017/09/21/sparse-to-dense-depth-prediction-from-sparse-depth-and-rgb.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2017 – Sparse-to-Dense: complete sparse runtime range measurements with RGB'
---
## 2017 – Sparse-to-Dense

**arXiv:** [1709.07492](https://arxiv.org/abs/1709.07492)

## Summary

> Sparse-to-Dense treats a few range measurements as runtime evidence, not as labels hidden behind a camera-only model. An RGB image supplies boundaries and semantic regularities; sparse depth fixes metric scale; an encoder-decoder fills the unobserved pixels. On NYU-Depth-v2, 100 random samples cut the reported RGB-only RMSE by more than half, and on KITTI the same order of sparsity reduces RMSE from 6.266 m to 4.303 m. The paper also shows saturation and blurry boundaries, so the useful promise is graceful completion under a known sampling contract rather than arbitrary recovery of metric geometry from appearance.

## Core Insights

### A depth sample is an anchor, not a dense target

The input is a full RGB image plus a sparse depth image in which most pixels are zero because they were never measured. The network must learn the difference between “missing” and “measured zero,” and use measured values as anchors while extending them into nearby regions; the output is still a learned dense prediction rather than a hard copy of the sparse pixels. During training, the authors sample depth pixels from the ground-truth image on the fly with a Bernoulli mask. The expected sample count is controlled, but the exact count varies from example to example, which acts as both data augmentation and a robustness test.

The architecture figure shows why the model is two-dimensional at the interface but not camera-only. The RGB and depth channels enter a ResNet-based encoder, and the decoder upsamples through four UpProj blocks before a final bilinear step. The KITTI and NYU models use different backbones because KITTI's image is roughly three times larger and the same network would exceed the available GPU memory. The design is therefore a practical bottleneck architecture, with the sparse sensor injected near the input and a dense prediction produced at the output.

![Figure 2 from Sparse-to-Dense: Depth Prediction from Sparse Depth and RGB](/assets/images/sparse-to-dense-depth-prediction-from-sparse-depth-and-rgb-source-figure-2.webp)
*Fig 1: The RGB and sparse-depth inputs enter dataset-specific ResNet encoders and meet in an upsampling decoder that produces a full-resolution depth map. The two drawings reflect the different KITTI and NYU memory budgets. | source: [Sparse-to-Dense, Figure 2](https://arxiv.org/abs/1709.07492)*

### The useful ablation is the input density curve

On NYU-Depth-v2, the RGB-only model in the paper reaches RMSE 0.514. With 20, 50, and 200 sparse depth samples, the RGB-plus-depth model reaches 0.351, 0.281, and 0.230 respectively. The gain is not simply an extra training signal: those samples are available at inference and supply absolute scale that the RGB branch cannot infer reliably indoors. As density increases, the sparse-depth-only model also improves, and the color cue becomes less important once the sampled geometry is sufficiently informative.

The outdoor curve exposes a harder boundary. KITTI spans distances to about 100 m, compared with NYU's roughly 10 m indoor range. The paper reports RGB-only RMSE 6.266 m, then 4.884 m with 50 samples, 4.303 m with 100, 3.851 m with 200, and 3.378 m with 500. The fraction of predictions within the paper's reliable threshold rises from 59.1% to 93.5% between zero and 500 samples. More points help, but the curve is not a promise that a fixed tiny sensor will work equally well across range, scene type, and sampling pattern.

The density figure should be read horizontally within each dataset rather than by comparing raw error between indoor and outdoor panels. The left plots are error metrics where lower is better; the right plots are threshold accuracies where higher is better. The RGBd curve drops quickly at low sample counts and then flattens. That shape is the central result: a small number of metric anchors resolves a large scale ambiguity, while later samples mostly refine already-supported surfaces.

![Figure 5 from Sparse-to-Dense: Depth Prediction from Sparse Depth and RGB](/assets/images/sparse-to-dense-depth-prediction-from-sparse-depth-and-rgb-source-figure-5.webp)
*Fig 2: On the NYU-Depth-v2 indoor split, RGBd error falls quickly as the expected number of sparse depth samples increases and then saturates; the paired threshold plots show the same diminishing-return pattern. The KITTI density results are reported separately in the paper's table. | source: [Sparse-to-Dense, Figure 5](https://arxiv.org/abs/1709.07492)*


### Architecture choices matter after the sensor contract is fixed

The paper's NYU architecture ablation needs a careful comparison. The RGB $\mathcal{L}_1$ model with a convolutional first layer and UpProj decoder reaches RMSE 0.528, while the RGBd model with the same choices reaches 0.264; that is the closer modality comparison. The 0.361-to-0.261 change is instead the first encoder convolution changing from ChanDrop to DepthWise while the UpProj decoder stays fixed. The RGB-only rows also change loss and decoder together: the $\mathcal{L}_2$ Conv/DeConv2 row is 0.610, berHu is 0.554, $\mathcal{L}_1$ Conv/DeConv2 is 0.552, and later decoder variants reach 0.533, 0.529, and 0.528. The sparse modality explains the large controlled jump; encoder and decoder choices explain the smaller architectural steps.

The authors also demonstrate dense maps from sparse visual-odometry landmarks and vertically denser LiDAR outputs. Those are useful interface demonstrations: a sparse SLAM map can become a surface that a planner can consume, and a low-resolution LiDAR can appear denser in the image plane. They use the ground-truth depth of the first frame to set absolute scale in the simple visual-odometry demonstration, so that example does not establish a complete metric SLAM system.

## High-Level Takeaways

- Sparse-to-Dense shows a large metric gain from a small runtime hint: on NYU, 100 samples cut the RGB-only RMSE from 0.514 to roughly 0.264 in the controlled RGBd comparison, while the curve saturates as samples accumulate.
- The architecture uses sparse depth as an input anchor and predicts the rest of the surface; it does not make every completed pixel an observation.
- The indoor Figure 5 curve is NYU-only. KITTI's separate table reports the outdoor density trend, where RGB-only RMSE 6.266 m falls to 4.303 m at 100 samples and 3.378 m at 500.
- The deployment claim is conditional on the sampling, range, timing, and calibration contract; blurry boundaries and uniform random masks leave a clear domain limit.
