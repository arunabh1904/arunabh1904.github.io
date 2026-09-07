---
title: 'SOLOFusion: Time Will Tell for Temporal Multi-View 3D Object Detection'
date: '2022-10-05T04:00:00.000Z'
section: paper-shorts
postSlug: solofusion-temporal-multiview-3d-object-detection
legacyPath: /paper shorts/2022/10/05/solofusion-temporal-multiview-3d-object-detection.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2022 – Time Will Tell: New Outlooks and a Baseline for Temporal Multi-View 3D Object Detection (SOLOFusion)'
---
## 2022 – SOLOFusion

**arXiv:** [2210.02443](https://arxiv.org/abs/2210.02443)

**Code:** [Divadi/SOLOFusion](https://github.com/Divadi/SOLOFusion)

## Summary

> SOLOFusion treats temporal camera detection as stereo with two complementary budgets: high-resolution features make nearby frames easier to match, while a long history creates enough baseline for coarse features to localize depth. On nuScenes, the reported ResNet-50 validation model reaches 42.7 mAP and 53.4 NDS; the larger test model reaches 54.0 mAP and 61.9 NDS. The paper's useful contribution is the measured resolution/history trade-off, not a claim that more frames or more depth planes are free.

## Core Insights

### Match resolution to temporal baseline

SOLOFusion begins from the observation that camera-only temporal fusion is a form of multi-view stereo. A nearby frame gives small parallax, so a high-resolution feature map is useful; a frame far back in the sequence gives a larger baseline, so a low-resolution feature can still distinguish competing depths. The model therefore uses two paths. Its long-term path aligns the BEV features from 16 previous timesteps and concatenates them into a low-resolution BEV cost volume. Its short-term path uses a high-resolution two-view depth module, group correlation, and Gaussian-spaced top-k depth hypotheses before the detection head. The two paths meet after their different matching jobs.

The architecture figure is easiest to read as a budget allocation. Follow the image backbone into the low-resolution BEV stream and see the historical features accumulate; then follow the separate high-resolution short-term branch that samples a few depth candidates. The design does not build a high-resolution cost volume across the whole history because the paper finds that combination too expensive.

![SOLOFusion: Time Will Tell for Temporal Multi-View 3D Object Detection source figure: The framework of SOLOFusion.](/assets/images/solofusion-temporal-multiview-3d-object-detection-paper-figure.webp)
*Fig 1: High-resolution short-term two-view depth matching and low-resolution long-term BEV feature fusion provide complementary temporal baselines before the detection head. | source: [SOLOFusion, Figure 7](https://arxiv.org/abs/2210.02443)*

### Long history buys localization potential

The paper's analysis defines localization potential from how far apart a 3D point projects between temporal views. The source Figure 5 plots the relative increase as more timesteps are added; each camera heatmap uses a different scale, so compare trends within a panel. The intuition is simple: a distant object can move several feature pixels over a long baseline even when a nearby frame produces a subpixel displacement. That extra separation makes depth matching identifiable.

![Figure 5 from SOLOFusion: Time Will Tell for Temporal Multi-View 3D Object Detection](/assets/images/solofusion-temporal-multiview-3d-object-detection-source-figure-5.webp)
*Fig 2: Relative localization potential rises as more timesteps create larger temporal baselines; each camera heatmap has its own scale, so the within-panel trend is the meaningful comparison. | source: [SOLOFusion, Figure 5](https://arxiv.org/abs/2210.02443)*

The ablation follows that geometry. Starting from a single-frame BEVDepth baseline, adding one historical frame improves velocity strongly but changes mAP only slightly. Increasing the long-term history from one to 16 frames raises mAP from 31.6 to 37.7 and improves translation error from 0.734 to 0.655; 41 frames no longer helps because the visible regions overlap less. This is a localization result, not simply a bigger temporal feature tensor.

### Short-term matching spends depth hypotheses carefully

Short-term stereo has the opposite bottleneck: high-resolution features are informative but every additional depth candidate multiplies the matching work. The paper evaluates all 112 depth hypotheses, 28 uniform samples, seven uniform samples, naive top-k from monocular depth, and Gaussian-spaced top-k. Matching all 112 falls to 2.9 FPS and 8.5 GB; seven guided hypotheses keep memory at 3.3 GB while preserving the useful candidate locations. Gaussian spacing improves mAP and mATE over naive top-k with only a small FPS decrease. The monocular depth estimate is therefore a retrieval prior, not the final answer.

![Figure 1 from SOLOFusion: Time Will Tell for Temporal Multi-View 3D Object Detection](/assets/images/solofusion-temporal-multiview-3d-object-detection-source-figure-1.webp)
*Fig 3: Candidate depth hypotheses project farther apart between temporal views as the baseline grows, which is the geometric signal that the short- and long-term matching modules exploit. | source: [SOLOFusion, Figure 1](https://arxiv.org/abs/2210.02443)*

Table 6 makes the complementarity concrete. The non-temporal baseline has mATE 0.722 m at 17.6 FPS. Short-term fusion reduces the error to 0.670 m at 12.2 FPS, while long-term fusion reaches 0.650 m at 15.9 FPS. Both together reach 0.605 m at 11.4 FPS. The corresponding absolute reductions are 0.052 m, 0.072 m, and 0.117 m: the combined result retains much of each component's benefit. These are distances, not percentage-point changes or a sum of mAP and localization improvements. Short-term-only memory remains 3.3 GB; adding long-term state raises the measured footprint to 3.6 GB.

The reported test model ranked first on the camera-only nuScenes track at submission. Comparisons across methods still involve different training and test-time settings, so the controlled component table gives clearer evidence for the proposed temporal trade-off.

### History also tests the calibration contract

Temporal baselines amplify pose error, timestamp drift, rolling-shutter mismatch, and moving-object effects. SOLOFusion aligns historical BEV features with ego motion, so its long-term gain assumes that alignment is accurate. A deployment evaluation should therefore slice the result by range, motion, frame delay, and pose noise; the source paper's localization-potential curve does not establish robustness under those corruptions.

## High-Level Takeaways

- SOLOFusion's key insight is a resolution/history trade: high-resolution short-term matching and low-resolution long-term BEV fusion solve different stereo regimes.
- Sixteen previous frames improve detection and translation before saturation; guided depth sampling reduces the short-term module’s tested memory cost from 8.5 GB with 112 candidates to 3.3 GB with seven.
- Table 6 reduces mATE from 0.722 m to 0.670 m with short-term fusion, 0.650 m with long-term fusion, and 0.605 m with both; the combined model trades additional runtime for complementary localization gains.
- Temporal alignment makes pose, timestamp, and moving-object errors part of the model's validity boundary.
