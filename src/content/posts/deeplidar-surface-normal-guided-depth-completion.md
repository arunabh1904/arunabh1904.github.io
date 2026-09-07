---
title: 'DeepLiDAR: Surface-Normal-Guided Depth Completion'
date: '2018-12-02T05:00:00.000Z'
section: paper-shorts
postSlug: deeplidar-surface-normal-guided-depth-completion
legacyPath: /paper shorts/2018/12/02/deeplidar-surface-normal-guided-depth-completion.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2018 – DeepLiDAR: combine direct completion with an intermediate surface-normal path'
---
## Summary

> DeepLiDAR completes sparse LiDAR depth using an RGB image and an intermediate surface-normal prediction. Its useful insight is that a geometric representation can help locally while becoming unreliable elsewhere: a small normal error can create a large depth error on a distant road. The model therefore combines normal-guided and color-guided depth estimates with learned spatial weights, and separately learns which sparse depth observations to trust near occlusions. It reports 758.38 mm RMSE on the KITTI test benchmark, with ablations supporting both mechanisms. The confidence mask is trained to improve reconstruction, rather than calibrated as a probability of measurement correctness.

## Core Insights

### A normal describes local shape, but recovering distance can amplify its error

A surface normal says how a surface is oriented. Together with nearby depth observations, it can guide completion across a road, wall, or vehicle surface without requiring an image network to infer every absolute distance independently. This is particularly useful where sparse LiDAR already anchors the geometry.

The difficulty is perspective. For a distant, nearly horizontal road, neighboring image rays meet the surface at shallow angles. Tilting the inferred surface slightly can move the ray–surface intersection a long distance. The same angular error is much less damaging nearby. DeepLiDAR uses this geometric sensitivity to motivate a second prediction route rather than requiring normal-based recovery everywhere.

![DeepLiDAR Figure 2 illustrates depth sensitivity to the same surface-normal error at different ranges](/assets/images/deeplidar-source-figure-2.png)
*Fig 1: The yellow normal has the same angular error in both examples, but its distant ray intersection produces a much larger red depth error. Useful local orientation is not equally reliable as a distance constraint at every range. | source: [DeepLiDAR, Figure 2](https://arxiv.org/abs/1812.00488)*

The normal-guided path is learned, not a separate analytic plane-fitting solver at inference. It first predicts normals from RGB and sparse depth, then combines those normals with sparse depth to predict a dense map. The parallel color path predicts dense depth from RGB and sparse depth directly. Both paths retain metric input; “color pathway” does not mean monocular-only estimation.

### Fuse the two depth estimates where each is useful

Each path produces a spatial score map from its final features. A softmax converts the two scores into per-pixel weights, and the final output is their weighted sum: $\hat D=w_c\hat D_c+w_n\hat D_n$. This attention is a two-way mixture over depth estimates, rather than transformer attention over image tokens.

![DeepLiDAR Figure 3 shows color and normal pathways, a confidence mask, and weighted depth fusion](/assets/images/deeplidar-source-figure-3.png)
*Fig 2: The upper path predicts depth directly; the lower path first predicts normals and then depth. Their attention maps decide how to combine outputs, while the separate confidence mask controls sparse-depth reliability in the normal pathway. | source: [DeepLiDAR, Figure 3](https://arxiv.org/abs/1812.00488)*

The learned attention maps put more normal-path weight on many nearby regions and more color-path weight on some distant regions. This matches the geometric motivation, but the weights remain features optimized for final depth accuracy. They are not a proof that one branch has a known uncertainty at every pixel.

The deep completion unit supplies the common building block. Separate encoders process RGB or normals and sparse depth. During decoding, appearance or normal features are concatenated, while sparse-depth features are added at matching resolutions. The authors argue that addition encourages the decoder and sparse-depth branch to use compatible features. Replacing this structure with early concatenation worsens validation RMSE from 687.00 to 767.82 mm, though that comparison changes the fusion architecture as a whole rather than isolating addition alone.

### A valid LiDAR return can still be wrong for the camera pixel

The confidence mechanism addresses a different problem from normal sensitivity. A camera and LiDAR occupy different positions. Near a foreground boundary, a LiDAR return from a background surface can project into a camera region affected by foreground occlusion. A binary availability mask says that a return exists, but cannot say whether it is reliable for that image pixel.

DeepLiDAR predicts a soft confidence mask from the color pathway and supplies it, alongside sparse depth and predicted normals, to the normal-guided depth estimator. It receives no direct ground-truth confidence labels; the reconstruction losses teach it to downweight confusing observations. Replacing this mask with binary availability raises validation RMSE from 687.00 to 756.32 mm. The mechanism therefore learns more than where data are missing, while remaining specific to the reconstruction objective and training distribution.

### Separate the component ablations from the benchmark headline

| KITTI validation configuration | RMSE, mm | What is removed |
| --- | ---: | --- |
| Full model | 687.00 | Nothing |
| Normal pathway only | 729.96 | Color-path contribution and attention integration |
| Color pathway only | 774.25 | Normal prediction and two-path fusion |
| Binary confidence mask | 756.32 | Learned sparse-depth reliability |
| Early-fusion architecture | 767.82 | Deep completion unit's late fusion |

These are retrained ablations on validation data. The separate held-out test result is 758.38 mm RMSE and 226.50 mm MAE, compared with 814.73 and 249.95 for the cited Sparse-to-Dense model. DeepLiDAR does not win every metric in that table: Spade-RGBsD has lower inverse-depth RMSE, 2.17 versus 2.56, and lower inverse-depth MAE, 0.95 versus 1.15. The paper's headline ranking uses ordinary depth RMSE, which emphasizes large absolute errors. Runtime is 0.07 seconds per image on a GTX 1080 Ti.

Training also matters. The authors render 50,000 synthetic CARLA examples with depth and normals, then fine-tune on KITTI, deriving normal targets from depth by local plane fitting. They train normal prediction first, then the two depth pathways, and finally their combined output. “End-to-end” describes the final joint optimization; it does not mean every component starts from scratch together or that normal supervision is absent.

Uniformly subsampling KITTI returns down to about 72 observed pixels tests extreme sparsity, but not every pattern of sensor failure. The NYUv2 experiment trains on indoor images before evaluating 654 test images, so it demonstrates reuse of the architecture indoors rather than zero-shot outdoor-to-indoor transfer. These distinctions keep the result useful: surface normals, spatial fusion, and learned confidence are complementary reconstruction tools whose value depends on geometry and supervision.

## High-Level Takeaways

- Surface normals supply a useful local shape constraint, but perspective can amplify their errors at long range. An alternative prediction path handles that weakness.
- Output attention and input confidence solve different problems: which depth estimate to use, and which sparse observations to trust.
- The normal path remains a learned depth estimator anchored by sparse measurements. It is neither an analytic guarantee nor a replacement for metric input.
- The component ablations favor combining geometric guidance, late feature fusion, and soft confidence; test metrics, validation ablations, and hardware timing should remain separate.
- Synthetic normal supervision and staged training are part of the method. Sparse-input and indoor experiments do not establish arbitrary failure robustness or zero-shot domain transfer.
