---
title: 'VGGT: Visual Geometry Grounded Transformer'
date: '2025-03-14T09:00:00.000Z'
section: paper-shorts
postSlug: vggt-visual-geometry-grounded-transformer
legacyPath: /paper shorts/2025/03/14/vggt-visual-geometry-grounded-transformer.html
tags: ["3D Vision", "Reconstruction", "Vision Foundations"]
field: 'Vision Foundations'
summary: '2025 – VGGT: Visual Geometry Grounded Transformer'
---

## 2025 – VGGT: Visual Geometry Grounded Transformer

**Paper:** [arXiv:2503.11651](https://arxiv.org/abs/2503.11651) · [Full text and appendix, v1](https://arxiv.org/html/2503.11651v1) · [Official code](https://github.com/facebookresearch/vggt)

## Summary

> VGGT jointly predicts cameras, depth, point maps, and tracking features from a collection of images in one forward pass. On ten-view RealEstate10K camera estimation, it reaches 85.3 AUC@30 in approximately 0.2 seconds on an H100; bundle adjustment raises this to 93.5 at approximately 1.8 seconds. Its central contribution is a shared, multi-view geometry representation trained with redundant supervision. The output uses a learned normalized scale, and the original model is a batch reconstruction system rather than a persistent streaming map or metric-scale driving perception stack.

## Core Insights

### Fuse the views before predicting their geometry

DUSt3R predicts point maps from image pairs; larger reconstructions require aligning those pairwise predictions. VGGT moves the multi-view interaction into the network. Every image contributes patch tokens, and alternating self-attention combines within-image structure with evidence across images. A room corner can therefore be interpreted jointly with its observations from other cameras before a depth or camera head commits to geometry.

The image tokenizer uses pretrained [DINOv2](/paper%20shorts/2023/04/14/dinov2-learning-robust-visual-features-without-supervision.html). The shared backbone contains **24 blocks, each with one frame-wise and one global attention layer**, for 48 attention layers. Each uses width 1,024 and 16 heads, with QK normalization and LayerScale initialized to 0.01. This is self-attention over different token groupings, not a cross-attention decoder that repeatedly treats another image as a separate conditioning source. The complete model has approximately 1.2 billion parameters.

The architecture figure makes the sharing explicit: camera prediction and dense geometry read the same cross-view representation.

![Source Figure 2: VGGT image tokens, alternating global and frame attention, camera head, and dense prediction heads](/assets/images/vggt-source-figure-2.webp)
*Fig 1: Image tokens exchange information across views before branching into camera and dense outputs. The shared backbone makes geometric supervision from one task available to the others. | source: [VGGT, Figure 2](https://arxiv.org/html/2503.11651v1#S1.F2)*

[Open figure at full resolution](/assets/images/vggt-source-figure-2.webp)

Each image also receives one camera token and four register tokens. The first image has a distinct set of learned tokens, identifying the reference coordinate frame; the remaining images can be reordered without changing their geometric role. The camera head uses four additional self-attention layers and a linear prediction layer to produce a rotation quaternion, translation, and horizontal/vertical field of view. The principal point is assumed to lie at the image center.

For dense outputs, DPT reads intermediate features from blocks 4, 11, 17, and 23 and upsamples them. Convolutional heads predict depth and point maps, together with confidence-related uncertainty outputs. A separate CoTracker2-style module uses dense features and correlations to track queried image points across views. The backbone thus predicts tracking **features**, not a ready-made trajectory for every possible query. These are geometry output heads; the original VGGT contains no language model or vision-to-language projector.

### The shared coordinate system does not provide metric scale

Point maps place each pixel's 3D point in the first camera's coordinate system. During training, camera translations, depths, and point maps are divided by the mean distance of ground-truth scene points from that origin. Predictions are trained to reproduce this normalization rather than being normalized afterward. This fixes the otherwise arbitrary reconstruction scale and reference frame, but it does not turn an uncalibrated photograph into an independently measured distance in metres.

That distinction matters for mapping and control. A geometrically consistent room reconstruction still needs an appropriate scale source before its distances can support metric actions. [Map-Det3D](/paper%20shorts/2026/08/12/map-det3d-metric-feed-forward-3d-reconstruction-prior-for-multi-view-3d-object-detection-from-streaming-inputs.html) uses MapAnything's explicit metric scale factor when decoding boxes; it does not simply attach a detection head to the original VGGT and inherit metric units.

VGGT also deliberately predicts overlapping quantities. Camera parameters and depth already determine a point map, yet the direct point-map branch remains useful during training. On ETH3D, using the direct branch gives an overall error of **0.709**, while unprojecting predicted depth with predicted cameras gives **0.677**. The better inference route need not be the only useful supervision route.

### Train on geometric labels, including correspondences built from depth

A training example consists of several views of the same scene with camera, depth, point-map, and correspondence supervision where available. The appendix samples a dataset, then a scene uniformly within it, then 2–24 views; it describes 48 total frames per batch. Sequences with fewer than 24 frames are excluded. Frames are resized to a longest side of 518 pixels and cropped around the principal point to a shorter side between 168 and 518, in multiples of the 14-pixel patch size. Independent color augmentation across views prevents identical illumination from becoming a requirement for matching.

The mixture spans real and synthetic sources: CO3Dv2, BlendedMVS, DL3DV, MegaDepth, Kubric, WildRGB, ScanNet, Hypersim, Mapillary, Habitat, Replica, MVS-Synth, PointOdyssey, Virtual KITTI, Aria synthetic/digital-twin data, and artist-created assets. Geometry comes from sensors, rendering engines, or SfM. Dataset weights are described as approximately similar but **exact mixture probabilities are not reported**. The artist-created asset collection is another reproducibility boundary.

Tracking targets are constructed geometrically: unproject a query-frame pixel using depth, transform it into another camera, project it, and retain the correspondence when the projected depth agrees with that camera's depth map. Low-overlap frames are excluded during sampling; examples without valid correspondences omit the tracking loss. These targets are not human-written trajectory labels.

The total objective adds camera, depth, and point-map losses with unit relative weights, plus a tracking term weighted by 0.05. Cameras use Huber regression. Depth and point maps use confidence-weighted value and gradient errors with a logarithmic regularizer; tracking uses coordinate errors and visibility classification. Optimization runs for **160,000 AdamW steps**, with an 8,000-step warmup, peak learning rate $2\times10^{-4}$, cosine decay, and gradient clipping at 1.0. The paper reports nine days on 64 A100 GPUs, using bfloat16 and gradient checkpointing. DINOv2 initializes tokenization; the geometry backbone and output tasks are trained jointly.

### Read each score with its construction and camera assumptions

VGGT reuses established evaluation datasets. Its results answer several different questions, so their alignment rules and allowed inputs matter as much as the headline numbers.

| Evaluation | How the scored example is formed | Result and boundary |
| --- | --- | --- |
| CO3Dv2 / RealEstate10K cameras | Ten randomly selected images per scene; relative rotation and translation-direction errors across image pairs; AUC integrated up to 30 degrees | Feed-forward AUC@30: 88.2 / 85.3. RealEstate10K is unseen during training; CO3Dv2 is a training source |
| DTU dense reconstruction | Compare predicted and reference point clouds using nearest-neighbour accuracy, completeness, and their mean | Overall 0.382 versus DUSt3R's 1.741, both without ground-truth cameras; camera-assisted GeoMVSNet is lower at 0.295 |
| ETH3D point maps | Ten sampled frames; Umeyama alignment to reference geometry and official invalid-point masks | Depth plus cameras gives 0.677 versus MASt3R's 0.826; alignment means this is not a raw metric-scale test |
| ScanNet-1500 matching | Detect ALIKED query points, track them into the second image, estimate an essential matrix, and score relative pose | AUC@20 is 73.4 versus RoMa's 70.9; corresponding ScanNet scenes are excluded from training |

The appendix exposes an additional boundary. On IMC phototourism, feed-forward VGGT reaches **71.26 AUC@10**, below VGGSfMv2's 76.82; adding bundle adjustment reaches 84.91. The training split follows DUSt3R/MASt3R, with some MegaDepth scenes overlapping IMC even when images differ. This is not a wholly unseen-scene benchmark. VGGT reduces the need for optimization in many settings, while strong camera refinement remains valuable.

The matched-backbone ablation is more specific than comparisons between whole systems. Holding parameter count and total attention layers fixed, alternating attention gives ETH3D error 0.709, versus 0.827 for global-only attention and 1.061 for cross-attention. Removing camera supervision worsens the direct point-map error to 0.834; removing tracking gives 0.790. Shared supervision contributes beyond simply having a large image encoder.

### Geometry priors help when observations are incomplete, but can also supply guesses

The paper's qualitative examples include an oil painting, non-overlapping views, and repeated textures. They show how a learned prior fills gaps that direct correspondence cannot resolve. They do not provide independent ground truth for the inferred scene behind a painting.

![Source Figure 3: VGGT and DUSt3R reconstructions from a painting, two views, and a 32-view collection](/assets/images/vggt-source-figure-3.webp)
*Fig 2: The examples contrast learned scene geometry under weak overlap and repeated texture. Their visual plausibility illustrates a prior; it is not a measurement of hidden surfaces or absolute scale. | source: [VGGT, Figure 3](https://arxiv.org/html/2503.11651v1#S3.F3)*

[Open figure at full resolution](/assets/images/vggt-source-figure-3.webp)

The efficiency claim also needs a frame count. At $336\times518$ on an H100 with FlashAttention v3, the **backbone alone** takes 0.14 seconds and 3.63 GB for ten frames, 3.12 seconds and 21.15 GB for 100, and 8.75 seconds and 40.63 GB for 200. A DPT head adds approximately 0.03 seconds and 0.2 GB per frame. “Hundreds of views in under a second” is therefore not a reliable deployment budget. Global attention still couples the whole input collection, and the original model has no persistent streaming state.

Features transfer beyond static reconstruction after adaptation: replacing CoTracker's feature extractor and fine-tuning the modified tracker on Kubric improves RGB-Stacking visible-point accuracy from 78.9 to 84.0. A separate novel-view-synthesis adaptation encodes target Plücker rays and predicts RGB through DPT, achieving 30.41 PSNR on GSO without input-camera parameters. These are fine-tuned downstream models, not extra capabilities obtained by calling the unchanged reconstruction checkpoint. The original model struggles with fisheye/panoramic imagery, extreme camera rotations, and substantial non-rigid motion.

## High-Level Takeaways

- Joint multi-view attention moves pairwise alignment into a learned representation, but optional bundle adjustment still improves difficult camera estimation.
- Camera, depth, point-map, and tracking losses provide complementary supervision even when their outputs are geometrically redundant.
- A first-camera coordinate frame and normalized scale are not a guarantee of metric distance; scale calibration remains a separate deployment decision.
- Use frame-count-specific backbone and head timings. The original 1.2B model is a batch geometry backbone, not an indefinitely streaming mapper.
- For robotics or VLA use, evaluate how a downstream connector consumes VGGT features and preserves geometry; the original paper does not establish an action policy.
