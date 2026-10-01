---
title: "Map-Det3D: Metric Feed-Forward 3D Reconstruction Prior for Multi-view 3D Object Detection from Streaming Inputs"
date: '2026-08-12T00:00:00.000Z'
section: paper-shorts
postSlug: map-det3d-metric-feed-forward-3d-reconstruction-prior-for-multi-view-3d-object-detection-from-streaming-inputs
legacyPath: /paper shorts/2026/08/12/map-det3d-metric-feed-forward-3d-reconstruction-prior-for-multi-view-3d-object-detection-from-streaming-inputs.html
tags: ["3D Detection", "Reconstruction Priors", "Indoor Perception"]
field: 'BEV Perception'
summary: "2026 – Map-Det3D: Metric Feed-Forward 3D Reconstruction Prior for Multi-view 3D Object Detection from Streaming Inputs"
---

## 2026 – Map-Det3D: Metric Feed-Forward 3D Reconstruction Prior for Multi-view 3D Object Detection from Streaming Inputs

**arXiv:** [2608.12179](https://arxiv.org/abs/2608.12179) · [Full text, v1](https://arxiv.org/html/2608.12179v1) · **Project:** [Map-Det3D](https://royyang0714.github.io/Map-Det3D) · [Official code](https://github.com/cvg/Map-Det3D)

## Summary

> Map-Det3D adapts MapAnything's metric reconstruction prior into a detector for streaming monocular video. A short window supplies multiple views; the detector predicts box geometry up to scale, then uses the backbone's scale factor to recover metric centers and dimensions. The evidence is class-agnostic indoor detection on CA-1M and zero-shot ScanNet, not autonomous-driving detection. Its main insight is that reconstruction features need object-aware adaptation: simply adding temporal views to a frozen backbone produces almost no gain.

## Core Insights

### Separate an object's geometry from the window's metric scale

The comparison with [VGGT](/paper%20shorts/2025/03/14/vggt-visual-geometry-grounded-transformer.html) is about the representation handed to the next task. VGGT establishes joint multi-view geometry prediction in a normalized coordinate system. Map-Det3D specifically builds on MapAnything, which separates metric scale from scene geometry, and adapts that representation for object instances. Neither model is itself a language-conditioned action policy.

At each time step, Map-Det3D processes the current image and four earlier frames, returning boxes for the current frame only. MapAnything fuses the views through a transformer and predicts a shared scale factor from a learned scale token. Camera intrinsics and poses are optional inputs, but the strongest ablation configuration uses both; no depth sensor is required at inference.

The detection head regresses unscaled horizontal coordinates, log-depth, log-dimensions, rotation, and objectness. Multiplying coordinates and positive exponentiated sizes by the shared scale factor produces metric boxes. This couples the objects to one window-level geometric interpretation instead of asking every detected object to recover metric depth independently.

![Map-Det3D shares reconstruction features and a scale token across temporal views](/assets/images/map-det3d-metric-feed-forward-3d-reconstruction-prior-for-multi-view-3d-object-detection-from-streaming-inputs-source-figure-2.webp)
*Fig 1: The shared multi-view transformer supplies object features, while the separate scale head converts unscaled geometry to metric units. The trainable components show where a reconstruction model is adapted into a detector. | source: [Map-Det3D, Figure 2](https://arxiv.org/abs/2608.12179)*

[Open figure at full resolution](/assets/images/map-det3d-metric-feed-forward-3d-reconstruction-prior-for-multi-view-3d-object-detection-from-streaming-inputs-source-figure-2.webp)

“Direct 3D” does not mean the architecture contains no 2D boxes. Dense 2D proposals initialize object queries; reference boxes guide deformable attention and Hungarian matching. The distinction is the final geometry parameterization, which avoids lifting a detected pixel center through a separately regressed range. The training loss still includes auxiliary 2D box supervision alongside disentangled 3D corner losses.

### Project geometry features into object queries

The 16-layer multi-view transformer produces 1,536-channel features. The detector reads its final encoder output and intermediate levels 7, 11, and 15. Per-level MLPs project these four feature sets to **256 channels**, then concatenate spatial positions and levels. Temporal fusion has already happened in the backbone; the detector subsequently processes each view's features independently. These MLPs are a geometry-to-detector interface, not a vision-to-language projector.

Dense 2D proposals supply reference boxes for deformable attention. A decoder refines object queries and predicts intermediate boxes for deep supervision. The [released head](https://github.com/cvg/Map-Det3D/blob/665e054ff4fb74e3363f72fb3f534c8addf60338/mapdet3d/op/mapdet3d/head.py) defaults to 900 matching queries and six decoder layers. Source Figure 3 locates the projection layers and shows why auxiliary 2D prediction remains part of a direct-3D detector.

![Source Figure 3: four MapAnything feature levels projected through MLPs into a detector with 3D, auxiliary 2D, and objectness heads](/assets/images/map-det3d-source-figure-3.webp)
*Fig 2: Per-level MLPs connect reconstruction features to object queries. The scale factor enters the 3D head, while auxiliary 2D boxes guide matching and query refinement. | source: [Map-Det3D, Figure 3](https://arxiv.org/html/2608.12179v1#S3.F3)*

[Open figure at full resolution](/assets/images/map-det3d-source-figure-3.webp)

For a detected chair, the head regresses unscaled horizontal coordinates, log-depth, log-size, and a continuous 6D rotation. The scale factor converts its center and dimensions to metres; the rotation is converted from an allocentric representation to the camera-relative frame using the center direction. A wrong window-scale estimate can therefore affect multiple objects coherently. This is a shared learned scale prior, not an independent range measurement for each chair.

### Adapt the backbone with a supervised detection recipe

The paper's final model trains for **100,000 steps**, batch size 64, on 16 RTX 4090 GPUs for 1.5 days; ablations use 50,000 steps. The base learning rate is $10^{-4}$ with cosine annealing, while the multi-view transformer and scale head use one tenth of that rate. The released implementation freezes the DINOv2 encoder, multimodal input encoders, and reconstruction depth/pose heads. The shared multi-view transformer, scale components, projection MLPs, and detection head are adapted for boxes.

Training samples one to five views and varies temporal sampling from 2 to 10 FPS. Images use MapAnything's aspect-ratio choices: resize the short edge and zero-pad, preserving scene content instead of discarding it through a center crop. The paper does not specify one universal pixel resolution for this variable-aspect setup. Inference uses the current frame and four past frames.

Hungarian matching uses auxiliary 2D boxes. Matched queries receive objectness focal loss, 2D L1/GIoU losses, and disentangled 3D corner losses. For the latter, predict one attribute—center, depth, size, or rotation—while holding the others at their ground-truth values, then compare resulting corners. Rotation uses a Chamfer distance to handle equivalent corner arrangements. Supervising encoder proposals and every decoder layer gives the projection and query machinery an object-level signal; no language alignment or action loss is involved.

The public code adds implementation detail but is **not an exact declaration of the final paper run**. At [commit 665e054](https://github.com/cvg/Map-Det3D/tree/665e054ff4fb74e3363f72fb3f534c8addf60338), the optimizer is AdamW with weight decay $10^{-4}$, and the 2D focal/L1/GIoU weights are 1/5/2. Its default experiment instead sets 50,000 iterations and `samples_per_gpu = 5`, whereas the paper describes a 100,000-step final run with a maximum of four samples per GPU. Reproducing the headline result requires reconciling the sampling and schedule, rather than assuming the default command reproduces it unchanged.

### Temporal evidence becomes useful after the backbone learns object structure

The 50,000-step CA-1M ablation begins at 11.7 AP15 with a frozen single-view setup. Adding multiple views gives 11.8, and unfreezing only the scale head gives 11.9. The prior can reconstruct geometry, yet its original features do not automatically expose the information a box decoder needs.

With both the scale head and multi-view transformer adapted, the single-view result is 14.5; adding temporal views under that same adaptation raises it to 17.2. This is the cleaner comparison for temporal evidence. Camera conditioning supplies another contribution: under the frozen multi-view setting, intrinsics raise AP15 from 11.8 to 12.6, and adding poses reaches 17.3. Combining adaptation, temporal views, intrinsics, and poses reaches 21.2.

The direct-3D head comparison is narrower: with the multi-view transformer and scale head frozen, it improves CA-1M AP15 from 15.6 to 17.3 and ScanNet AP15 from 10.8 to 12.6. These ablations support complementary choices rather than attributing the whole gain to one scale token.

### How the detection benchmarks are constructed

CA-1M supplies measured scene geometry and human object annotations. The [dataset paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Lazarow_Cubify_Anything_Scaling_Indoor_3D_Object_Detection_CVPR_2025_paper.pdf) starts from ARKitScenes captures registered to high-resolution FARO laser scans. Annotators draw oriented 3D boxes on the scans, consult aligned RGB views for incomplete or transparent surfaces, and can initialize boxes through a model-assisted 2D selection. The supervision is therefore human-reviewed spatial annotation, with model assistance, rather than boxes generated solely by a VLM.

A rendering stage projects world-space annotations into each camera and accounts for view-frustum and occlusion effects to create per-frame 2D/3D targets. Map-Det3D describes more than 400,000 objects across over 1,000 scenes and 3,500 captures, yielding about 13 million training and 1.8 million validation frames. Its in-domain test uses held-out scenes. Millions of correlated views should not be read as millions of independent physical environments.

For out-of-domain evaluation, the authors select **100 ScanNet scenes** following BoxFusion and sample every 25th frame. Per-frame evaluation uses the long-tailed ScanNet200 setting; per-scene evaluation uses ScanNetV2's 18-class annotation setting. Predictions and labels are then evaluated class-agnostically: AP matches oriented 3D cuboids by IoU at 0.15, 0.25, or 0.50. These thresholds measure progressively stricter geometric overlap, not category recognition. The paper evaluates all supplied ground-truth boxes regardless of visibility and truncation, so its results should not be mixed with protocols that filter difficult boxes.

### Better transfer does not imply category understanding or depth-sensor parity

The final model trains for 100,000 steps on CA-1M, whose exhaustive annotations cover indoor objects beyond a short class list. All results are evaluated class-agnostically. A box indicates an object hypothesis; the model does not assign closed-set or open-vocabulary semantic labels. The released configuration enables both camera intrinsics and extrinsics. Absence of a depth sensor should therefore not be confused with absence of camera metadata; the metadata-free case needs its own comparison.

| CA-1M detector | AP25 | AP50 |
| --- | ---: | ---: |
| CuTR, monocular | 13.5 | 2.4 |
| ImVoxelNet, offline multi-view | 10.1 | 2.3 |
| Map-Det3D, online multi-view | 16.9 | 3.5 |
| FCAF, point cloud | 29.3 | 11.2 |

Map-Det3D improves on the listed image-only baselines while remaining below the depth-based FCAF result, particularly at stricter overlap. AP25 and AP50 refer to 0.25 and 0.50 cuboid IoU thresholds. This evaluation includes boxes regardless of visibility or truncation, which also affects comparison with other detection protocols.

On the selected ScanNet200 zero-shot benchmark, the final model reaches 15.2 AP15 and 9.7 AP25, versus CuTR's 4.3 and 2.1. That controls the detector-training dataset, but the models still differ in reconstruction pretraining and architecture; it does not isolate architecture alone.

### A bounded history limits cost without making streaming free

The temporal/runtime table reports 14.3 FPS and 5.8 GB for one frame, 8.3 FPS and 6.4 GB for five, and 6.3 FPS and 6.7 GB for seven. The table does not state the inference GPU; the training GPU should not be silently treated as the timing hardware. Five frames give 21.2 CA-1M AP15 in that ablation; seven give 21.1. More context adds cost without improving the measured result. The separate offline five-view row predicts all five frames together and should not be described as the causal streaming setting.

For scene-level evaluation, the authors add simple IoU-based tracking in world coordinates and retain a larger associated box when an object becomes more visible. This reaches 22.7 AP25 on ScanNetV2, versus 11.8 for RGB-only BoxFusion; depth-using BoxFusion reaches 24.6. The heuristic can improve incomplete boxes but does not establish robust tracking of arbitrary dynamic objects.

Indoor-only training, class-agnostic output, optional camera metadata, and a relatively expensive backbone define the supported application. Outdoor driving and semantic detection remain extensions to test.

## High-Level Takeaways

- A shared reconstruction scale can coordinate metric box geometry across temporal views, while 2D proposals remain useful for attention and matching.
- Adapt the reconstruction representation for objects before expecting additional views to help detection.
- Keep camera-conditioning, training length, IoU threshold, and per-frame versus per-scene evaluation attached to each result.
- The demonstrated transfer is indoor and class-agnostic; reconstruction priors are promising geometry, not evidence of an already general driving detector.
