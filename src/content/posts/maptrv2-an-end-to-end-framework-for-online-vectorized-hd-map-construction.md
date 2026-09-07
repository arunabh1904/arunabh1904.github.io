---
title: 'MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction'
date: '2023-08-10T00:00:00.000Z'
section: paper-shorts
postSlug: maptrv2-an-end-to-end-framework-for-online-vectorized-hd-map-construction
legacyPath: /paper shorts/2023/08/10/maptrv2-an-end-to-end-framework-for-online-vectorized-hd-map-construction.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: "2023 – MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction"
---
## Summary

> MapTRv2 keeps MapTR's structured vector-map prediction but changes how efficiently it learns. It separates attention across map elements from attention within an element, gives each ground-truth element multiple positive training matches, and supervises depth and foreground geometry densely. It also distinguishes undirected boundaries from directed lane centerlines: reversing a boundary can preserve its meaning, while reversing a centerline can reverse the intended traffic flow. With ResNet-50 on nuScenes, the full recipe reaches 61.5 mAP in 24 epochs versus MapTR's 58.7 in 110; the advantage comes with substantially more training supervision and memory, rather than a uniformly cheaper model.

## Core Insights

### A useful symmetry depends on the element's meaning

MapTR's central idea is that equivalent point sequences should not create different learning targets. MapTRv2 retains the two allowed directions of an undirected polyline and the cyclic shifts and reversals of a closed polygon. But it gives a directed centerline only its annotated ordering. The same coordinates in reverse can encode an incorrect direction of travel, so treating every geometric reversal as equivalent would erase useful semantics.

Following LaneGAP, the centerline extension represents a lane graph through directed paths. This brings centerlines into the same instance-and-point query framework as crossings, dividers, and road boundaries. It does not make ordinary map AP a complete test of graph connectivity. The main three-class nuScenes benchmark and the four-class centerline experiment are separate settings: the latter reports 53.1 centerline AP and 54.0 mean AP, rather than the main experiment's 61.5.

### Factorized attention reduces the cost of maintaining structured queries

Each element has an instance embedding, and each sampled point has a shared point embedding. Adding them yields the hierarchical query used to locate that point in the map. Flattening all instance–point pairs into one sequence makes full self-attention expensive as either dimension grows. MapTRv2 instead separates interaction across instances from interaction among points within an instance.

![MapTRv2 Figure 4 compares structured attention and image-feature sampling choices](/assets/images/maptrv2-an-end-to-end-framework-for-online-vectorized-hd-map-construction-paper-figure.png)
*Fig 1: The middle panel separates communication between map elements from communication along each element. The right panel asks a different question: whether those queries retrieve visual evidence through a BEV map, projected camera views, or both. | source: [MapTRv2, Figure 4](https://arxiv.org/abs/2308.05736)*

The isolated attention ablation excludes the extra one-to-many matching branch. Under that setup, decoupled attention uses 8,458 MB instead of 10,443 MB and raises mAP from 57.1 to 57.6. At 125 instance queries, vanilla attention runs out of memory on the tested 24-GB GPU, while decoupled attention uses 10,378 MB. The practical gain is room for more structured predictions. It is not an inference-speed win in this comparison: FPS falls slightly from 14.7 to 14.1.

### More positive matches spend training memory to improve convergence

One-to-one set matching permits only one positive prediction for each ground-truth element. MapTRv2 retains that assignment for its inference branch, but adds a second group of instance queries during training. This auxiliary group shares point queries and decoder weights, while matching against repeated copies of the ground truth. More predictions therefore receive positive supervision for the same visible road geometry.

The default auxiliary branch uses 300 instance queries and six copies of each ground-truth element. In the matching ablation, adding this branch raises mAP from 57.6 to 61.5, while training memory rises from 8,458 to 19,426 MB. Inference remains 14.1 FPS across these settings because it uses the original prediction branch. Thus, the model separates how much supervision it can exploit during optimization from how many candidates it needs to emit at deployment.

Dense losses complement this assignment change. LiDAR points supply training depth targets; projected and rasterized map annotations supervise foreground segmentation in camera views and BEV. Camera-only inference therefore still benefits from LiDAR-derived supervision during training. With the matching recipe in place, the dense-loss ablation improves from 56.6 to 61.5 mAP when all three losses are used. The contribution is richer guidance for the features behind the sparse vector predictions.

![MapTRv2 Figure 7 compares validation accuracy across training epochs](/assets/images/maptrv2-an-end-to-end-framework-for-online-vectorized-hd-map-construction-source-figure-7.webp)
*Fig 2: MapTRv2 reaches 61.5 mAP at 24 epochs, above MapTR's 58.7 at 110. The curve demonstrates better convergence in epochs; the additional training branch means it should not be read as an equal reduction in training compute. | source: [MapTRv2, Figure 7](https://arxiv.org/abs/2308.05736)*

### Direct image sampling depends on the geometry available for supervision

The default encoder uses an LSS-based transformation with BEVPoolv2. Decoder queries sample the resulting BEV features through deformable cross-attention. A BEV-free alternative instead projects predicted reference points into camera views and samples image features directly. It removes a dense intermediate representation, but now the quality of each projection matters more directly.

On nuScenes, which supplies 2D map annotations without height, replacing BEV sampling with camera-view sampling reduces mAP from 61.5 to 49.5. Speed changes only from 14.1 to 14.4 FPS. On Argoverse2's 3D maps, the gap is smaller: 64.7 versus 59.1 mAP. Combining both feature sources improves Argoverse2 to 65.6 mAP but costs speed, reducing 12.0 to 10.0 FPS; on nuScenes it lowers accuracy to 60.5. The authors attribute the dataset difference to height supervision making projected reference points more accurate. This is evidence that a representation choice interacts with its supervision, rather than proof that adding another feature path always helps.

### Compare the full recipe under one evaluation protocol

| Camera-only model, as measured in the MapTRv2 paper | Epochs | nuScenes validation mAP | RTX 3090 FPS |
| --- | ---: | ---: | ---: |
| MapTR, ResNet-50 | 24 | 50.3 | 15.1 |
| MapTRv2, ResNet-50 | 24 | 61.5 | 14.1 |
| MapTR, ResNet-50 | 110 | 58.7 | 15.1 |
| MapTRv2, ResNet-50 | 110 | 68.7 | 14.1 |
| MapTRv2, ResNet-18 | 110 | 52.3 | 33.7 |

These timing measurements differ from the original MapTR paper's implementation measurements, so mixing the two papers' FPS rows would give a misleading speed comparison. Within this table, v2 improves accuracy substantially at slightly lower speed for the matched ResNet-50 configuration. Its 33.7-FPS result belongs to the smaller ResNet-18 model. The strongest camera-only result, 73.4 mAP at 9.9 FPS, uses a VoVNetV2-99 backbone initialized from DD3D, rather than the ImageNet initialization used by the ResNet models.

Calibration remains consequential: the rotation-noise experiment falls from 61.5 to 35.6 mAP at 0.02-radian Gaussian standard deviation across camera rotation coordinates. Better training and directed outputs do not establish robust localization under arbitrary sensor errors, nor does geometric AP establish closed-loop driving performance.

## High-Level Takeaways

- Preserve only meaning-preserving symmetries. Reversing an undirected boundary and reversing a directed lane centerline are different operations.
- Decoupled attention makes larger structured query sets more affordable in training; its measured benefit is primarily memory and accuracy, not higher FPS.
- One-to-many matching gives the shared decoder more positive supervision, then removes the extra prediction branch at inference. Faster convergence in epochs still consumes additional training memory.
- Dense depth and segmentation targets materially improve sparse vector predictions. Camera-only deployment should be distinguished from camera-only training supervision.
- BEV and direct camera-view sampling respond differently to available height supervision. Their accuracy and latency trade-offs need to be measured under the intended dataset and geometry.
