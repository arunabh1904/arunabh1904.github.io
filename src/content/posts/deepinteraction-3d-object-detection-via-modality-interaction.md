---
title: 'DeepInteraction: 3D Object Detection via Modality Interaction'
date: '2022-08-23T00:00:00.000Z'
section: paper-shorts
postSlug: deepinteraction-3d-object-detection-via-modality-interaction
legacyPath: /paper shorts/2022/08/23/deepinteraction-3d-object-detection-via-modality-interaction.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2022 – DeepInteraction: 3D Object Detection via Modality Interaction'
---
## 2022 – DeepInteraction

**arXiv:** [2208.11112](https://arxiv.org/abs/2208.11112)

**Code:** [fudan-zvg/DeepInteraction](https://github.com/fudan-zvg/DeepInteraction)

### Method and reported result

DeepInteraction challenges the assumption that fusion should collapse camera and LiDAR into one hybrid tensor. It maintains an image representation and a LiDAR BEV representation through the encoder, lets them update each other bidirectionally, and alternates object-query decoding against both streams.

## Summary

The paper's enduring question is what unification must preserve. A single coordinate frame simplifies reuse, but a single feature tensor can erase the neighborhood structure and failure profile that made each sensor complementary.

## Core Insights

The encoder has two jobs. Intra-modal learning improves each stream in its native representation. Multi-modal representational interaction projects and samples between image and LiDAR coordinates in both directions, so visual semantics can densify sparse geometry and geometry can sharpen image features. The decoder then performs predictive interaction: object queries alternate between the two representations rather than consulting one fused map.

The paper's ablations support the interaction claim. Using both representations in the decoder beats repeatedly using the LiDAR stream; adding cross-modal interaction improves over intra-modal processing alone; and the method improves its LiDAR-only baseline with both voxel and pillar backbones. Category gains are largest for some sparse or small classes, including 11.8 mAP for bicycles, 6.9 for motorcycles, and 5.9 for traffic cones in the reported nuScenes validation breakdown.

![Figure 1 from DeepInteraction, contrasting feature collapse with retained modality-specific representations](/assets/images/deepinteraction-paper-figure-1.png)
_The left pipeline fuses once; the right keeps both representations alive and exchanges information during encoding and decoding. Source: [DeepInteraction](https://arxiv.org/abs/2208.11112), Figure 1._

| Strategy | Benefit | Cost |
| --- | --- | --- |
| Single fused BEV | Simple shared interface | Can hide provenance and discard native structure. |
| Two retained streams | Preserves sensor-specific evidence | Carries more memory through the network. |
| Bidirectional encoder interaction | Improves both representations before prediction | Depends on calibrated cross-view sampling. |
| Alternating query decoder | Lets each object use geometry and semantics separately | Adds architectural coupling to the detector. |

## High-Level Takeaways

- DeepInteraction informs whether representation sharing should mean a shared tensor or a shared interaction protocol. Its atomic units are image and LiDAR features plus object queries. Parameters are not fully shared, and the two feature spaces remain explicit until prediction.
- The missing matched control gives a BEVFusion-style model the same parameter count, decoder depth, camera backbone, and latency, then tests corruption and calibration slices. At 10× temporal history, retaining two feature memories becomes expensive. The two-stream design would fail if a fused BEV with explicit sensor embeddings preserves the same robustness and rare-class gains with less memory, or if downstream dense tasks need a single scene tensor more than the detector needs native views.
- DeepInteraction is the strongest counterpoint to “unified representation” as feature collapse. Later unified backbones must show that parameter sharing does not erase modality-specific information.
- Sharing a model is not the same as forcing every sensor into one latent; sometimes the right shared object is the rule for exchanging evidence.
