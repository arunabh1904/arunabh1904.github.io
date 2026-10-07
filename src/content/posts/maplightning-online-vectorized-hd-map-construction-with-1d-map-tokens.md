---
title: "MapLightning: Online Vectorized HD Map Construction with 1D Map Tokens"
date: '2026-10-01T09:00:00.000Z'
section: paper-shorts
postSlug: maplightning-online-vectorized-hd-map-construction-with-1d-map-tokens
legacyPath: /paper shorts/2026/10/01/maplightning-online-vectorized-hd-map-construction-with-1d-map-tokens.html
tags: ["Mapping", "Transformers", "Autonomous Driving"]
field: "BEV Perception & Mapping"
summary: "2026 \u2013 MapLightning: Online Vectorized HD Map Construction with 1D Map Tokens"
---

## 2026 – MapLightning: Online Vectorized HD Map Construction with 1D Map Tokens

**Paper:** [arXiv:2610.01905](https://arxiv.org/abs/2610.01905) · [Full text](https://arxiv.org/html/2610.01905v1)

## Summary

> MapLightning replaces a dense BEV intermediate with learned map tokens. Its strongest lesson for adapter design is that token compression needs an explicit interaction mechanism; reducing the token count alone is not enough.

## Core Insights

MapLightning jointly updates image tokens and learned map tokens with four self-attention layers. It then discards the image tokens and decodes vector map elements from the retained tokens. Learned row, column, camera, and time embeddings describe the image inputs. The mapper uses no camera calibration or explicit ego-pose alignment. Four timesteps yield 1,200 map tokens on six-camera nuScenes. The default decoder has 50 instance queries, 20 points per element, and six layers.

On the geography-separated Near-Extrapolation split, the ResNet-50 system reports 35.3 mAP on nuScenes and 63.3 on Argoverse 2. The corresponding MapTRv2 scores are 26.7 and 53.1. With one frame and a common decoder, self-attention scores 30.1 on nuScenes versus 25.5 for cross-attention. Training uses map classification, point, and direction losses; depth and segmentation auxiliary losses are removed. Reported throughput is 18.9 FPS on RTX 3090 for the default system.

The diagram asks where information can move before compression. In the self-attention branch, image tokens can exchange evidence before the compact map representation is read out.

![Source Figure 1 compares dense BEV projection, cross-attention, and joint self-attention for map construction.](/assets/images/maplightning-source-figure-1.png)

*Fig 1: Source Figure 1, cropped from the PDF. Joint attention updates both token groups before the image tokens are discarded; the retained map tokens are learned slots. | source: [MapLightning](https://arxiv.org/abs/2610.01905)*

### A learned slot is not a grid cell

A BEV cell has a declared support region in metric space. A learned map token has no such guarantee. Its index identifies a storage slot, not a road coordinate. This distinction remains true when the number of slots happens to factor into a convenient rectangle.

Consider a bend that crosses several camera images. A grid-based pipeline assigns evidence to metric locations before decoding. A slot-based pipeline can gather evidence from several images into one token. That token may describe part of the bend, a relation between boundaries, or another useful mixture. A downstream language adapter must not attach a made-up row and column to that mixture and call it physical position.

The result complements the [mapping guide](/blog/2026/10/01/from-bev-features-to-lane-graphs-and-changing-maps.html): the output can remain geometric even when the intermediate representation is not a dense metric grid. Geometry can be imposed by the prediction head and its supervision. Whether that representation is suitable for another task remains a separate question.

### Separate representation cost from total cost

A smaller retained sequence reduces the work of downstream decoders. It does not make the preceding full attention free. The attention block still processes the combined input sequence. If more cameras, higher resolution, or longer history increase that sequence, the mapper can become the dominant cost.

For a new system, measure encoder time, mapper time, language-model prefill, and decoding separately. Compare the same scene coverage and image resolution. Count retained tokens and temporary tokens. This avoids describing a compression ratio as an equal reduction in end-to-end latency.

The calibration perturbation test also needs a narrow interpretation. A model that never consumes calibration values cannot respond to corrupted calibration values. That does not prove robustness to a camera physically moving, a new camera layout, or a different lens. My proposed transfer test changes the images and camera rig together, then measures geometry error on unseen roads.

## High-Level Takeaways

- Use joint token refinement when information must move between input regions before compression.
- Keep learned slots distinct from metric grid cells in every downstream adapter.
- Evaluate map generalization on held-out geography, not only different frames from familiar roads.
- Reject an efficiency claim that reports only retained token count without total latency and memory.
