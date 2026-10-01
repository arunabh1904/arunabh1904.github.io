---
title: "P-MapNet: Far-seeing Map Generator Enhanced by both SDMap and HDMap Priors"
date: '2024-03-15T00:00:00.000Z'
section: paper-shorts
postSlug: p-mapnet-far-seeing-map-generator-enhanced-by-sdmap-and-hdmap-priors
legacyPath: /paper shorts/2024/03/15/p-mapnet-far-seeing-map-generator-enhanced-by-sdmap-and-hdmap-priors.html
tags: ["Autonomous Driving", "HD Maps", "Topology"]
field: Mapping
summary: "2024 – P-MapNet: Far-seeing Map Generator Enhanced by both SDMap and HDMap Priors"
---

# 2024 – P-MapNet: Far-seeing Map Generator Enhanced by both SDMap and HDMap Priors

**Paper:** [2403.10521](https://arxiv.org/abs/2403.10521)

**Project:** [P-MapNet](https://jike5.github.io/P-MapNet)

## Summary

> P-MapNet separates two priors: an SD map of the current location conditions BEV perception, while a masked autoencoder learns a distribution over HD-map shapes and refines the prediction. On nuScenes at 240 × 60 m, the camera-only model raises mIoU from 26.77 to 42.20 with the SD prior and to 45.50 with both priors. Refinement reduces throughput from 19.2 to 9.1 FPS in that setting. The appendix also shows a concrete failure: a prior emphasizing main roads can suppress real nearby forks.

## Core Insights

### The two priors answer different questions

An SD map supplies location-specific road structure. The learned HD-map prior supplies regularities about what a detailed map tends to look like. P-MapNet combines these roles without requiring a prebuilt HD map for the current location at inference.

The main pipeline is raster-based. Cameras, optionally combined with LiDAR, produce BEV features. An SD-map branch conditions those features before a segmentation head predicts dividers, crossings, and boundaries. The optional HD-map refinement module then turns that initial prediction into another semantic map. Vector outputs for the main system are obtained through post-processing; a separate experiment inserts the SD branch into [MapTR](/paper%20shorts/2022/08/30/maptr-structured-modeling-and-learning-for-online-vectorized-hd-map-construction.html).

The source architecture locates the two commitments. The SD branch adds context before prediction; the refinement stage adds another model after prediction.

![P-MapNet source Figure 2 shows SD-map cross-attention followed by optional masked-autoencoder HD-map refinement](/assets/images/pmapnet-source-figure-2.png)
*Fig 1: The SD prior conditions perception with the current road skeleton. The HD prior is learned from masked training maps and refines the resulting raster prediction, adding a separate inference cost. | source: [P-MapNet, Figure 2](https://arxiv.org/abs/2403.10521)*

### Attention handles approximate alignment; reconstruction learns completion

OpenStreetMap geometry and GPS pose are only weakly aligned with sensor BEV features. P-MapNet downsamples the BEV representation and uses its queries to attend to encoded raster-map tokens. This allows feature retrieval across positions rather than requiring pixelwise concatenation to be correct.

At 120 × 60 m with camera–LiDAR inputs, the fusion ablation reports 54.73 mIoU for simple concatenation, 56.27 for CNN-encoded concatenation, and 60.20 for cross-attention. Image/BEV self-attention without the map also improves the baseline, reaching 53.43. Some benefit therefore comes from added feature processing, but the map-conditioned model provides a further gain.

For the HD prior, a ViT and segmentation head reconstruct clean semantic maps from masked training maps using pixelwise cross-entropy. The model is pretrained for 20 epochs, then attached to the sensor-conditioned predictor for ten epochs of joint fine-tuning. This differs from copying local HD-map geometry into the output: the prior lives in learned reconstruction behavior.

### Longer range increases the value of context and the cost of refinement

The following camera-only results come from the same nuScenes table and the 240 × 60 m range. The baseline and SD-only stages use 30 training epochs; the combined model adds the refinement stage and fine-tuning.

| Configuration | Raster mIoU | Post-processed vector mAP | FPS |
| --- | ---: | ---: | ---: |
| HDMapNet baseline | 26.77 | 11.35 | 22.3 |
| P-MapNet, SD prior | 42.20 | 16.38 | 19.2 |
| P-MapNet, SD + HD priors | 45.50 | 22.75 | 9.1 |

The SD prior accounts for most of the raster gain. The learned refinement improves quality further while roughly halving throughput relative to SD-only. These experiments use nuScenes and Argoverse 2, with four RTX 3090 GPUs for training. The reported profiling is also on an RTX 3090. Their segmentation and Chamfer-based map metrics do not directly evaluate directed lane connectivity.

In the separate direct-vector experiment, adding SD-map conditioning to MapTR raises long-range mAP from 8.03 to 16.53. At the shorter 60 × 30 m range, the corresponding table values are 47.25 and 49.37. The differing gains support the range argument without implying that P-MapNet's raster pipeline is the best choice for every output format.

### A more regular map can lose a real road

The appendix's negative example is essential to interpreting the refinement stage. The sensor-only baseline detects nearby forks that are absent or weakly represented in the SD map. Adding the prior strengthens the main road but weakens those branches; refinement can smooth the prediction further while preserving the mistake.

![P-MapNet source Figure 8 shows near-side forks disappearing as SD-map and learned HD-map priors are added](/assets/images/pmapnet-source-figure-8.png)
*Fig 2: The nearby forks expose a conflict between observations and prior structure. Better long-range completion does not ensure that local branches survive, even when the final map looks more regular. | source: [P-MapNet, Figure 8](https://arxiv.org/abs/2403.10521)*

Including service roads in OSM does not automatically fix the issue. Those roads are often absent from benchmark annotations, so the model can learn to suppress them as noise. In the category ablation, including service roads gives 58.53 mIoU versus 60.20 without them. Dataset consistency can therefore reward removing real-world detail.

My decision rule would choose the SD-only model first under a tight runtime budget and require a separate false-branch-removal audit before adding refinement. The unresolved question is whether the learned map distribution improves the actual road graph, rather than merely its agreement with a dataset's labeling conventions.

## High-Level Takeaways

- Distinguish a location-specific SD-map input from a learned distribution over HD-map geometry; P-MapNet uses both at different stages.
- Most long-range raster improvement comes from SD conditioning, while optional reconstruction adds quality and substantial latency.
- Segmentation quality and visual regularity do not establish correct lane connectivity.
- Missing forks and service-road annotation conflicts make observation-versus-prior disagreement a necessary evaluation slice.
