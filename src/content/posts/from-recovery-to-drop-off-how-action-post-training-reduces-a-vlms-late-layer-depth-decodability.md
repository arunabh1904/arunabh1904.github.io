---
title: "From Recovery to Drop-off: How Action Post-training Reduces a VLM's Late-Layer Depth Decodability"
date: '2026-08-14T00:00:00.000Z'
section: paper-shorts
postSlug: from-recovery-to-drop-off-how-action-post-training-reduces-a-vlms-late-layer-depth-decodability
legacyPath: /paper shorts/2026/08/14/from-recovery-to-drop-off-how-action-post-training-reduces-a-vlms-late-layer-depth-decodability.html
tags:
  - VLA
  - Mechanistic Analysis
  - Spatial Representation
field: 'Vision-Language-Action & Robotics'
summary: "2026 – From Recovery to Drop-off: How Action Post-training Reduces a VLM's Late-Layer Depth Decodability"
---

## 2026 – From Recovery to Drop-off: How Action Post-training Reduces a VLM's Late-Layer Depth Decodability

**arXiv:** [2608.08904](https://arxiv.org/abs/2608.08904)

## Summary

> This paper measures what action post-training does to a VLM's spatial representation rather than treating the resulting VLA as a black box. Using a weight-matched Molmo2-ER/MolmoAct2-LIBERO pair, it probes depth from every decoder layer and finds a persistent degradation floor plus a late-layer cliff. Matched causal ablations localize most of the cliff to late-layer MLP writes, not attention.

## Core Insights

The comparison is controlled at the backbone level: the base VLM and action-trained VLA share weights except for the post-training changes under study. A lightweight depth probe is trained at visual-token positions across all 36 layers. The VLA is worse at every depth, but the difference grows late: the reported mean depth score gap is 0.089 in early layers, 0.095 in middle layers, 0.166 in late layers, and 0.246 at the final layer.

The layer profile matters more than the aggregate drop. The base VLM's depth decodability improves toward its final layers, whereas the VLA's late-layer decodability falls. Ablating the late MLP writes recovers most of that terminal loss; matched attention ablations do not produce comparable recovery. Module decomposition points to accumulated MLP writes as the channel where the base model makes depth most accessible and action post-training overwrites it.


![Figure 2 from From Recovery to Drop-off: How Action Post-training Reduces a VLM](/assets/images/from-recovery-to-drop-off-how-action-post-training-reduces-a-vlms-late-layer-depth-decodability-source-figure-2.webp)
*Fig 1: Dense Prediction Transformer probing schematic. A LIBERO observation is fed to the VLM/VLA backbone; a capacity-matched DPT head decodes depth from the visual tokens at every decoder layer, supervised by a Depth-Anything-3 teacher. | source: [From Recovery to Drop-off: How Action Post-training Reduces a VLM](https://arxiv.org/abs/2608.08904)*

![Figure 1 from From Recovery to Drop-off: How Action Post-training Reduces a VLM](/assets/images/from-recovery-to-drop-off-how-action-post-training-reduces-a-vlms-late-layer-depth-decodability-source-figure-1.webp)
*Fig 2: The cliff, qualitatively. DPT-probe depth readouts of the same LIBERO observation (Obs.; Depth-Anything-3; Molmo2-ER DPT head, MolmoAct2-LIBERO DPT head) at the first and final decoder layer. | source: [From Recovery to Drop-off: How Action Post-training Reduces a VLM](https://arxiv.org/abs/2608.08904)*


The result does not mean that a VLA loses all geometry or that late MLPs are universally harmful. It establishes a causal failure mode for one weight-matched pair and one depth probe. The next question is whether action objectives can preserve useful spatial features through routing, auxiliary losses, or selective parameter updates without reducing control quality.

## High-Level Takeaways

- The paper informs whether action post-training should be audited layer-by-layer instead of evaluated only by downstream success.
- The atomic object is a visual-token representation at a decoder layer; the probe exposes which accumulated module writes carry depth information.
- The controlled evidence favors late MLP interference as the source of the cliff, while the broader floor likely reflects distributed changes that are not isolated here.
- A decisive follow-up would repeat the intervention across VLM/VLA families, tasks, and action heads while measuring both depth and control. The conclusion would weaken if the MLP-specific recovery does not generalize beyond Molmo.
