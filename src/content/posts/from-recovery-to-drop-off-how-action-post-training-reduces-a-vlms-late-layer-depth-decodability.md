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

**arXiv:** [2608.08904](https://arxiv.org/abs/2608.08904)

## Summary

> This paper measures what action post-training does to a VLM's spatial representation rather than treating the resulting VLA as a black box. Using a weight-matched Molmo2-ER/MolmoAct2-LIBERO pair, it probes depth from every decoder layer and finds a persistent degradation floor plus a late-layer cliff. Matched causal ablations localize most of the cliff to late-layer MLP writes, not attention.

## Core Insights

### The damage is a layer profile, not one average score

The comparison is controlled at the backbone level: the base VLM and action-trained VLA share weights except for the post-training changes under study. A lightweight depth probe is trained at visual-token positions across all 36 layers. The VLA is worse at every depth, but the difference grows late: the reported mean depth score gap is 0.089 in early layers, 0.095 in middle layers, 0.166 in late layers, and 0.246 at the final layer.

The layer profile matters more than the aggregate drop. The base VLM's depth decodability improves toward its final layers, whereas the VLA's late-layer decodability falls. Ablating the late MLP writes recovers most of that terminal loss; matched attention ablations do not produce comparable recovery. Module decomposition points to accumulated MLP writes as the channel where the base model makes depth most accessible and action post-training overwrites it.

| Layer band | Molmo2-ER d1 | MolmoAct2-LIBERO d1 | Gap |
| --- | ---: | ---: | ---: |
| Early, L0–8 | 0.705 | 0.617 | 0.089 |
| Middle, L9–27 | 0.701 | 0.606 | 0.095 |
| Late, L28–35 | 0.744 | 0.578 | 0.166 |
| Final, L35 | 0.752 | 0.506 | 0.246 |

Across L28–L35, the base VLM rises by 0.040 while the VLA falls by 0.123. That inversion is the paper's useful diagnostic: a policy can retain a respectable middle-layer representation while action post-training specifically damages the layers where the base model consolidates depth.


![Figure 2 from From Recovery to Drop-off: How Action Post-training Reduces a VLM](/assets/images/from-recovery-to-drop-off-how-action-post-training-reduces-a-vlms-late-layer-depth-decodability-source-figure-2.webp)
*Fig 1: Dense Prediction Transformer probing schematic. A LIBERO observation is fed to the VLM/VLA backbone; a capacity-matched DPT head decodes depth from the visual tokens at every decoder layer, supervised by a Depth-Anything-3 teacher. | source: [From Recovery to Drop-off, Figure 2](https://arxiv.org/abs/2608.08904)*

![Figure 1 from From Recovery to Drop-off: How Action Post-training Reduces a VLM](/assets/images/from-recovery-to-drop-off-how-action-post-training-reduces-a-vlms-late-layer-depth-decodability-source-figure-1.webp)
*Fig 2: The cliff, qualitatively. DPT-probe depth readouts of the same LIBERO observation (Obs.; Depth-Anything-3; Molmo2-ER DPT head, MolmoAct2-LIBERO DPT head) at the first and final decoder layer. | source: [From Recovery to Drop-off, Figure 1](https://arxiv.org/abs/2608.08904)*


### A subtractive intervention localizes the cliff

The result does not mean that a VLA loses all geometry or that late MLPs are universally harmful. It establishes a causal failure mode for one weight-matched pair and one depth probe. In the symmetric three-layer-window sweep, removing the final VLA MLP writes raises final-layer d1 from 0.506 to 0.584 (+0.078); removing attention raises it only to 0.537 (+0.031). The corresponding VLM MLP intervention is +0.015 and VLM attention is −0.003. Because the intervention deletes writes and adds no new geometry, the result localizes interference rather than proving that a replacement representation has been learned.

### Depth is a probe, not a policy guarantee

The probe is a 41.8M-parameter DPT head trained at all 36 visual-token taps on rollout-strided 256-pixel LIBERO frames, with Depth-Anything-3 as affine-aligned pseudo-ground truth. That makes the curves useful for comparing the matched pair, but d1 measures agreement with a monocular teacher rather than metric depth. The paper explicitly does not claim that deleting late MLP writes improves closed-loop control. The next repair experiment must measure depth preservation and action success together across backbones and action objectives.

## High-Level Takeaways

- The paper informs whether action post-training should be audited layer-by-layer instead of evaluated only by downstream success.
- The controlled evidence favors late VLA MLP interference as the source of the cliff, while the broader floor likely reflects distributed changes that are not isolated here.
- The exact recovery is informative: 0.506→0.584 for the final VLA MLP window versus 0.506→0.537 for attention. It is a representation-level causal localization, not a claim that MLP deletion is a deployable policy fix.
- Depth-Anything-3 supplies pseudo-ground truth, and the study covers one weight-matched Molmo pair, one dataset, one probe class, and one seed. Those limits belong beside the striking number.
- A useful follow-up would preserve the late-layer depth signal during action tuning, then evaluate both the probe and closed-loop recovery across VLM/VLA families, tasks, and action heads.
