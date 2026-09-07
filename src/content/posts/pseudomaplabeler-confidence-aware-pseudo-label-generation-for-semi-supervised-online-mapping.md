---
title: "PseudoMapLabeler: Confidence-Aware Pseudo-Label Generation for Semi-Supervised Online Mapping"
date: '2026-08-12T00:00:00.000Z'
section: paper-shorts
postSlug: pseudomaplabeler-confidence-aware-pseudo-label-generation-for-semi-supervised-online-mapping
legacyPath: /paper shorts/2026/08/12/pseudomaplabeler-confidence-aware-pseudo-label-generation-for-semi-supervised-online-mapping.html
tags:
  - Autonomous Driving
  - HD Maps
  - Semi-Supervised Learning
field: 'BEV Perception & Mapping'
summary: "2026 – PseudoMapLabeler: Confidence-Aware Pseudo-Label Generation for Semi-Supervised Online Mapping"
---

**arXiv:** [2608.12600](https://arxiv.org/abs/2608.12600)

## Summary

> PseudoMapLabeler turns partially reliable teacher predictions into better training labels for online mapping. It accumulates map elements across a scene, estimates spatial confidence, clips uncertain portions of polylines, and uses the surviving geometry as a prior for a second teacher pass. A student learns from those improved pseudo-labels before labeled fine-tuning. With 16.5% of the training scenes labeled, the main nuScenes result improves from 21.5 to 27.6 mAP. The contribution is selective reuse of geometry, with confidence calibration and teacher bias still limiting what unlabeled data can supply.

## Core Insights

### Refine the teacher's input before trusting its output

The initial teacher is a Uni-PrevPredMap model trained on the small labeled subset. Its temporal priors already reuse previous predictions within a scene; the baseline is not deliberately deprived of temporal context. The new procedure runs the fixed teacher over unlabeled scenes, aligns the resulting polylines with ego poses, and constructs a refined scene-local map.

That map is not immediately treated as the final student label. It is rasterized and fed back into the teacher's BEV features and query initialization. A second pass produces new vector predictions, which supervise a separately trained student. The student is then fine-tuned on the original human-labeled subset. This distinction matters: clipping supplies partial geometric context, while the second teacher reconstructs the map elements used as training targets.

![Scene-level refinement produces a teacher prior before a second prediction pass trains the student](/assets/images/pseudomaplabeler-confidence-aware-pseudo-label-generation-for-semi-supervised-online-mapping-source-figure-1.webp)
*Fig 1: Follow the refined map into the teacher, then follow its output into the student loss. The clipped prior and the final pseudo-label are different artifacts; the teacher gets another chance to interpret the images. | source: [PseudoMapLabeler, Figure 1](https://arxiv.org/abs/2608.12600)*

Accumulation is offline within each 20-second nuScenes scene. The method does not require one globally consistent map covering all locations, and temporal priors reset between scenes. Using a scene's repeated observations to generate training labels should not be mistaken for an online controller having access to future observations.

### Confidence should vary along a line

The model temperature-scales prediction logits, then accumulates weighted detections on a 0.5 m grid separately for each map class. A Beta prior with mean 0.2 and strength 2 regularizes cells with little evidence. The posterior mean combines calibrated detection scores with the number of frames whose BEV extent covers the cell.

The intuition is that one confident fragment and many repeated detections should not carry identical evidence. But coverage is defined by the configured BEV extent, not a complete visibility model, and nearby frames are correlated. This confidence field measures accumulated support from the teacher; it does not independently establish that the map is correct.

![Per-class confidence maps remove unreliable parts of accumulated vector predictions](/assets/images/pseudomaplabeler-confidence-aware-pseudo-label-generation-for-semi-supervised-online-mapping-source-figure-2.webp)
*Fig 2: Noisy overlapping predictions become class-specific confidence fields. Spatial clipping preserves supported stretches while cutting away uncertain ends and intersections, instead of accepting or rejecting every polyline as a whole. | source: [PseudoMapLabeler, Figure 2](https://arxiv.org/abs/2608.12600)*

For polylines, the method samples confidence along the points and keeps sufficiently long contiguous segments above a class-specific percentile threshold. Here p=30 means retaining the top 30% of sampled confidence values within each class. Pedestrian-crossing polygons follow a stricter rule: every point must pass, so a partial crossing polygon is not emitted as a valid element. The segment-level argument therefore applies directly to line classes, with a separate polygon policy.

### Better pseudo-labels and a better student are separate results

The geographically disjoint training split contains 115 labeled scenes, 4,636 samples, and 581 pseudo-unlabeled scenes, 23,332 samples. Their hidden labels permit a diagnostic evaluation of pseudo-label quality. On that subset, the teacher without refined priors reaches 23.4 mAP; p=20 gives 26.4 and p=30 gives 26.2. The authors choose p=30 because raster Dice/IoU is slightly higher, favoring geometric coverage. Ground-truth priors give 36.2, leaving substantial headroom.

The final student comparison uses the separate nuScenes validation set:

| Student | Labeled only | Whole-element filtering | Spatial clipping |
| --- | ---: | ---: | ---: |
| Uni-PrevPredMap | 21.5 | 24.8 | 27.6 |
| MapTR | 16.5 | 20.1 | 21.5 |

The main 6.1-point gain combines pseudo-label training and refinement; the cleaner comparison for spatial clipping itself is 27.6 versus 24.8. Transfer to a MapTR student supports portability of the generated labels. It does not establish that any arbitrary teacher can consume the refined prior without an appropriate interface.

### Repeated confidence can preserve repeated mistakes

The retention sweep is not monotone: keeping the top 50% reduces pseudo-label mAP to 20.7, below the no-prior result. Expanding coverage admits enough unreliable geometry to harm the second pass. Accurate ego poses and a teacher with sufficient initial signal are essential dependencies.

The calibration protocol also deserves attention. Temperature is fitted on the validation set, and the retention setting is selected using pseudo-unlabeled ground truth. Although temperature scaling preserves raw confidence ranking, its values feed the confidence maps and later training labels. A stronger untouched evaluation would reserve separate calibration and selection data before testing the final student. The current result supports the reported semi-supervised recipe, with cross-region calibration and severe teacher-domain shift still open.

## High-Level Takeaways

- Preserve reliable portions of a polyline when whole-element filtering would discard useful geometry.
- Keep refined priors, second-pass pseudo-labels, and final student scores distinct; they measure different stages.
- The filtering-versus-clipping comparison isolates refinement more directly than the total gain over labeled-only training.
- Temporal agreement is useful evidence, but teacher bias, pose error, and validation-based selection remain part of the result's interpretation.
