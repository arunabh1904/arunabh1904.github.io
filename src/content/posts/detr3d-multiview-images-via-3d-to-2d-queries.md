---
title: 'DETR3D: 3D Object Detection from Multi-View Images via 3D-to-2D Queries'
date: '2021-10-14T04:00:00.000Z'
section: paper-shorts
postSlug: detr3d-multiview-images-via-3d-to-2d-queries
legacyPath: /paper shorts/2021/10/14/detr3d-multiview-images-via-3d-to-2d-queries.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2021 – DETR3D: 3D Object Detection from Multi-View Images via 3D-to-2D Queries'
---
## 2021 – DETR3D

**arXiv:** [2110.06922](https://arxiv.org/abs/2110.06922)

**Code:** [WangYueFt/detr3d](https://github.com/WangYueFt/detr3d)

## Summary

> DETR3D turns camera geometry into a retrieval operation: a 3D object hypothesis asks where its supporting pixels should appear. Each query projects a reference point into the surrounding cameras, gathers image features, and updates its box estimate. This avoids constructing a dense depth map or merging independent per-camera detections, but it does not remove the underlying ambiguity of distance. The strongest evidence is the gain where objects straddle cameras and the improvement from repeatedly revisiting image evidence.

## Core Insights

### Start with a 3D hypothesis and retrieve its image evidence

A conventional multiview extension of a monocular detector predicts boxes separately in each camera, then reconciles duplicates. DETR3D reverses that order. Its learned object queries are shared priors; each query decodes a candidate centre in metric 3D, projects it into all six cameras using known intrinsics and extrinsics, and samples the corresponding multiscale image features by bilinear interpolation. The scene’s object representation is shared before classification and box prediction, rather than assembled from competing camera outputs afterward.

The geometry is explicit while the evidence remains learned. A reference point says where to look, not that an object is already there. In the paper’s formulation, invalid image projections are masked and valid sampled features are averaged across cameras and scales, with a residual update to the query. Query self-attention models relationships between hypotheses. Repeated layers can move the reference point and retrieve different evidence as the box estimate improves.

![DETR3D architecture showing 3D reference points projected into multiview image features.](/assets/images/detr3d-multiview-images-via-3d-to-2d-queries-source-figure-1.webp)
*Fig 1: Trace the reference-point loop: decode a 3D location, project it into the cameras, retrieve image features, and refine the query. The matching diagram on the right assigns predictions jointly to objects across the scene. | source: [DETR3D, Figure 1](https://arxiv.org/abs/2110.06922)*

The six-layer head uses 256-dimensional queries and four FPN scales. Hungarian matching pairs the predicted set with ground-truth objects, combining classification and box regression losses. Extra predictions learn the no-object outcome. Every layer receives supervision, but inference uses the final layer. The one-to-one training assignment is what makes non-maximum suppression unnecessary; simply using a transformer would not establish that property.

### Camera overlap tests the value of sharing hypotheses early

The paper evaluates objects whose 3D centres are visible in multiple cameras: 18,147 validation boxes, or 9.7% of the total. This subset matters because a vehicle crossing an image boundary may look incomplete in either camera, while the two observations still support one coherent 3D hypothesis.

With FCOS3D backbone initialization, DETR3D reaches 26.8 mAP and 38.4 NDS in these overlap regions, compared with 22.9 and 32.9 for the compared strengthened FCOS3D configuration. Translation error improves only slightly, from 0.816 to 0.807 metres, while velocity error falls from 1.084 to 0.788 metres per second. The aggregate improvement therefore reflects more than just more accurate depth.

Initialization also matters outside the overlap subset. The basic validation DETR3D result is 30.3 mAP and 37.4 NDS; FCOS3D backbone initialization raises that to 34.6 and 42.5, and the CBGS configuration reaches 34.9 and 43.4. These are different recipes. Reporting only the best number would hide how much the shared image representation contributes before the new detection head is considered.

### Repeated retrieval helps more than indefinitely adding queries

The refinement ablation reads predictions from successive layers of the trained network. From layer 0 to layer 5, mAP improves from 30.2 to 34.6 and NDS from 38.0 to 42.5. Translation error falls from 0.855 to 0.773 metres. Most of the mAP gain arrives in the early layers, while later layers continue to refine attributes. This is an exit-layer comparison, rather than independently retrained networks with different depths.

![Full DETR3D source comparison of successive refinement layers against ground truth.](/assets/images/detr3d-source-figure-2-full-refinement.png)
*Fig 2: Compare the same scene across layers and against ground truth. Repeated sampling changes the box hypotheses as evidence accumulates. The displayed LiDAR points provide visual context only; DETR3D receives camera images. | source: [DETR3D, Figure 2](https://arxiv.org/abs/2110.06922)*

| Number of queries | mAP | NDS |
| ---: | ---: | ---: |
| 30 | 20.1 | 33.1 |
| 100 | 31.3 | 40.8 |
| 600 | 34.7 | 42.0 |
| 900 | 34.6 | 42.5 |
| 1,500 | 34.6 | 42.0 |

Too few hypotheses constrain the set, but more queries eventually stop helping. NDS peaks at 900 in this sweep, while mAP is slightly higher at 600. Query count is a capacity choice, not a monotonic accuracy knob. It also shifts the cost from maintaining a dense scene grid toward query interactions and feature retrieval; the image backbone itself remains dense.

### Removing a depth module does not solve depth uncertainty

On the test set, DETR3D with a DD3D-initialized backbone reaches 47.9 NDS versus DD3D’s 47.7, but its mAP is lower, 41.2 versus 41.8. Translation error is worse too, 0.641 versus 0.572 metres; velocity error is better, 0.845 versus 1.014. The paper’s headline score should therefore not be read as uniform superiority across detection attributes.

Its pseudo-LiDAR comparison is narrower than the architecture argument might suggest. The authors build a PackNet-depth-plus-CenterPoint baseline that reaches only 4.8 mAP, versus 30.3 for basic DETR3D. That result demonstrates failure of the implemented pipeline, rather than establishing that every depth-supervised or dense-lifting approach must lose. DETR3D avoids committing to a whole reconstructed scene, but its object centres still need to resolve where along the camera rays the objects lie.

The authors identify single-point sampling as a limitation: a centre projection provides limited direct support for the object’s extent and surrounding evidence. Sampling multiple adaptive points is a natural extension of this interface. The important abstraction is therefore the calibrated route from a 3D hypothesis back to image evidence, with room to improve what each query retrieves.

## High-Level Takeaways

- DETR3D shares object hypotheses across cameras before predicting boxes. The camera-overlap evaluation is its most direct comparative evidence for this design choice.
- Geometry enters as calibrated feature retrieval. The model avoids a dense depth intermediate while retaining uncertainty about object distance.
- Iterative refinement improves the same hypotheses substantially; increasing query count beyond the reported 600–900 region does not keep improving the metrics.
- Backbone initialization, CBGS, and the DD3D-based test recipe are distinct contributors. The best NDS result also has lower mAP and worse translation error than the compared DD3D result.
- Single-point support is the architecture’s useful limitation: extending what a query samples can enrich object evidence without abandoning the shared 3D query interface.
