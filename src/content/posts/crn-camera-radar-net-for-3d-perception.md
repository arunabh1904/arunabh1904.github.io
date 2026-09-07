---
title: 'CRN: Camera Radar Net for 3D Perception'
date: '2023-04-03T04:00:00.000Z'
section: paper-shorts
postSlug: crn-camera-radar-net-for-3d-perception
legacyPath: /paper shorts/2023/04/03/crn-camera-radar-net-for-3d-perception.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2023 – CRN: let radar guide camera lifting before BEV fusion'
---
## 2023 – CRN

**arXiv:** [2304.00670](https://arxiv.org/abs/2304.00670)

**Code:** [youngskkim/CRN](https://github.com/youngskkim/CRN)

## Summary

> Camera Radar Net uses radar twice: first to help place image features at the right distance, then to reconcile remaining camera–radar disagreement in bird’s-eye view. Sparse range measurements influence the construction of the camera representation before fusion tries to repair it. The paper also exposes the limits: radar cannot replace dense image depth, and faster sparse fusion loses more than classification accuracy alone reveals.

## Core Insights

### Radar supplies a second route into depth, rather than replacing the first

A camera can recognize a distant vehicle while remaining uncertain about its distance. Radar measures range and motion, but its returns are sparse, can include clutter, and lack reliable elevation. CRN keeps the camera’s dense depth prediction and adds a learned radar occupancy signal alongside it.

The camera branch predicts image context and a categorical depth distribution. The radar branch projects points into each camera’s frustum and encodes pillars indexed by image width and depth. Height is collapsed because radar lacks dependable elevation. Its occupancy head uses a sigmoid, allowing multiple occupied distances along a column; a softmax would force them to compete for one probability budget.

Radar-assisted view transformation constructs two versions of image context: one weighted by predicted depth and another weighted by radar occupancy. A convolution combines them before transformation into BEV. A missing radar return therefore does not erase the camera’s depth pathway, while useful returns can sharpen where semantic evidence enters metric space. LiDAR supplies projected depth supervision during training; deployed inputs are cameras and radar.

![CRN architecture showing radar-assisted view transformation and multimodal feature aggregation.](/assets/images/crn-camera-radar-net-for-3d-perception-paper-figure.webp)
*Fig 1: Follow the two radar paths: occupancy helps lift camera features, while radar context remains available for later BEV fusion. Depth supervision trains the camera branch but is not an inference input. | source: [CRN, Figure 2](https://arxiv.org/abs/2304.00670)*

Table 5 isolates view transformation by withholding point features from later aggregation. Camera depth alone gives 33.2 mAP; replacing it with binary radar occupancy falls to 24.3. Combining depth with learned radar assistance reaches 44.8 mAP and reduces translation error from 0.716 to 0.521 metres. Radar should guide dense lifting without becoming its sole geometric support.

Even pooling contains a geometric choice. CRN averages frustum features landing in each BEV cell instead of summing them. Nearby cells receive more frustum contributions because of perspective. Averaging limits that distance-dependent change in magnitude, so a larger response need not merely mean more samples landed there.

### Fusion searches around a location instead of assuming perfect agreement

After lifting, camera evidence may remain smeared along a depth ray, while radar may return clutter or locate an object away from its visual centre. Concatenating identical BEV coordinates asks a small convolution to resolve these cases through a fixed neighbourhood.

CRN’s multimodal deformable attention predicts separate sampling offsets and weights for camera and radar. Each query looks around its reference location differently in each modality, then weights the sampled evidence jointly. The implementation uses six fusion layers, eight heads, and four sampling points. This is learned local correspondence, rather than evidence that the modalities agree geometrically before fusion.

![Fused, image, and radar feature maps around occluded vehicles and radar clutter.](/assets/images/crn-camera-radar-net-for-3d-perception-source-figure-4.webp)
*Fig 2: Compare fusion, image, and radar columns at the marked locations. Camera features miss occluded and distant vehicles; radar introduces wall clutter and weak pedestrian evidence. Fusion combines useful responses while suppressing some misleading ones. | source: [CRN, Figure 4](https://arxiv.org/abs/2304.00670)*

Table 6 shows overlapping benefits. BEVFusion-style convolution reaches 42.4 mAP; deeper convolutions reach 42.8, and adding radar-assisted transformation reaches 44.3. Deformable aggregation alone reaches 44.5, with both CRN components reaching 45.2. Better lifting and adaptive fusion partly solve the same localization problem, explaining why their gains are not additive. These aggregation experiments differ from Table 5 and should not be combined into one cumulative sequence.

### Sparse fusion spends less context on object attributes

The long-range configuration expands BEV to 256 × 256, or 65,536 locations. CRN can select top-ranked queries using camera-depth and radar-occupancy confidence. Sparse-mode training also keeps only foreground LiDAR depth targets inside object boxes and changes the camera depth head to a sigmoid. Selection is trained to favour object-bearing regions, rather than applied as arbitrary pruning to the ordinary model.

With 4,096 queries, fusion latency falls from 21.01 to 4.96 ms. That 76.4% reduction concerns the fusion module; total throughput rises from 11.5 to 14.0 FPS because the rest of the detector still costs time.

| Long-range car evaluation | All queries | 4,096 queries |
| --- | --- | --- |
| AP, higher is better | 56.9 | 54.0 |
| Translation error, metres | 0.325 | 0.367 |
| Orientation error, radians | 0.158 | 0.194 |
| Velocity error, metres/second | 0.298 | 0.340 |
| End-to-end throughput | 11.5 FPS | 14.0 FPS |

The authors suggest classification tolerates sparse features better than regression, which needs surrounding evidence for position, heading, and velocity. Preserving much of AP does not preserve the quality of the boxes and motion estimates. Increasing to 8,192 queries raises AP to 54.6, but velocity error becomes 0.352: every attribute does not improve monotonically with query count.

### The strongest numbers describe different operating conditions

The real-time ResNet-50 model with 256 × 704 images reaches 49.0 mAP and 56.0 NDS on nuScenes validation at 20.4 FPS. The 57.5 mAP and 62.4 NDS test result uses the larger ConvNeXt-B model. Timing uses batch size one, FP16, and an RTX 3090. Previous BEV features are cached, so temporal input adds work to the BEV head without re-encoding historical images. Radar sweeps are also accumulated; these results do not describe an isolated image and instantaneous radar scan.

For long-range cars, CRN reaches 7.0 AP at 60–100 metres versus 4.8 for the compared LiDAR CenterPoint model—a relative gain from a low absolute level. This experiment doubles class evaluation ranges and disables usual points-in-box filtering. The appendix notes resulting inconsistencies between visible objects and annotations. It supports camera–radar fusion at distance under that protocol, rather than general LiDAR replacement.

![CRN long-range detections in day, rain, and night, with bottom-row failures highlighted.](/assets/images/crn-camera-radar-net-for-3d-perception-source-figure-7.webp)
*Fig 3: Day, rain, and night examples use green ground-truth boxes, blue predictions, and black radar points. Inspect the bottom-row red circles too: rare or occluded objects without useful radar returns remain difficult. | source: [CRN, Figure 7](https://arxiv.org/abs/2304.00670)*

CRN improves night mAP from the reproduced BEVDepth baseline’s 16.8 to 30.4, but remains below its own daytime 55.1. Under complete radar dropout, it retains 43.8 car AP versus BEVFusion’s 34.4; under complete camera dropout, it reaches 12.8 versus BEVFusion’s 14.3. Adaptive fusion improves several failure cases without making both sensors dispensable.

## High-Level Takeaways

- Preserve dense camera depth while giving sparse radar measurements another way to position visual evidence. Replacing depth outright performs substantially worse.
- Radar-assisted lifting and deformable fusion repair overlapping errors. Their isolated ablations explain the mechanism more clearly than attributing the full camera-baseline gain to either component.
- Query budget changes geometry and motion accuracy as well as AP. The 4,096-query configuration saves fusion time while worsening translation, orientation, and velocity errors.
- The 20.4 FPS validation model, larger test model, and extended-range experiment are separate configurations. Their best numbers do not describe one operating point.
- Radar helps at night and under some sensor failures, but camera dropout remains severe. The source figures show where complementary sensing still leaves missing evidence.
