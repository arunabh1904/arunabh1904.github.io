---
title: "OpenLane-V2: A Topology Reasoning Benchmark for Unified 3D HD Mapping"
date: '2023-04-20T00:00:00.000Z'
section: paper-shorts
postSlug: openlane-v2-a-topology-reasoning-benchmark-for-unified-3d-hd-mapping
legacyPath: /paper shorts/2023/04/20/openlane-v2-a-topology-reasoning-benchmark-for-unified-3d-hd-mapping.html
tags: [Autonomous Driving, Mapping]
topics: [autonomy]
field: Mapping
summary: "2023 \u2013 OpenLane-V2: A Topology Reasoning Benchmark for Unified 3D HD Mapping"
---

# 2023 – OpenLane-V2: A Topology Reasoning Benchmark for Unified 3D HD Mapping

**Paper:** [2304.10440](https://arxiv.org/abs/2304.10440)

**Data and evaluation:** [official repository](https://github.com/OpenDriveLab/OpenLane-V2)

## Summary

> OpenLane-V2 turns road understanding into joint prediction of directed lane centerlines, traffic elements, and their relationships. The original benchmark contains 2,000 scene segments derived from Argoverse 2 and nuScenes. Its baselines show that geometric map prediction alone transfers poorly to directed connectivity: the adapted MapTR reaches 20.0 OLS on subset_A, while the topology-specific TopoNet reaches 35.4 in this paper. Later lane-segment tasks and metric revisions must be distinguished from this original result.

## Core Insights

### The labeled object is a directed route through the lane

A visible laneline marks a boundary; a centerline describes the intended direction through a lane, including where no paint exists. OpenLane-V2 derives centerlines from existing HD-map boundaries and merges unambiguous intermediate segments. Intersections, forks, and merges retain separate instances. Ordered points encode driving direction, and an adjacency matrix connects each lane's end to its successors' starts.

The source overview follows a vehicle toward a junction. Geometry becomes useful only when the lane graph and traffic-control associations explain which continuation the vehicle should consider.

![OpenLane-V2 source Figure 1 illustrates centerlines, direction, traffic elements, and relations through a junction](/assets/images/openlane-v2-source-figure.png)
*Fig 1: The scene representation adds directed lane connectivity and lane-control associations to geometric perception. A nearby traffic light is useful only when the model identifies the lanes it governs. | source: [OpenLane-V2, Figure 1; figure crop](https://arxiv.org/abs/2304.10440)*

Subset_A inherits Argoverse 2's seven-camera scenes from six US cities; subset_B inherits nuScenes' six-camera scenes from Boston and Singapore. Each subset has 700/150/150 train/validation/test scene segments, sampled at 2 Hz over a 100 × 50 m annotation region. Calibration and ego poses accompany the images. Subset_B's map heights are set to zero because its source maps lack elevation; the shared “3D” task should not obscure that distinction.

### Human annotation supplies the regulatory relation

Traffic elements are front-camera 2D boxes with thirteen attributes, including signal colors and directional permissions or prohibitions. Composite meanings are decomposed into multiple attribute records sharing a box. Unknown semantics receive an unknown attribute rather than an invented interpretation. Experienced annotators and multiple validation stages add and check these labels and their lane associations.

Lane–traffic associations follow the benchmark's rules: directional controls are assigned to compatible movements, and traffic elements are associated with lanes outside intersections. The paper's convention for non-directional controls is narrower than a complete traffic-law model. An annotation policy is therefore part of the learned relationship, not merely a neutral transcription of proximity.

The original task does not explicitly provide every stop-line, curb-type, temporary-restriction, or lane-change-permission output a production map might need. The official repository later introduces lane segments, a Map Element Bucket, and SD-map inputs. Those extensions broaden the contract but should not be retroactively attributed to the original centerline benchmark.

### Score geometry before relations, and preserve evaluation versions

Lane detection uses discrete Fréchet distance, which respects point order, with nominal 1, 2, and 3 meter thresholds adjusted by range. Traffic detection uses boxes and attribute AP. Relationship evaluation first matches predicted entities to ground truth, then scores neighbor predictions on the aligned graph. Missed lane instances consequently affect topology as well as detection.

The original OLS averages lane detection, traffic detection, and scaled lane–lane and lane–traffic topology components. It is not an ordinary mean of four raw percentages. On subset_A, adapted STSU scores 25.4, adapted MapTR 20.0, and TopoNet 35.4 in the benchmark paper. These baselines are retrained with added task heads where necessary; their original mapping scores are not directly comparable.

The source repository documents subsequent metric and task changes. [TopoLogic](/paper%20shorts/2024/05/23/topologic-an-interpretable-pipeline-for-lane-topology-reasoning-on-driving-scenes.html) and [LaneSegNet](/paper%20shorts/2023/12/26/lanesegnet-map-learning-with-lane-segment-perception-for-autonomous-driving.html) should therefore be read with their stated evaluation versions. The original scene split also does not guarantee geographic separation; later mapping work exposes that source of optimistic generalization. A useful follow-up would hold geometry detections fixed while comparing association methods, then separately measure how missed lanes limit the graph.

## High-Level Takeaways

- Ordered centerlines and typed edges make direction and control assignment measurable.
- Source-map processing and human association rules define what counts as a correct graph.
- Missing entities limit topology before any relationship head runs.
- Preserve subset, ontology, split, and evaluator version when comparing OLS, OLUS, or topology numbers.
