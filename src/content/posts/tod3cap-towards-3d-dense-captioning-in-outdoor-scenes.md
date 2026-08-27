---
title: 'TOD3Cap: Towards 3D Dense Captioning in Outdoor Scenes'
date: '2024-03-28T00:00:00.000Z'
section: paper-shorts
postSlug: tod3cap-towards-3d-dense-captioning-in-outdoor-scenes
legacyPath: /paper shorts/2024/03/01/tod3cap-towards-3d-dense-captioning-in-outdoor-scenes.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2024 – TOD3Cap: Towards 3D Dense Captioning in Outdoor Scenes"
---
## 2024 – TOD3Cap

**arXiv:** [2403.19589](https://arxiv.org/abs/2403.19589)

### Method and reported result

TOD3Cap turns outdoor driving scenes into a dense captioning problem. Given multi-sensor input, a model must localize objects in 3D and describe each one with useful language.

## Summary

> That is harder than standard object detection because it requires attributes, context, and grounded descriptions, not just boxes and class IDs.

## Core Insights

TOD3Cap introduces outdoor 3D dense captioning: localize objects in 3D scenes and generate grounded descriptions for them. The dataset contains 850 scenes, 64.3k objects, and 2.3M captions, making it much larger and more driving-relevant than small indoor captioning setups. The task requires geometry, object detection, and language generation together. It matters for autonomous driving because planners and assistants need grounded object descriptions, not just boxes. The caveat is evaluation: a fluent caption can still miss safety-critical geometry.

![Figure 1: We introduce the task of 3D dense captioning in outdoor scenes (right) from TOD3Cap: Towards 3D Dense Captioning in Outdoor Scenes](/assets/images/tod3cap-towards-3d-dense-captioning-in-outdoor-scenes-paper-figure.png)
*Fig 1: We introduce the task of 3D dense captioning in outdoor scenes (right). | source: [TOD3Cap: Towards 3D Dense Captioning in Outdoor Scenes paper](https://arxiv.org/abs/2403.19589)*

![Figure 3 from TOD3Cap: Towards 3D Dense Captioning in Outdoor Scenes](/assets/images/tod3cap-towards-3d-dense-captioning-in-outdoor-scenes-source-figure-3.webp)
*Fig 2: Architecture of our proposed Cap network. Firstly, BEV features are extracted from 3D LiDAR point cloud and 2D multi-view images, followed by a query-based detection head that generates a set of 3D object proposals from the BEV features. | source: [TOD3Cap: Towards 3D Dense Captioning in Outdoor Scenes](https://arxiv.org/abs/2403.19589)*

![Figure 13 from TOD3Cap: Towards 3D Dense Captioning in Outdoor Scenes](/assets/images/tod3cap-towards-3d-dense-captioning-in-outdoor-scenes-source-figure-13.webp)
*Fig 3: Qualitative driving-QA examples combine surround-view images and BEV detections to answer object-status and relation questions with matching ground-truth and predicted responses. | source: [TOD3Cap: Towards 3D Dense Captioning in Outdoor Scenes](https://arxiv.org/abs/2403.19589)*


**What to look at:**
- The task combines 3D localization with object-level captions.
- LiDAR plus RGB fusion matters because captions are grounded to 3D boxes.
- This is a perception-to-explanation benchmark.

### Reported evidence

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Dataset | 850 scenes / 64.3k objects / 2.3M captions | Large outdoor dense-captioning setup. |
| Task | 3D object captioning | Requires boxes plus descriptions. |
| Use case | Scene explanation and planner context | Turns perception into grounded language. |

## High-Level Takeaways

- TOD3Cap informs whether outdoor perception should stop at 3D boxes or attach language descriptions that expose object attributes and relations. The atomic output is a detected 3D instance paired with a caption, so localization and language quality are jointly constrained by the same scene.
- Dense captions can support open-ended reasoning, but caption metrics may reward generic descriptions and ignore metric grounding. The missing study conditions on oracle versus predicted boxes and evaluates whether captions improve a downstream driving decision, not only language similarity. At 10× objects, proposal-caption pairing and annotation consistency dominate. The task formulation would fail if richer detection attributes delivered the same downstream utility with less free-form language ambiguity.
- Dense captioning is a bridge between perception and explanation. A driving system that can say what every relevant object is doing has a better interface to planners, annotators, and safety reviewers.
- Rich scene understanding requires language that is spatially grounded. Captions without 3D grounding are not enough for driving.
