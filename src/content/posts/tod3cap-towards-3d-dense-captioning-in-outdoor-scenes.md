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

**arXiv:** [2403.19589](https://arxiv.org/abs/2403.19589) · **Project:** [TOD3Cap](https://jxbbb.github.io/TOD3Cap) · **Code:** [jxbbb/TOD3Cap](https://github.com/jxbbb/TOD3Cap)

## Summary

> TOD3Cap asks a model to output a 3D box and a grounded description for each outdoor object. It contributes a nuScenes-derived dataset with 2.3 million descriptions and a model that connects fused BEV perception to a frozen language decoder through a Relation Q-Former. Its useful insight is that an object's caption needs evidence beyond the object itself: motion history, nearby agents, and road context determine what a phrase such as “waiting beside the bus” actually means.

## Core Insights

### Outdoor captioning needs motion and context as well as appearance

Outdoor scenes combine dynamic objects, sparse LiDAR returns, a fixed camera rig, and a much larger spatial extent than typical indoor scans. A distant pedestrian may have few points and limited pixels, while its motion and relationship to a lane are more relevant than a detailed shape description. Simply applying an indoor captioner therefore confounds weak localization with weak language generation.

The dataset separates four annotation components: appearance, motion, environment, and relationships. The last category uses a target object, spatial relation, and anchor object. Relationships account for an average 11.2 words, versus 3.7 for appearance, making context a substantial part of the language target. A captioner that only identifies color and category misses much of what the dataset asks it to express.

The annotations cover 850 nuScenes scenes, split into 700 training and 150 validation scenes. Pre-labeled 3D boxes are projected into images, LLaMA-Adapter proposes descriptions, and human annotators correct the four components. GPT-4 summarizes them, followed by another human check requiring agreement among three annotators. The 2.3 million descriptions are therefore a product of model assistance and human revision, not independently authored descriptions of 2.3 million distinct objects.

### The Relation Q-Former exposes the scene before language generation

Camera features enter a BEVFormer-style encoder: temporal attention incorporates the preceding BEV state, and spatial cross-attention samples the current multi-view images. LiDAR features are voxelized, flattened along height, and fused with camera BEV features through convolutions. A query-based detection head then produces the object proposals.

![TOD3Cap fuses camera and LiDAR BEV features, proposes objects, and passes scene-aware object queries to a frozen language decoder](/assets/images/tod3cap-towards-3d-dense-captioning-in-outdoor-scenes-source-figure-3.webp)
*Fig 1: Follow the two inputs to the Relation Q-Former: object proposals identify what to describe, while the BEV feature map supplies surrounding evidence. The adapter translates those contextualized queries into prompts for the frozen LLaMA decoder. | source: [TOD3Cap, Figure 3](https://arxiv.org/abs/2403.19589)*

The Relation Q-Former embeds proposals with an MLP and lets them interact with other proposals and the scene's BEV features through self-attention. Another projection and adapter align the resulting object queries with LLaMA-7B. The language backbone stays frozen; learning concentrates on perception and the interface that supplies visual evidence to it.

Training is staged: 24 epochs of detector pretraining, 10 epochs of captioner training with the detector frozen, and 10 epochs of joint refinement at a lower learning rate. This final stage updates the detector and captioning interface together while the LLaMA backbone remains frozen. To avoid decoding hundreds of sentences simultaneously, training matches proposals to ground truth and randomly samples a subset for caption supervision. This is a practical distinction between constructing all object proposals and paying language-decoding cost for all of them in every update.

### The metric makes a fluent caption conditional on localization

The evaluation averages a caption score over ground-truth objects, counting a prediction only when its matched box exceeds an IoU threshold. CIDEr@0.5 is therefore a joint localization-and-caption measure. A detailed sentence attached to the wrong box does not receive the same credit as a correctly grounded one.

The baselines are adapted with the same outdoor detector and pretrained detector weights, then trained on TOD3Cap. Under camera-plus-LiDAR input, TOD3Cap reaches CIDEr@0.5 of 108.0 versus 98.4 for adapted Vote2Cap-DETR, a 9.6-point gain. This comparison is more informative than comparing an outdoor model against an indoor detector that cannot reliably propose distant objects.

| TOD3Cap input | CIDEr@0.5 |
| --- | ---: |
| LiDAR only | 74.4 |
| Cameras only | 94.1 |
| Cameras and LiDAR | 108.0 |

Camera appearance and LiDAR geometry are complementary under this protocol. The camera-only relation-module ablation also isolates useful context: CIDEr@0.5 rises from 82.7 with a relational graph to 90.0 with a transformer decoder and 94.1 with the Relation Q-Former. Those are camera-only rows, so they should not be compared directly to the 108.0 multimodal headline as if only the relation module changed.

### Localization gates do not make every sentence factual

![TOD3Cap predicted and ground-truth boxes and captions, with incorrect descriptive phrases marked in red](/assets/images/tod3cap-source-figure-4.png)
*Fig 2: Several boxes align well while parts of the captions still differ, including clothing color and an object's relationship to its neighbors. Correct spatial grounding is necessary, but does not guarantee every generated attribute. | source: [TOD3Cap, Figure 4](https://arxiv.org/abs/2403.19589)*

The qualitative failures show why detection and language need separate inspection. A caption can correctly identify a pedestrian and its motion while inventing clothing details; another can locate a bus but describe the wrong neighboring object. Aggregate language similarity can obscure such local factual mistakes.

The scale experiment also gives a limited cost picture: the full model has 124.5 million tuned parameters and takes 350.4 minutes for inference across all 150 validation scenes, versus 316.1 minutes for the tiny variant. These are dataset-level runtimes, not per-frame latency or real-time control claims. The paper demonstrates dense scene description; downstream driving benefits remain an application to test rather than an established consequence of the caption score.

## High-Level Takeaways

- Captioning outdoor objects requires temporal and relational evidence in addition to object appearance.
- The Relation Q-Former is the bridge from localized proposals and scene context to language; the frozen decoder does not remove the need to learn that bridge.
- IoU-gated caption metrics appropriately penalize misplaced descriptions, while attribute-level factual errors still require closer inspection.
- The strongest comparison matches the outdoor detector and input modalities. The runtime and downstream examples do not establish a real-time driving system.
