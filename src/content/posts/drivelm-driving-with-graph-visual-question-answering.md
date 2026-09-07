---
title: 'DriveLM: Driving with Graph Visual Question Answering'
date: '2023-12-21T00:00:00.000Z'
section: paper-shorts
postSlug: drivelm-driving-with-graph-visual-question-answering
legacyPath: /paper shorts/2023/12/21/drivelm-driving-with-graph-visual-question-answering.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2023 – DriveLM: Driving with Graph Visual Question Answering"
---

**arXiv:** [2312.14150](https://arxiv.org/abs/2312.14150) · **Project:** [DriveLM](https://opendrivelab.com/DriveLM/) · **Code:** [OpenDriveLab/DriveLM](https://github.com/OpenDriveLab/DriveLM)

## Summary

> DriveLM makes driving supervision a graph of questions: identify relevant objects, predict their interactions, consider actions, choose a behavior, and generate a trajectory. Its contribution is a task, annotated datasets, evaluation framework, and deliberately simple VLM baseline. Graph context helps transfer to an unfamiliar driving dataset, but does not uniformly improve in-domain planning. The experiments expose both the value of structured language supervision and the cost of propagating imperfect answers through it.

## Core Insights

### The graph specifies which earlier answers a decision can use

A node is a question-answer pair, not an entire reasoning stage. Directed edges connect the facts needed by another question, including relationships across objects: planning around a sedan may depend on whether a pedestrian is crossing in front of it. The task stages organize these nodes into perception, prediction, planning, behavior, and motion. This is a directed acyclic graph of context dependencies, rather than a single chain in which every answer receives only the immediately preceding response.

The baseline implements an edge by appending the parent QA as text prefixed with “Context.” Multiple parents are concatenated. It uses ground-truth parent answers during training and predicted answers at inference, creating a clear source of error propagation. Inference can select a smaller subgraph to control cost; the paper does not require answering every annotated question for every scene.

![DriveLM's object-level and task-level dependencies connect scene questions to behavior and motion](/assets/images/drivelm-source-figure-1.png)
*Fig 1: Trace how facts about separate objects feed the planning question before converging on behavior and motion. The graph defines information dependencies; it does not prove that generated explanations faithfully describe the model's internal decision process. | source: [DriveLM, Figure 1](https://arxiv.org/abs/2312.14150)*

### Behavior compresses the graph before numerical trajectory generation

The behavior stage gathers perception, prediction, and planning answers into a coarse driving decision. Speed and steering each have five categories, producing a pair such as slow and straight. Several different hazards can imply the same behavior, so this intermediate representation compresses diverse scene reasoning into a small action vocabulary.

Motion generation then receives the image and behavior description. Each waypoint coordinate is quantized into 256 bins, represented by selected numeric vocabulary tokens. A separate set of LoRA weights adapts the BLIP-2 architecture for this motion-only task; the graph-answering and trajectory stages are not simply one unchanged checkpoint. The quantizer avoids asking an ordinary language decoder to invent arbitrary precise coordinates, although its resolution still constrains the resulting trajectory.

### The datasets separate human judgment from scalable expert labels

DriveLM-nuScenes annotates 4,871 keyframes, averaging 91.4 QA pairs per frame. Some factual perception answers come from existing annotations; humans provide much of the prediction and planning supervision. DriveLM-CARLA instead uses privileged simulator information and the PDM-Lite rule-based expert to generate roughly 1.6 million QA pairs across 64,285 frames. Its smaller keyframe subset concentrates on changes in the expert's decision.

![Human annotation and quality checks in nuScenes compared with rule-based QA generation in CARLA](/assets/images/drivelm-driving-with-graph-visual-question-answering-source-figure-2.webp)
*Fig 2: The two branches trade annotation diversity against generation scale. Human judgment supplies open-ended nuScenes answers, while privileged simulator state and an expert's rules produce CARLA labels. | source: [DriveLM, Figure 2](https://arxiv.org/abs/2312.14150)*

That distinction affects evaluation. More uniform rule-generated answers can be easier to reproduce than varied human explanations. PDM-Lite's closed-loop driving score measures the data-collection expert; it must not be attributed to the trained VLM agent, whose reported planning evaluation is open loop.

### Graph context helps transfer more clearly than it helps the original domain

Table 2 compares three context settings for behavior prediction: none, only the final planning answer, and the full graph. All three pass the predicted behavior to the motion stage.

| Behavior context | nuScenes ADE, meters | Waymo ADE, meters | Waymo speed accuracy |
| --- | ---: | ---: | ---: |
| None | 1.39 | 2.76 | 43.90% |
| Chain | 2.07 | 2.85 | 41.28% |
| Graph | 1.74 | 2.63 | 54.29% |

On nuScenes, simply introducing behavior is already effective: the no-context variant improves on motion-only BLIP-RT-2 at 2.63 meters and UniAD-Single at 1.80 meters. Adding the graph does not improve that best in-domain result. Video-based UniAD reaches 0.80 meters, but receives temporal input unavailable to the single-frame VLM.

The transfer test uses 1,000 Waymo validation frames without further training. Here, full graph context improves both speed classification and ADE relative to the simpler variants. The sensor comparison is asymmetric: the VLM already uses only a front image, whereas UniAD-Single loses its rear-camera inputs on Waymo. The result supports this particular transfer setup, without isolating reasoning quality from input compatibility.

### Asking the right question helps, but perception still limits the answer

In the supplement's pedestrian test, the CARLA driving training set contains no pedestrians. DriveLM-Agent also trains with COCO and GQA, so “unseen” refers to its driving data, not to never seeing people in any training source. Adding a pedestrian-specific question raises behavior accuracy from 4.59% to 27.04%. Ground-truth graph context plus that question reaches 92.35%, revealing how much performance is lost before the final behavior decision. The test contains pedestrians crossing straight roads, so every method gets perfect steering accuracy; speed determines the useful distinction.

The method remains expensive: the paper reports about 8.5 generated tokens per second and roughly ten times UniAD's runtime. Multiple QA rounds compound that cost. Closed-loop VLM driving is left to future work, so neither improved language scores nor closer logged trajectories establish that the agent can recover from its own control errors.

## High-Level Takeaways

- Graph VQA makes supervision and context dependencies inspectable. Its simplest implementation also makes earlier answer errors available to later decisions.
- A coarse behavior stage is useful even without full graph context; the strongest evidence for the graph comes from the tested domain-transfer setting.
- Question selection matters alongside model capacity. The pedestrian experiment shows a useful prompt intervention while exposing a substantial perception gap.
- Keep the data-collection expert, the VLM's offline planning results, and any future closed-loop system distinct when assessing the paper's driving claim.
