---
title: "NARRATE: A Multimodal Real-World Australian Driving Dataset for Human-Centred Explanations in Automated Driving"
date: '2026-08-14T00:00:00.000Z'
section: paper-shorts
postSlug: narrate-a-multimodal-real-world-australian-driving-dataset-for-human-centred-explanations-in-automated-driving
legacyPath: /paper shorts/2026/08/14/narrate-a-multimodal-real-world-australian-driving-dataset-for-human-centred-explanations-in-automated-driving.html
tags:
  - Autonomous Driving
  - Human-Centred AI
  - Datasets
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2026 – NARRATE: A Multimodal Real-World Australian Driving Dataset for Human-Centred Explanations in Automated Driving"
---

## 2026 – NARRATE: A Multimodal Real-World Australian Driving Dataset for Human-Centred Explanations in Automated Driving

**arXiv:** [2608.14767](https://arxiv.org/abs/2608.14767)<br />
**Code and data:** [NARRATE](https://github.com/ashkan-zadeh/NARRATE)

## Summary

> NARRATE collects explanations from the drivers who performed real-world manoeuvres, rather than asking observers or language models to reconstruct reasons after the fact. It contains 2,050 annotated events from 35 experienced drivers and instructors in Brisbane, synchronized with four camera views, LiDAR, localization, and motion streams. The labels cover driver action, six high-level and 32 fine-grained contexts, and span-level Perception, Comprehension, and Projection annotations.

## Core Insights

The dataset's contribution is the collection contract. Each event has a 15-second synchronized clip, an action, scenario context, and either in-vehicle or post-drive free-text explanation; some events have both. A dual-timing protocol captures what drivers say under workload and what they can explain after replay. Participant-disjoint splits prevent a model from memorizing a driver's language.

The benchmark is deliberately diagnostic rather than a new end-to-end model. Text baselines recover much of the SA structure, but fine-grained context recognition and explanation generation remain difficult. The dataset is also long-tailed: slow down accounts for 47.6% of actions, while the less frequent projection label appears in about 69–70% of explanations depending on timing.

![NARRATE data-collection protocol combining instrumented driving and post-drive explanation elicitation](/assets/images/narrate-data-collection-paper-figure.png)
*Fig 1: The protocol combines in-vehicle narration, event tagging, and video-cued post-drive interviews. | source: [NARRATE](https://arxiv.org/abs/2608.14767)*

![Figure 2 from NARRATE: A Multimodal Real-World Australian Driving Dataset for Human-Centred Explanations in Automated Driving](/assets/images/narrate-a-multimodal-real-world-australian-driving-dataset-for-human-centred-explanations-in-automated-driving-source-figure-2.webp)
*Fig 2: Representative events from NARRATE. Each panel shows an annotated driving event with its driver-action label, high-level and fine-grained scenario-context tags, three front-centre camera frames sampled around the tagged event time, and the corresponding driver explanations. | source: [NARRATE: A Multimodal Real-World Australian Driving Dataset for Human-Centred Explanations in Automated Driving](https://arxiv.org/abs/2608.14767)*


## High-Level Takeaways

- NARRATE informs whether driving explanation models should learn from driver-produced reasons rather than observer-written or generated text.
- The atomic unit is a participant-disjoint driving event with synchronized sensors, action/context labels, and one or two explanations.
- Realism and human grounding come at the cost of scale: one Brisbane route, 35 participants, daytime collection, and long-tailed labels.
- The conclusion would weaken if models trained on NARRATE do not transfer across regions, traffic conventions, or independently collected driver explanations.
