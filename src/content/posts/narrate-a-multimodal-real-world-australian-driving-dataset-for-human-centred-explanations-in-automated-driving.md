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

## Summary

> NARRATE records explanations from the people who performed real-world driving actions. It contains 2,050 annotated events from 35 experienced drivers and instructors on a 14.5 km Brisbane route, synchronized with four 10 fps camera views, forward LiDAR, GPS/INS, speed, and acceleration. Each 15-second event can have an in-vehicle explanation, a post-drive video-cued explanation, or both. The labels cover ten driver-action classes, six high-level and 32 fine-grained contexts, and span-level Perception, Comprehension, and Projection. Text makes broad situation-awareness labels recoverable, but RoBERTa reaches only 0.035 macro-F1 on fine-grained context and T5 reaches 0.238 ROUGE-L for explanation generation. The dataset is a human-centred explanation resource, not a fleet-scale perception benchmark.

**arXiv:** [2608.14767](https://arxiv.org/abs/2608.14767)<br />
**Code and data:** [NARRATE](https://github.com/ashkan-zadeh/NARRATE)

## Core Insights

### The collection contract ties a reason to the driver and the moment

NARRATE uses a Kia Niro with a roof-mounted sensor rig and ROS 2 recording. Four lossless cameras run at 10 fps—front-centre, forward-left, forward-right, and rear-centre—alongside a forward Leishen CH128X LiDAR, GPS-aided INS/IMU, GNSS position, speed, and longitudinal acceleration. The final corpus contains 2,050 events and 8,200 camera sequences from 35 participants: 30 experienced non-instructors and five licensed driving instructors. Sessions last about 40 minutes on public roads in Brisbane, under a left-hand-traffic, right-hand-drive Australian convention.

The paper’s distinction is who supplies the explanation. Events come from participant think-aloud comments, researcher prompts after completed actions, or engineer-flagged deferrals when discussing the event during driving would be unsafe or impractical. Each event is represented by a synchronized 15-second clip with roughly 10 seconds before the tagged maneuver and five seconds after it. In-vehicle narration captures what is salient under workload. A post-drive interview replays selected events and asks what the driver did, why, what they perceived, and what could have happened otherwise.

![NARRATE’s instrumented vehicle and dual-timing explanation protocol](/assets/images/narrate-data-collection-paper-figure.png)
*Fig 1: NARRATE combines multi-sensor driving, real-time event tagging, in-vehicle explanation, and a video-cued post-drive interview; short or deferred events are revisited after the drive. | source: [NARRATE, Figure 1](https://arxiv.org/abs/2608.14767)*

The final 2,050 events come from 2,138 raw tags after excluding erroneous or unlocatable events and three without a valid explanation. Participant-disjoint stratified splits contain 1,402 train, 272 validation, and 376 test events across 24, five, and six participants. There are 1,273 in-vehicle explanations, 1,110 post-drive explanations, and 333 events with both. Post-drive explanations are longer: median 19 words versus 13 in-vehicle, which is a measurable difference between reflection and real-time narration rather than a nuisance to average away.

### The labels represent reasoning at both span and event scale

Situational Awareness labels follow Endsley’s three levels. L1 Perception marks what the driver noticed, L2 Comprehension marks its relevance to the maneuver, and L3 Projection marks an anticipated future state or risk. A single explanation can contain all three, and annotators highlight the supporting spans. Context is separate: each event can receive multiple labels from 32 fine-grained categories grouped into Traffic Compliance, Social Interaction and Traffic Flow, Navigation and Routing, Hazard and Obstacle Management, Special Zones and Stops, and Environmental and Adaptation.

![Representative NARRATE events with action, context, frames, and explanations](/assets/images/narrate-a-multimodal-real-world-australian-driving-dataset-for-human-centred-explanations-in-automated-driving-source-figure-2.webp)
*Fig 2: Each representative event pairs an action and context label with three front-centre frames and the driver’s in-vehicle and post-drive explanations. | source: [NARRATE, Figure 2](https://arxiv.org/abs/2608.14767)*

The annotation is not an unsupported single-pass label. Annotator A labels the full dataset; B and C independently label SA on a stratified 303-event reliability subset covering all participants, and C additionally labels context. L1 three-way Fleiss’ kappa is 0.580, L2 is 0.050 under a prevalence-heavy label, and L3 is 0.203 because annotator B used Projection conservatively. The A–C pairwise L3 kappa is 0.690. Context agreement is stronger: fine-grained exact agreement is 82.2% and high-level exact agreement 85.8%, with αMASI of 0.852 and 0.861. These numbers make the ambiguity visible, especially for “future risk” language.

The action distribution is long-tailed. Slow down is 976 events (47.6%), lane change 451 (22.0%), and speed up 187 (9.1%); four non-action classes together account for 9.0%. L2 appears in 93.6% of in-vehicle and 91.3% of post-drive explanations, L1 in 89.0% and 83.3%, and L3 in 69.1% and 70.2%. All three levels co-occur in 59.1% of in-vehicle and 58.6% of post-drive explanations.

### Baselines show that language helps, but grounding is still missing

NARRATE defines four participant-disjoint diagnostic tasks. Text baselines use a maximum length of 128 and five seeds. Visual baselines sample eight frames from each 15-second clip through frozen CLIP ViT-B/32; kinematic features contain speed and longitudinal acceleration at those same times. The design asks what can be recovered from driver language alone and what requires the event evidence.

For SA classification, RoBERTa reaches an in-vehicle macro-F1 of 0.914±0.007 and a post-drive macro-F1 of 0.911±0.008. Those high scores are partly prevalence-driven: L1 and L2 appear in most explanations. L3 provides the more informative margin, with RoBERTa at 0.863 versus a 0.773 majority baseline in-vehicle. Context is harder. RoBERTa’s six-class macro-F1 is 0.407±0.023 in-vehicle and 0.418±0.004 post-drive, but the 32-class primary-text macro-F1 falls to 0.035±0.011 and weighted-F1 to 0.100±0.048.

For driver action, frozen video alone is weaker than language and motion. RoBERTa text reaches 0.728±0.009 accuracy and 0.289±0.018 macro-F1; CLIP plus kinematics gives the best macro-F1 at 0.306±0.016, showing that simple visual and motion cues add complementary signal. Explanation generation is evaluated from action, context, and kinematic inputs rather than end-to-end pixels. T5-base is best on BLEU-4 (0.036±0.002), ROUGE-L (0.238±0.010), and BERTScore-F1 (0.895±0.003), while GPT-2 has the best METEOR at 0.185±0.010. Low n-gram scores are expected because multiple natural explanations can be valid; they also show that generating a driver-like reason remains open.

NARRATE’s boundary is deliberate. One Brisbane route, 35 drivers, mostly daytime collection, and long-tailed actions limit geographic and behavioral coverage. The participant-disjoint split tests new drivers within this cohort; transfer to new cohorts, regions, and traffic conventions still needs to be tested. The dataset gives explanation models a human source of reasons and a sensor-grounded event.

## High-Level Takeaways

- NARRATE’s distinctive unit is a driver-produced explanation tied to a synchronized maneuver, not an observer’s post-hoc caption.
- Span-level SA labels make “noticed,” “understood,” and “anticipated” separately testable, including their annotation uncertainty.
- Text recovers broad action and SA structure, while fine-grained context and natural explanation generation remain difficult.
- The decisive follow-up is cross-region, cross-driver evaluation with multimodal models that use the event evidence rather than only the explanation text.
