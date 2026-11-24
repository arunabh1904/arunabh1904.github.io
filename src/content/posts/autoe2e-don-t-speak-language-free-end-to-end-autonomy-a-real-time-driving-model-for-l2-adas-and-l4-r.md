---
title: 'AutoE2E: Don''t speak - Language free end-to-end autonomy A Real-Time Driving Model for L2++ ADAS and L4 Robotaxis'
date: '2026-11-17T09:00:00.000Z'
section: paper-shorts
postSlug: autoe2e-don-t-speak-language-free-end-to-end-autonomy-a-real-time-driving-model-for-l2-adas-and-l4-r
legacyPath: /paper shorts/2026/11/17/autoe2e-don-t-speak-language-free-end-to-end-autonomy-a-real-time-driving-model-for-l2-adas-and-l4-r.html
tags:
  - Autonomous Driving
  - Planning
  - Bird's-Eye View
field: "Autonomous Driving: VLA & Planning"
summary: "2026 – AutoE2E: Don't speak - Language free end-to-end autonomy A Real-Time Driving Model for L2++ ADAS and L4 Robotaxis"
---

## 2026 – AutoE2E: Don't speak - Language free end-to-end autonomy A Real-Time Driving Model for L2++ ADAS and L4 Robotaxis

**Paper:** [Khawaja et al., arXiv:2611.22374v1](https://arxiv.org/abs/2611.22374v1) · **Code:** [Autoware v1.0 release](https://github.com/autowarefoundation/auto_e2e/releases/tag/v1.0)

## Summary

> AutoE2E is the reactive component of Autoware's proposed driving stack. It converts camera history, map and route rasters, and ego motion into one trajectory without generating language. The released system supports an inspectable open-loop baseline, but its reported displacement errors do not establish closed-loop driving capability.

## Core Insights

### Keep the camera representation and learn how to add navigation

The camera pathway fuses eight bird's-eye-view (BEV) frames sampled half a second apart. Separate encoders process the HD map and selected route. A learned channel gate scales the route features before they are added to the map features. Camera BEV cells then query this navigation representation through deformable attention. The fusion residual starts at zero, so attaching navigation does not immediately alter the pretrained image BEV.

![Temporal camera BEV and separately encoded map and route join at navigation fusion, before the trajectory planner receives ego history.](/assets/images/autoe2e-reactive-path.svg)
*Fig 1: An original schematic of the evaluated reactive path. Camera features query the encoded navigation grid before the planner uses the fused representation. | source: [Khawaja et al., Figure 2](https://arxiv.org/html/2611.22374v1#S2.F2)*

[Open the figure at full size](/assets/images/autoe2e-reactive-path.svg)

A GRU uses ego-motion history to initialize its state, samples the fused BEV at each future step, and emits acceleration and curvature. A unicycle model integrates those outputs into positions. The policy returns a single 6.4-second trajectory at 10 Hz. The evaluated release updates about 1.45 million of 79.9 million parameters, with the bulk of the visual stack frozen. The planned reasoning and world-action components are not part of these results.

### The route evaluation knows which lane sequence was driven

On 11,035 KITScenes validation samples with cameras, HD map, and route, the fine-tuned checkpoint reaches 0.745 m average displacement error and 2.020 m final displacement error at three seconds. The baseline reaches 0.915 m and 2.423 m under that same protocol. The route is derived retrospectively by fitting the logged drive to lanes; it is not a route suggested independently before the drive. Camera-only Test uses different scenes, so the two scores cannot isolate the effect of providing navigation. Replay also scores only the first five seconds of the 6.4-second forecast.

The headline throughput of 76 FPS was measured with a lighter camera backbone and different fusion, not the ResNet-50 configuration used in the main accuracy tables. The paper reports a separate 101 ms measurement on the EDGAR platform for a checkpoint retrained on EDGAR data. Neither is a controlled speed measurement of the KITScenes checkpoint, and the release reports no closed-loop safety or intervention metrics.

## High-Level Takeaways

- Separate the deterministic reactive model that shipped from the reasoning interfaces proposed for later.
- Measure the effect of navigation by changing inputs on identical scenes, and test mistakes from a live route source.
- Keep displacement error, throughput for the exact checkpoint, and closed-loop safety as distinct deployment questions.
