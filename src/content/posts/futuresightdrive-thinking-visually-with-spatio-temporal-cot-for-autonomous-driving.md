---
title: 'FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving'
date: '2025-05-23T09:55:32.000Z'
section: paper-shorts
postSlug: futuresightdrive-thinking-visually-with-spatio-temporal-cot-for-autonomous-driving
legacyPath: /paper shorts/2025/05/23/futuresightdrive-thinking-visually-with-spatio-temporal-cot-for-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving"
---
## 2025 – FutureSightDrive (FSDrive)

**arXiv:** [2505.17685](https://arxiv.org/abs/2505.17685)

**Code:** [MIV-XJTU/FSDrive](https://github.com/MIV-XJTU/FSDrive)

## Summary

> FutureSightDrive makes its chain of thought a predicted visual scene rather than a text trace. A world-model path generates a future frame with background, future lane dividers, and 3D boxes; an inverse-dynamics VLA then plans a trajectory from the current observation and that visual spatio-temporal CoT. The paper reports improved trajectory accuracy and fewer collisions on nuScenes and NAVSIM, plus competitive video-generation FID and DriveLM understanding results. The abstract does not report the planning latency or a control that replaces the imagined frame with an equally informative nonvisual state.

## Core Insights

The representation choice is deliberate. Textual reasoning can discard geometry and temporal relations before planning, whereas a predicted future scene can carry lanes, actors, and motion in one visual object. FSDrive expands the vocabulary with visual tokens and jointly trains VQA and future-frame prediction. Its progressive curriculum predicts structural priors before rendering the whole scene, which makes physical constraints part of the generation target rather than an after-the-fact caption.

The cost is prediction error. The planner consumes a generated scene, so errors in lanes, boxes, or background can become action errors even when the current camera view is clear. The abstract does not disclose the visual-token budget, forecast horizon, loss weights, or a teacher-forced imagined-scene comparison. The key safety test should measure how action quality changes as forecast error is injected independently into map structure, moving actors, and scene appearance.

## High-Level Takeaways

- FSDrive makes a predicted visual future the intermediate reasoning object, then conditions an inverse-dynamics planner on that object.
- Its reported nuScenes and NAVSIM results support the visual-CoT hypothesis, but do not isolate prediction quality from the rest of the VLA training recipe.
- The approach is justified only if planning degrades gracefully under forecast errors; an equally sized latent or structured prediction state that matches safety would weaken the case for rendering a future frame.
