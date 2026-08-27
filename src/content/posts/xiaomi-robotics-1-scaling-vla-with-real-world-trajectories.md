---
title: 'Xiaomi-Robotics-1: Scaling VLA Models with Real-World Trajectories'
date: '2026-07-16T00:00:00.000Z'
section: paper-shorts
postSlug: xiaomi-robotics-1-scaling-vla-with-real-world-trajectories
legacyPath: /paper shorts/2026/07/16/xiaomi-robotics-1-scaling-vla-with-real-world-trajectories.html
tags: [Vision-Language-Action, Robotics]
field: 'Vision-Language-Action & Robotics'
summary: '2026 – Xiaomi-Robotics-1: Scaling VLA Models with Real-World Trajectories'
---

## 2026 – Xiaomi-Robotics-1: Scaling VLA Models with Real-World Trajectories

**arXiv:** [2607.15330](https://arxiv.org/abs/2607.15330)

**Project:** [Xiaomi-Robotics-1](https://robotics.xiaomi.com/xiaomi-robotics-1.html)

## Summary

> Xiaomi-Robotics-1 pretrains on more than 100,000 hours of real-world UMI manipulation trajectories, then aligns the model to robot embodiments and imperative instructions. An automatic pipeline labels state transitions with language. The paper reports 57.4 percent success on RoboCasa365, above the previous 46.6 percent, and 20.07 on RoboDojo versus 13.07.

## Core Insights


![Figure 1 from Xiaomi-Robotics-1: Scaling VLA Models with Real-World Trajectories](/assets/images/xiaomi-robotics-1-scaling-vla-with-real-world-trajectories-source-figure-1.webp)
*Fig 1: Overview. Xiaomi-Robotics-1 is pre-trained on over 100k hours of real-world UMI trajectories with auto-labeled state-transition language prompts. | source: [Xiaomi-Robotics-1: Scaling VLA Models with Real-World Trajectories](https://arxiv.org/abs/2607.15330)*


The scale comes from UMI collection rather than robot-only teleoperation. Automatic state-transition captions give each clip a language condition tied to what changed, then post-training translates that broad skill base into the target embodiment.

The reported scaling trend covers this data pipeline and model family. It does not show that hours from different collection devices are interchangeable. Label quality, action normalization, and embodiment coverage remain part of the scaling law.

## High-Level Takeaways

- Xiaomi-Robotics-1 moves VLA pretraining into the 100,000-hour regime with UMI trajectories.
- State-transition language connects semantic goals to observed action outcomes.
- Scale transfers only through an explicit embodiment-alignment stage.
