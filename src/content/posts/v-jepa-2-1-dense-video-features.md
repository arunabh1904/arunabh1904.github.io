---
title: 'V-JEPA 2.1: Dense Features in Video Self-Supervised Learning'
date: '2026-03-15T00:00:00.000Z'
section: paper-shorts
postSlug: v-jepa-2-1-dense-video-features
legacyPath: /paper shorts/2026/03/15/v-jepa-2-1-dense-video-features.html
tags: [Video Models, Dense Prediction]
field: 'Video & Interactive World Models'
summary: '2026 – V-JEPA 2.1: Dense Features in Video Self-Supervised Learning'
---

## 2026 – V-JEPA 2.1: Dense Features in Video Self-Supervised Learning

**arXiv:** [2603.14482](https://arxiv.org/abs/2603.14482)

## Summary

> V-JEPA 2.1 adds dense predictive loss, deep self-supervision, and joint image-video training to improve local features without giving up global scene understanding. The paper reports 7.71 mAP on Ego4D interaction anticipation, 40.8 Recall@5 on EPIC-KITCHENS, and a 20-point real-robot grasping gain over V-JEPA 2-AC.

## Core Insights

![Dense V-JEPA 2.1 patch features compared with V-JEPA 2 across video frames](/assets/images/v-jepa-2-1-paper-figure-1.png)
_Patch-feature projections remain spatially and temporally coherent in V-JEPA 2.1, while the earlier model is more globally semantic. Source: [V-JEPA 2.1](https://arxiv.org/abs/2603.14482), Figure 1._

The dense loss trains both visible and masked tokens instead of concentrating prediction on a global target. Deep self-supervision applies the objective at intermediate layers, where local structure has not yet been compressed into the final representation.

This directly tests a recurring limitation of global objectives. Strong recognition does not guarantee depth, correspondence, or contact-relevant features. V-JEPA 2.1 improves those outputs by measuring them during representation learning.

## High-Level Takeaways

- Dense predictive targets make local spatial evidence part of self-supervised video learning.
- Intermediate-layer supervision prevents all useful structure from depending on the final representation.
- Global and dense quality can coexist, but only when the objective pressures both.
