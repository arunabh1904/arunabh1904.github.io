---
title: 'Can MLLMs Reason over Surround-View Driving Videos? A Multi-View Autonomous Driving QA Benchmark'
date: '2026-11-11T09:00:00.000Z'
section: paper-shorts
postSlug: can-mllms-reason-over-surround-view-driving-videos-a-multi-view-autonomous-driving-qa-benchmark
legacyPath: /paper shorts/2026/11/11/can-mllms-reason-over-surround-view-driving-videos-a-multi-view-autonomous-driving-qa-benchmark.html
tags:
  - Autonomous Driving
  - Video Understanding
  - Benchmarks
field: "Autonomous Driving: VLMs & Evaluation"
summary: "2026 – Can MLLMs Reason over Surround-View Driving Videos? A Multi-View Autonomous Driving QA Benchmark"
---

## 2026 – Can MLLMs Reason over Surround-View Driving Videos? A Multi-View Autonomous Driving QA Benchmark

**Paper:** [Liu et al., arXiv:2611.15225v2](https://arxiv.org/abs/2611.15225v2)

## Summary

> MVDriveBench tests whether a video-language model can answer driving questions that need more than the front camera. Removing side and rear views hurts tracking and cross-camera association much more than it hurts planning questions. A plausible answer about what the car should do can survive the removal of the evidence that would justify it.

## Core Insights

### Choose clips where the missing view matters

The benchmark starts with real driving clips from NVIDIA's PhysicalAI-Autonomous-Vehicles dataset. A front-only model flags clips where another view may matter and localizes the relevant interval. Six synchronized views are then rendered as a five-second, 30 Hz tiled video. A model first describes the scene, then uses that description to generate questions, answers, and reference rationales. Reviewers remove items with unsupported evidence, wrong object associations, or ambiguous answers.

![The front view is used to filter clips; selected clips then undergo six-view analysis and annotation review.](/assets/images/mvdrivebench-evidence-pipeline.svg)
*Fig 1: An original schematic of the filtering and annotation pipeline. The front view helps choose clips; all six views supply the evidence for the final questions. | source: [Liu et al., Figure 2](https://arxiv.org/html/2611.15225v2#S3.F2)*

[Open the figure at full size](/assets/images/mvdrivebench-evidence-pipeline.svg)

The 12 tasks distinguish motion understanding, scene understanding across views, and driving reasoning. Binary and multiple-choice items use exact matching; short and open responses use a model judge informed by the reference rationale. The aggregate, therefore, combines automatically graded choices and judged answers. It does not directly establish that a model's explanation matches the evidence it used.

### Run the same benchmark without the supporting views

Across 23 models, GPT-5.5 led with 80.1 overall, 8.4 points below the reported human result. Several specialist models topped 95 on planning while scoring below 55 overall. The front-only ablation makes the mismatch concrete: InternVL3-78B fell from 91.1 to 40.5 on cross-view consistency, while several models still scored above 95 on planning items with only the front view. That persistence is consistent with generic driving priors; it is not proof of the mechanism behind any particular answer.

The [AlpaSim experiment](https://arxiv.org/html/2611.15225v2#S6) compared QA scores with safety outcomes for specialist models. The authors report a positive association after leaving out Alpamayo-R1-10B, a trajectory-trained outlier. This is a model-level association within the tested set, not evidence that improving QA will improve driving or that the ordering holds on public roads.

## High-Level Takeaways

- Compare each model against itself with side and rear evidence removed. The category-level change is more informative than the overall ranking.
- Treat a high planning QA score as a separate outcome from correctly tracking an actor across cameras.
- Read the simulation comparison as a limited test of whether QA and control performance move together.
