---
title: "Towards Reliable Vision-Language Models for Autonomous Driving"
date: '2026-10-01T09:00:00.000Z'
section: paper-shorts
postSlug: towards-reliable-vision-language-models-for-autonomous-driving
legacyPath: /paper shorts/2026/10/01/towards-reliable-vision-language-models-for-autonomous-driving.html
tags: ["Autonomous Driving", "Evaluation", "Uncertainty"]
field: "Autonomous Driving: VLMs & Evaluation"
summary: "2026 \u2013 Towards Reliable Vision-Language Models for Autonomous Driving"
---

## 2026 – Towards Reliable Vision-Language Models for Autonomous Driving

**Paper:** [arXiv:2610.01531](https://arxiv.org/abs/2610.01531) · [Full text](https://arxiv.org/html/2610.01531v1)

## Summary

> A driving VLM can keep a similar answer accuracy while its confidence becomes less useful. This paper evaluates those two properties separately and shows why visual enhancement should not be accepted on accuracy alone.

## Core Insights

The study evaluates five models: Qwen3.5-9B, Gemma4-E4B, LLaVA-OneVision-7B, DriveFusionQA-4B, and Alpamayo-1.5-10B. It uses 1,000 DrivingVQA examples, 1,117 NuScenes-QA-mini examples, 50 OSR questions, and generally 1,000 STRIDE-QA examples. Each image receives separate seeded glare, fog, blur, and lens-occlusion corruptions. Multiple-choice tasks use exact matching; open-ended NuScenes answers use a GPT-4 judge. STRIDE applies distance and heading tolerances.

Metrics include accuracy, vision-aware uncertainty AUROC, calibration error, and Brier score. Inference-time Visual Evidence Augmentation changes accuracy and reliability differently across models. For Alpamayo, clean DrivingVQA accuracy improves by 7.4 percentage points, while confidence metrics worsen. Qwen3.5's NuScenes accuracy falls from 42.1% to 29.5% under glare. OSR is too small for strong condition-level conclusions. The intervention adds task-dependent latency, reaching roughly three seconds for DriveFusionQA on STRIDE. These are question-answering experiments, not closed-loop driving tests.

What changes when the same scene is degraded? The source example makes the input intervention visible. It is an illustration of the experiment, not an independent proof of its aggregate result.

![Source Figure 1 shows clean and perturbed driving inputs with Gemma4-E4B responses.](/assets/images/driving-vlm-reliability-source-figure-1.png)

*Fig 1: Source Figure 1 illustrates how visual perturbations can change a model's response to a driving scene. Aggregate accuracy and confidence require separate evaluation. | source: [Towards Reliable Vision-Language Models for Autonomous Driving](https://arxiv.org/abs/2610.01531)*

### Accuracy and confidence serve different decisions

Accuracy asks how often an answer is correct. Confidence is useful when a system must decide whether to trust that answer, gather more evidence, or defer. A model that is usually correct can still fail the second task if its wrong answers receive its highest confidence.

Calibration and error discrimination also differ. A model can assign approximately correct average probabilities while ranking individual failures poorly. Conversely, it can rank failures well but report probabilities that are systematically too high. An adapter evaluation needs both properties if the output will control an abstention threshold.

For a box-and-attribute interface, test missingness as an explicit input condition. An absent box can mean no object, detector failure, occlusion, or an unavailable camera. Those cases require different answers. A confidence number without a visibility flag cannot reliably distinguish them.

### A controlled corruption is a diagnostic, not a road test

Synthetic blur isolates one intervention. Real driving can combine blur, glare, calibration drift, and stale tracks. An adapter may also receive inconsistent modalities: a current camera frame beside an old object list. The relevant question then becomes which evidence the model follows and whether it reports the conflict.

My proposed evaluation would retain paired clean and corrupted examples, add combined failures, and separate visual questions from questions answerable through traffic-rule knowledge. Replace or shuffle the image, boxes, and attributes independently. Record the answer, confidence, invalid-output rate, and added latency for every condition.

Finally, split calibration fitting from final evaluation. Choose an abstention threshold on a validation set, then report coverage and error on held-out scenes. Otherwise the threshold can become another way to tune against the test distribution. None of these checks establishes vehicle-level safety, but each can reject a misleadingly strong interface before a more expensive evaluation.

## High-Level Takeaways

- Report task accuracy, calibration, and error discrimination separately.
- Evaluate contradictory and stale context as well as degraded images.
- Measure the latency added by a visual intervention under the intended input format.
- Treat driving question answering as one component test, not a substitute for closed-loop validation.
