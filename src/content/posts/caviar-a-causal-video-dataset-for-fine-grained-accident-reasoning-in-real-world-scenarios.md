---
title: 'CAViAR: A Causal Video Dataset for Fine-Grained Accident Reasoning in Real-World Scenarios'
date: '2026-08-19T09:00:00.000Z'
section: paper-shorts
postSlug: caviar-a-causal-video-dataset-for-fine-grained-accident-reasoning-in-real-world-scenarios
legacyPath: /paper shorts/2026/08/19/caviar-a-causal-video-dataset-for-fine-grained-accident-reasoning-in-real-world-scenarios.html
tags:
  - Autonomous Driving
  - Vision-Language Models
  - Safety Evaluation
  - Datasets
field: 'Autonomous Driving: VLMs & Evaluation'
summary: '2026 – CAViAR tests whether driving VLMs can ground accident responsibility in visible actions'
---

## 2026 – CAViAR

**arXiv:** [2608.19380](https://arxiv.org/abs/2608.19380)<br />
**Code and annotations:** [NEC Labs CAViAR](https://github.com/nec-labs-ma/CAViAR)

## Summary

> CAViAR turns accident understanding into a responsibility-grounding benchmark rather than another captioning task. Its 2,249 real dashcam videos carry 20,108 question-answer pairs for scene context, accident type, apparent at-fault and affected agents, and apparent rule-violation category. Across six open VLM configurations, lighting recognition is nearly saturated, while accident classification remains close to the majority baseline and the best reported rule-violation score is only 0.82 out of 5. The benchmark isolates a useful failure, but its labels describe responsibility visible from one camera—not legal liability—and lack a formal inter-annotator agreement or human-performance baseline.

## Core Insights

### The dataset separates visible context from responsibility

CAViAR adds a reviewed responsibility layer to 1,500 Car Crash Dataset clips and 749 Nexar clips. The fixed tasks ask what happened, which visible road user initiated the unsafe interaction, who was affected, and which of eleven jurisdiction-agnostic rule families best describes the visible behavior. Ambiguous or off-screen responsibility cases are excluded from responsibility evaluation instead of being forced into a single label. That decision makes the target narrower than legal fault, but also more defensible from dashcam evidence.

![CAViAR example showing a dashcam accident and question-answer targets for context, agent roles, and rule-relevant behavior](/assets/images/caviar-causal-question-answering.webp)
*Fig 1: CAViAR decomposes one accident clip into visible context, accident type, apparent at-fault and affected agents, and an apparent rule-violation category. | source: [CAViAR](https://arxiv.org/abs/2608.19380)*

![Figure 2 from CAViAR: A Causal Video Dataset for Fine-Grained Accident Reasoning in Real-World Scenarios](/assets/images/caviar-a-causal-video-dataset-for-fine-grained-accident-reasoning-in-real-world-scenarios-source-figure-2.webp)
*Fig 2: Row-normalized accident-type confusion matrices, aggregated over all six models (base vs. fine-tuned) at 16 FPS ( predictions each). Both regimes collapse onto Rear-End and rarely recover Side-by-Side or Head-on. | source: [CAViAR: A Causal Video Dataset for Fine-Grained Accident Reasoning in Real-World Scenarios](https://arxiv.org/abs/2608.19380)*


The split is deliberately cross-corpus: CCD supplies training data and Nexar supplies the test set, with no shared videos, scenes, or devices. Three model families—Cosmos-Reason2, Qwen3-VL, and InternVL3—are evaluated at 2B and 8B scales before and after language-backbone LoRA fine-tuning. The vision encoder and projector remain frozen, so the experiment tests whether language-side adaptation can map existing visual evidence to the new task rather than whether better visual grounding can be learned end to end.

| Diagnostic | Reported result | What it establishes |
| --- | ---: | --- |
| Lighting accuracy | 98.6% base / 98.7% fine-tuned, averaged across models | Some visible context is easy for the tested VLMs. |
| Accident-type macro-F1 | 18.6% / 21.1% | Aggregate accuracy hides collapse toward common classes. |
| Best apparent at-fault score | 2.27 / 5 | The best model often identifies a relevant agent without complete reasoning. |
| Best rule-violation score | 0.82 / 5 | Mapping visible action to a rule category is the hardest reported task. |
| Same-source CCD holdout | 31.12–39.60 BERTScore-F1 | Removing the CCD-to-Nexar shift does not remove the reasoning gap. |


The comparison with existing site benchmarks is instructive. [DriveBench](/paper%20shorts/2025/01/01/are-vlms-ready-for-autonomous-driving-drivebench.html) tests whether driving answers remain grounded under visual corruption, while [NARRATE](/paper%20shorts/2026/08/14/narrate-a-multimodal-real-world-australian-driving-dataset-for-human-centred-explanations-in-automated-driving.html) records explanations from the drivers who performed ordinary maneuvers. CAViAR changes the evaluated object: a model must connect an observed accident sequence to agent roles and a rule-relevant behavior.

### The benchmark still needs a human and annotation ceiling

The annotation protocol uses two primary annotators and two reviewers, but the paper does not report independent inter-annotator agreement. GPT-4o judging correlates with two human raters on 45 samples, yet that small validation cannot establish an absolute scale for responsibility quality. A human baseline is also not reported. The expensive decision is therefore whether to use CAViAR as a diagnostic training and evaluation set, not whether its current scores measure deployable accident adjudication.

## High-Level Takeaways

- CAViAR extends driving-VLM evaluation from recognizing scene attributes to grounding apparent responsibility in visible agents, actions, and rule categories.
- The cross-corpus split and same-source holdout suggest that domain shift contributes to the errors but does not fully explain the gap between context recognition and responsibility reasoning.
- Frozen vision modules limit the fine-tuning conclusion: marginal 8B gains do not show that larger models cannot improve when visual grounding is adapted jointly.
- The benchmark should not be used for legal or insurance decisions; its labels are single-view research annotations of apparent responsibility.
- The central claim would weaken if an independently re-annotated test set, expert human baseline, and counterfactual agent-role evaluation showed that the present gap mostly reflects label ambiguity or metric failure.
