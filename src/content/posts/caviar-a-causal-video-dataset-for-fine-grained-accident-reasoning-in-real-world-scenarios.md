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

## Summary

> CAViAR makes accident understanding a responsibility-grounding problem. Its 2,249 real dashcam clips—1,500 from Car Crash Dataset (CCD) and 749 filtered Nexar clips—carry 20,108 QA pairs for environmental context, accident type, dense description, apparent at-fault and affected agents, and apparent rule-relevant behavior. Six open model variants are evaluated before and after LoRA fine-tuning on CCD and testing on Nexar at 16 FPS. Lighting reaches 98.6–98.7% accuracy, but accident-type macro-F1 is only 18.6–21.1%, the best apparent-fault judge score is 2.27/5, and the best rule-violation score is 0.82/5. These labels describe visible responsibility cues, not legal liability, and the benchmark still needs a larger independent annotation and human-performance study.

**arXiv:** [2608.19380](https://arxiv.org/abs/2608.19380)<br />
**Code and annotations:** [NEC Labs CAViAR](https://github.com/nec-labs-ma/CAViAR)

## Core Insights

### The dataset turns an accident clip into visible responsibility claims

CAViAR adds a structured annotation layer to established dashcam footage instead of collecting a new fleet. CCD supplies 1,500 training videos and Nexar supplies 749 held-out test videos. The sources share no video, scene, or device in the reported split, so the cross-corpus evaluation tests transfer as well as recognition. Each clip is mapped to nine prompts grouped into eight task families: dense captioning, weather, lighting, road condition, accident type, apparent at-fault agent, affected agent, and apparent rule-violation category. The maximum nine-field total is 20,241; 133 unavailable fields leave 20,108 QA pairs.

The word “apparent” is doing real work. Annotators identify the visible road user whose action appears to initiate the unsafe interaction, the road user visibly struck or endangered, and the rule-relevant behavior that can be supported by the video. Eleven jurisdiction-agnostic rule families cover unsafe following distance, failure to yield, lane changes, turns, signals, stopping, speed, observation, pedestrian interaction, loss of control, and overtaking. Ambiguous cases—such as an off-screen initiator or an interaction that cannot be assigned from the camera—are excluded from responsibility evaluation rather than forced into a legal-sounding label.

![CAViAR structured questions for one accident video](/assets/images/caviar-causal-question-answering.webp)
*Fig 1: One dashcam sequence is annotated with environmental MCQs, accident type, dense captions, apparent agent roles, and an apparent rule-violation category. | source: [CAViAR, Figure 1](https://arxiv.org/abs/2608.19380)*

Two primary annotators produce labels and two reviewers check cross-field consistency. GPT-4 is used only to normalize grammar under an instruction not to add or change facts; human labels remain the source of truth. This makes the dataset useful for grounding research, while the absence of formal independent agreement means that the labels should not be treated as an adjudication standard.

### Perceptual recognition and responsibility reasoning separate sharply

The benchmark evaluates Cosmos-Reason2, Qwen3-VL, and InternVL3 at 2B and 8B scales, with zero-shot base models and LoRA fine-tuning on CCD. The vision encoder and projector stay frozen; only language-model attention and MLP projections receive adapters. Frames are sampled uniformly at 16 FPS, with model-specific caps, and greedy decoding is used. MCQs are scored by accuracy, dense answers by BERTScore-F1, and responsibility answers by a 0–5 GPT-4o judge whose rubric gives a 5 only for the correct agent and reasoning.

![CAViAR accident-type confusion matrices](/assets/images/caviar-a-causal-video-dataset-for-fine-grained-accident-reasoning-in-real-world-scenarios-source-figure-2.webp)
*Fig 2: Row-normalized accident-type confusion matrices aggregated over the six base and fine-tuned models at 16 FPS; each regime contains N=4,494 predictions. Both collapse toward Rear-End and rarely recover Side-by-Side or Head-on. | source: [CAViAR, Figure 2](https://arxiv.org/abs/2608.19380)*

The separation is visible in Table 5. Averaged over the six model configurations, lighting accuracy is 98.6% for base and 98.7% after fine-tuning. Weather is 58.4%/62.6%, road condition 65.3%/74.0%, and accident type 33.1%/35.4%. Accident-type macro-F1 is only 18.6%/21.1%, showing that raw accuracy is mostly a class-prior measure. On open-ended responsibility, the best apparent-at-fault score is 2.27/5 and the best affected-agent score is 2.70/5; rule violation is weaker still, peaking at 0.82/5. A model can name a relevant vehicle while failing to connect the agent, action, and rule.

The fine-tuning gains are concentrated in the smaller models. On Nexar, Qwen3-2B moves from 55.67% to 67.16% MCQ accuracy and from 1.280 to 1.763 on the judge, while InternVL3-8B changes from 69.06% to 68.83% and 1.527 to 1.549. The frozen visual pathway and small one-run gains at 8B make language-side adaptation a limited answer to a grounding problem.

### Same-source controls expose class priors and domain shift

Figure 2 shows why accident accuracy needs balanced diagnostics. Both base and fine-tuned predictions over-predict Rear-End by 2.58× and 2.01×, respectively; Side-by-Side recall is about 1%, and Head-on is rarely recovered. Fine-tuning shifts some mass toward None and nudges raw accuracy upward without repairing the class prior.

The rule distribution is also not stationary. On the Nexar test set, unsafe following distance is 32.8%, failure to yield 30.6%, and improper lane change 17.2%. Loss of control and overtaking appear in CCD training at 8.3% and 1.3% but are essentially absent in Nexar. A same-source CCD holdout of 1,200 training and 300 test videos still produces 62.17–67.92% MCQ accuracy and 31.12–39.60 BERTScore-F1. The reasoning gap therefore persists when the CCD-to-Nexar shift is removed, although the shift remains a real confound.

CAViAR is best used to test whether a model grounds agent roles and rules in temporal evidence. It does not establish legal fault, closed-loop avoidance, or expert human ceiling performance. The most useful next experiment is a stratified re-annotation with disagreement-aware labels and a temporal model that must point to the evidence used for its responsibility claim.

## High-Level Takeaways

- CAViAR measures the step from “what happened?” to “which visible action appears to have caused it?”
- Lighting and some context cues can be nearly solved while rule-grounded responsibility remains below one point on a five-point judge scale.
- Raw accident accuracy is misleading when predictions collapse toward Rear-End; balanced accuracy, macro-F1, confusion matrices, and source-shift controls are essential.
- The benchmark’s next ceiling is evidence grounding: models should identify the relevant agent and time span before naming a rule.
