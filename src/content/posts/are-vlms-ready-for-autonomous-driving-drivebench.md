---
title: 'Are VLMs Ready for Autonomous Driving?'
date: '2025-01-07T00:00:00.000Z'
section: paper-shorts
postSlug: are-vlms-ready-for-autonomous-driving-drivebench
legacyPath: /paper shorts/2025/01/01/are-vlms-ready-for-autonomous-driving-drivebench.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2025 – Are VLMs Ready for Autonomous Driving?"
---

## Summary

> DriveBench asks whether a driving VLM is using the image or merely completing a familiar question. It covers 19,200 frames and 20,498 question–answer pairs across four driving tasks and 17 input settings: clean images, 15 visual corruptions, and text-only prompts. The revealing comparison is that GPT-4o scores 59.0% on clean perception multiple-choice questions and 59.5% with the image removed, while a human score is 93.3% on the clean subset. That is a benchmark diagnostic, not evidence that every model ignores vision: Qwen2-VL-72B falls from 60.0% to 23.5% on the same comparison. The result is that fluent driving language and aggregate QA accuracy are insufficient evidence of visual grounding.

**arXiv:** [2501.04003](https://arxiv.org/abs/2501.04003)

## Core Insights

### The benchmark changes the evidence while keeping the question recognizable

DriveBench is built by subsampling 200 keyframes from the DriveLM training data and balancing the ground-truth distribution so a majority answer such as “going ahead” cannot dominate as easily. Each frame is paired with questions about perception, prediction, planning, or ego behavior. The clean benchmark has 400 perception samples, 61 prediction samples, 600 planning samples, and 200 behavior samples. The broader robustness set adds corruption-recognition, perception, prediction, planning, and captioning questions for 19,237 total items across corruption types.

The key design is the 17-setting view of the same driving problem. There is a clean input, a black-image/text-only condition, and 15 corruptions grouped into weather and lighting, external lens disturbances, sensor failures, motion blur, and transmission errors. These include dark, fog, rain, frame loss, camera crash, zoom blur, bit error, color quantization, and H.265 compression. The benchmark therefore changes the visual evidence while keeping much of the language interface stable.

![DriveBench overview across perception, prediction, behavior, and planning](/assets/images/drivebench-paper-figure-1-overview.png)
*Fig 1: DriveBench spans perception, prediction, planning, and behavior under clean, corrupted, and text-only inputs, with 19,200 frames and 20,498 QA pairs. | source: [Are VLMs Ready for Autonomous Driving?, Figure 1](https://arxiv.org/abs/2501.04003)*

That construction matters because a model can answer a scene question from the wording alone. The paper evaluates 12 VLMs, including generalist open models, GPT-4o, and driving-specialist systems, using accuracy, BLEU, ROUGE-L, and rubric-conditioned GPT scores. It prompts models to provide explanations even for multiple-choice questions, so the evaluator can inspect the reasoning text as well as the selected answer.

### Plausible answers can survive the removal of pixels

The paper’s most useful experiment is not a leaderboard. It is the black-image test. On perception multiple-choice questions, GPT-4o changes from 59.0% with a clean image to 59.5% with no image; LLaVA-1.5-13B stays at 50.0%, while Qwen2-VL-72B drops from 60.0% to 23.5%. On behavior questions, GPT-4o moves from 25.5% to 24.0%, but Qwen2-VL-72B rises from 23.0% to 36.5%. These patterns are too uneven to summarize as “vision is useless,” yet they show why a clean score cannot establish grounding.

![A clean driving question compared with a text-only answer](/assets/images/are-vlms-ready-for-autonomous-driving-drivebench-source-figure-2.webp)
*Fig 2: A model can produce a plausible answer from the question’s camera and coordinate text even when the image is absent; the dataset answer and a GPT score can therefore agree while the response is visually unsupported. | source: [Are VLMs Ready for Autonomous Driving?, Figure 2](https://arxiv.org/abs/2501.04003)*

The mechanism is visible in the question format. A prompt can mention a camera and a coordinate, and the model has learned priors about what commonly appears in that view. The dataset itself is imbalanced: “going ahead” is frequent, and the paper shows that fine-tuning on such data encourages majority-choice memorization. GPT-4o maintains roughly 95% of its clean performance on some text-only open-ended tasks. The answer sounds like driving reasoning, but it can be a completion of a statistical template.

This also exposes a metric problem. ROUGE-L and BLEU reward overlap with a reference answer, and a GPT evaluator gives more varied scores only when it receives the question, rubric, ground truth, and contextual information. A fluent explanation can still be wrong about the object or action. The paper’s Figure 2 is valuable because it makes the failure concrete: the visual answer and the no-image answer look similarly reasonable even though only one had access to the evidence.

### Corruption awareness is prompt-sensitive, not the same as robustness

Humans lose accuracy under corrupted images, while many VLM scores change only slightly. The paper’s radar plot averages GPT scores over 1,261 questions from four driving tasks and shows model traces close to their clean baselines, with text-only scores sometimes higher than visual scores. That pattern can reflect shortcut reliance as much as robustness.

![Performance changes after visual corruptions](/assets/images/are-vlms-ready-for-autonomous-driving-drivebench-source-figure-8.webp)
*Fig 3: Accuracy changes relative to clean inputs across five corruption groups; human performance drops more consistently than most evaluated VLMs. | source: [Are VLMs Ready for Autonomous Driving?, Figure 8](https://arxiv.org/abs/2501.04003)*

A second experiment explicitly names the corruption in the prompt. GPT-4o’s accuracy drops by 8.69 points for bright inputs, 12.98 for dark, 12.94 for camera crash, and 14.30 for H.265 compression. LLaVA-NeXT-7B shows an average reduction of about 19.62 points under these corruption-aware prompts. The authors interpret this as a model becoming less willing to fabricate a confident answer once the prompt acknowledges degraded evidence. That is useful behavior, but it is conditional on the prompt supplying the diagnosis; the model did not reliably detect and communicate its own uncertainty.

DriveBench remains an open-loop, single-frame-centered benchmark built from DriveLM data. It does not establish safe control, closed-loop behavior, or performance on fresh camera rigs and geographies. Its durable contribution is the evaluation question: before celebrating a driving VLM’s language, remove or perturb the evidence and see whether the answer changes for the right reason.

## High-Level Takeaways

- DriveBench informs whether a driving answer is visually grounded, not merely linguistically plausible.
- Clean-versus-black comparisons should be reported alongside class priors, per-task accuracy, and human performance; a stable score can mean robustness or a shortcut.
- Explicit corruption prompts can induce caution, but prompt-conditioned caution is different from autonomous uncertainty detection.
- The next decisive test is a fresh, participant- and geography-held-out benchmark with balanced answers, temporal context, and human disagreement estimates.
