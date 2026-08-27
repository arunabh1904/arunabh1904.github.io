---
title: 'DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model'
date: '2023-10-02T17:59:52.000Z'
section: paper-shorts
postSlug: drivegpt4-interpretable-end-to-end-autonomous-driving-via-large-language-model
legacyPath: /paper shorts/2023/10/02/drivegpt4-interpretable-end-to-end-autonomous-driving-via-large-language-model.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2023 – DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model"
---
## 2023 – DriveGPT4

**arXiv:** [2310.01412](https://arxiv.org/abs/2310.01412)

## Summary

> DriveGPT4 puts natural-language explanation and low-level driving control behind one multimodal language-model interface. It consumes multi-frame video and textual queries, then produces scene-grounded answers, action rationales, and control signals. The paper reports quantitative and qualitative results on BDD-X and compares its domain-tuned system with GPT-4V for driving grounding; the abstract does not disclose a closed-loop safety evaluation.

## Core Insights

The paper's central choice is to make the same model answer a question about a maneuver and predict the maneuver's low-level control. A custom visual-instruction dataset supplies the driving-specific supervision, while the paper's mix-finetuning recipe combines that data with the base model's broader capabilities. This is a tighter coupling than a VLM used only as a captioner, but it does not by itself establish that a fluent explanation caused the control output.

![DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model source figure: DriveGPT4 overview.](/assets/images/drivegpt4-interpretable-end-to-end-autonomous-driving-via-large-language-model-paper-figure.webp)
*Fig 1: DriveGPT4 aligns video and language during pretraining, mixes driving and instruction data during fine-tuning, and decodes both textual explanations and low-level control signals. | source: [DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model](https://arxiv.org/abs/2310.01412)*

![Figure 6 from DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model](/assets/images/drivegpt4-interpretable-end-to-end-autonomous-driving-via-large-language-model-source-figure-6.webp)
*Fig 2: Comparison of DriveGPT4 and GPT4-V. GPT4-V is prompted with BDD-X QA pairs before the comparison. | source: [DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model](https://arxiv.org/abs/2310.01412)*

![Figure 1 from DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model](/assets/images/drivegpt4-interpretable-end-to-end-autonomous-driving-via-large-language-model-source-figure-1.webp)
*Fig 3: BDD-X examples pair video clips with an action description and a natural-language justification, such as stopping for a red light or changing lanes for faster traffic. | source: [DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model](https://arxiv.org/abs/2310.01412)*


BDD-X makes the evaluation legible because it pairs driving video with human-facing explanations. The abstract does not report the action horizon, control representation, loss weighting, training-set size, or a matched ablation that removes explanation supervision while holding control data fixed. Those omissions matter: a shared decoder can correlate language and action without ensuring that its language evidence drives the control decision.

## High-Level Takeaways

- DriveGPT4 makes driving explanation and low-level control co-products of a multi-frame, text-conditioned model rather than separate modules.
- Its reported BDD-X result is evidence for driving grounding and explanation, not a demonstration of closed-loop robustness under distribution shift.
- The expensive decision is whether to train one autoregressive interface for words and controls; a matched control-only versus joint-training study would test whether the language channel improves action rather than merely describes it.
