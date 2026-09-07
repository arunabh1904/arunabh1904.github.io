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

The figure should be read as a contract between three representations. DriveGPT4 takes front-view monocular RGB video, samples eight frames for a BDD-X example, and predicts the next-step speed and turning angle alongside text. The model is first aligned with 593K CC3M image-text and 703K WebVid-2M video-text pairs, then mix-finetuned on 56K driving and 223K general instruction examples. That staging explains why the overview ends in both words and numbers: the driving data teaches the shared interface, while the general data helps retain language-model behavior. The BDD-X panel is the evidence-bearing part of the story because its human action description and justification sit next to the control target.

![DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model source figure: DriveGPT4 overview.](/assets/images/drivegpt4-interpretable-end-to-end-autonomous-driving-via-large-language-model-paper-figure.webp)
*Fig 1: DriveGPT4 aligns video and language during pretraining, mixes driving and instruction data during fine-tuning, and decodes both textual explanations and low-level control signals. | source: [DriveGPT4, Figure 2](https://arxiv.org/abs/2310.01412)*

![Figure 6 from DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model](/assets/images/drivegpt4-interpretable-end-to-end-autonomous-driving-via-large-language-model-source-figure-6.webp)
*Fig 2: Comparison of DriveGPT4 and GPT4-V. GPT4-V is prompted with BDD-X QA pairs before the comparison. | source: [DriveGPT4, Figure 6](https://arxiv.org/abs/2310.01412)*

![Figure 1 from DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model](/assets/images/drivegpt4-interpretable-end-to-end-autonomous-driving-via-large-language-model-source-figure-1.webp)
*Fig 3: BDD-X examples pair video clips with an action description and a natural-language justification, such as stopping for a red light or changing lanes for faster traffic. | source: [DriveGPT4, Figure 1](https://arxiv.org/abs/2310.01412)*


BDD-X makes the evaluation legible because it pairs driving video with human-facing explanations. The dataset has about 20,000 clips, with 16,803 for training and 2,123 for testing; each clip is represented by eight frames from a front-view monocular RGB camera. The model predicts the next-step speed and turning angle as text-compatible tokens, with the speed in m/s and the relative turning angle in degrees. Stage-one alignment freezes CLIP and LLaMA2 while training the projector on 593K CC3M image-text pairs and 703K WebVid-2M video-text pairs. Stage two first uses 223K general instruction examples and then 56K driving examples, including 16K BDD-X QAs and 40K ChatGPT-generated driving QAs.

The quantitative result is stronger than the short note suggested. On the full BDD-X test set, DriveGPT4 reaches CIDEr 99.10, BLEU-4 18.32, and ROUGE-L 44.73 for combined action description and justification; on the hard split it reaches 57.29, 12.28, and 42.07 respectively. For control, speed RMSE is 1.30 m/s with 60.88% within 0.5 m/s, and turning-angle RMSE is 8.98° with 72.89% within 0.5°. The ablations show why the data recipe is part of the result: removing either BDD-X QAs or ChatGPT QAs lowers the corresponding capabilities, and removing the general-data mix causes a severe drop in broad question answering. Figure 6 adds the complementary qualitative boundary: GPT-4V can describe the scene but does not reliably emit the fixed-format numerical controls.

The evidence remains open-loop and one-step. The BDD-X test split is filtered for inconsistent control and text reasoning, the ChatGPT score is averaged over three unstable evaluations, and the paper does not close the loop in a simulator or vehicle. A fluent explanation can therefore share features with the control prediction without proving that the language evidence caused the action; a control-only versus joint-training comparison would test that causal claim.

## High-Level Takeaways

- DriveGPT4 makes driving explanation and low-level control co-products of a multi-frame, text-conditioned model rather than separate modules.
- Its reported BDD-X result is evidence for driving grounding and explanation, not a demonstration of closed-loop robustness under distribution shift.
- The expensive decision is whether to train one autoregressive interface for words and controls; a matched control-only versus joint-training study would test whether the language channel improves action rather than merely describes it.
