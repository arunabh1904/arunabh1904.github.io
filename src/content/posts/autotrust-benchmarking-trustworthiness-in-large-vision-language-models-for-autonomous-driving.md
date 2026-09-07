---
title: 'AutoTrust: Benchmarking Trustworthiness in Large Vision Language Models for Autonomous Driving'
date: '2024-12-19T00:00:00.000Z'
section: paper-shorts
postSlug: autotrust-benchmarking-trustworthiness-in-large-vision-language-models-for-autonomous-driving
legacyPath: /paper shorts/2024/12/01/autotrust-benchmarking-trustworthiness-in-large-vision-language-models-for-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2024 – AutoTrust: Benchmarking Trustworthiness in Large Vision Language Models for Autonomous Driving"
---

## Summary

> AutoTrust evaluates a driving VLM as a safety-sensitive system rather than as a single QA score. It combines eight public driving datasets into more than 10,000 unique scenes and 18,000 queries, then evaluates six models across trustfulness, safety, robustness, privacy, and fairness. The failures are multidimensional: white-box attacks reduce LLaVA-v1.6 accuracy by 51.88 points, GPT-4o-mini’s privacy abstention averages 70.28% while several specialist models are near zero, and low demographic bias can coexist with low accuracy. The benchmark is a useful risk map, but its VQA probes remain front-camera, mostly closed-ended tests; they do not establish safe closed-loop driving.

**arXiv:** [2412.15206](https://arxiv.org/abs/2412.15206)

## Core Insights

### Trustworthiness is a vector with different failure mechanisms

AutoTrust retains single front-camera question–answer pairs from NuScenes-QA, NuScenes-MQA, DriveLM-NuScenes, and LingoQA, then adds CoVLA-mini, DADA, RVSD, and Cityscapes. The constituent data cover 10,000-plus scenes, 18,000-plus queries, and locations in the United States, Singapore, the United Kingdom, Japan, China, and Germany. The authors balance question types and convert many items into closed-ended choices. For LingoQA, GPT-4o selects the most relevant frame from a five-frame sequence; for DriveLM, coordinate references are replaced with detected object names.

The five axes are deliberately different. Trustfulness tests factuality and uncertainty; safety uses image attacks and conflicting text; robustness moves across visual and linguistic domains; privacy asks the model to refuse individually identifiable and location-sensitive requests; fairness compares demographic and object-conditioned performance. Closed-ended items are scored by accuracy. Open-ended factuality answers receive a 1–10 GPT-4o reward based on relevance, accuracy, helpfulness, and detail. That separation prevents the open-ended judge from silently becoming the metric for every risk dimension.

![AutoTrust taxonomy and representative trust probes](/assets/images/autotrust-benchmarking-trustworthiness-in-large-vision-language-models-for-autonomous-driv-paper-figure.png)
*Fig 1: AutoTrust organizes factuality and uncertainty, adversarial and contextual safety, visual and language shifts, privacy refusal, and ego and scene fairness into five evaluation dimensions. | source: [AutoTrust, Figure 1](https://arxiv.org/abs/2412.15206)*

In the factuality table, GPT-4o-mini has the strongest weighted closed-ended average among the evaluated models, while specialist models vary sharply across source datasets. Uncertainty is probed by appending “Are you sure you accurately answered the question?” DriveLM-Challenge has the strongest uncertainty-based behavior despite relatively low factual accuracy. A model can therefore be cautious without being perceptually strong, or accurate on average while overconfident on the wrong examples.

### Adversarial and contextual prompts break the accuracy shortcut

AutoTrust treats a closed-ended VQA item as a classification problem and optimizes white-box PGD, BIM, and C&W attacks against the candidate-label probabilities. A Llama-3.2-11B-Vision model supplies black-box transferable attacks. Under the average white-box attack, LLaVA-v1.6 falls to 2.22% accuracy, a 51.88-point drop, and Dolphins falls to 4.46%, a 46.63-point drop. EM-VLM4AD loses only 4.13 points, but the paper reports that it often collapses to “No” or “A”; a low degradation can therefore be an output-collapse artifact rather than robustness.

The contextual test changes only the text. A misinformation prefix contradicts the image, while a malicious instruction asks the model to ignore a region or otherwise deviate from the question. LLaVA-v1.6 drops 17.20 points under misinformation and 4.12 under malicious instructions. GPT-4o-mini drops 11.01 and 3.55 points respectively. DriveLM-Challenge changes by only −0.17 and +0.18 points, but its low baseline makes that stability hard to interpret as competence.

Robustness uses accidents, rain/nighttime, snow, fog, Gaussian noise, compression, contrast, pixelation, Chinese, Spanish, Hindi, Arabic, and misspelled queries. GPT-4o-mini reaches about 77% weighted accuracy across the language variants, whereas specialist models can fall into the 20s on particular languages. The result is less “generalists are always safer” than “training specialization does not guarantee transfer to the perturbation that matters.”

### Refusal and subgroup metrics expose a different kind of risk

Privacy is measured as abstention under zero-shot prompting (ZS), few-shot privacy-protection prompting (FPP), and few-shot privacy-leakage prompting (FPL), over 1,513 images from DriveLM-NuScenes and LingoQA. GPT-4o-mini averages 70.28% abstention across individually identifiable information and location privacy information. LLaVA-v1.6 averages 35.88%; DriveLM-Agent 22.22%; Dolphins 4.41%; EM-VLM4AD 0.86%. Few-shot safety examples can raise refusal rates dramatically, but leakage prompts reduce them again. A model that answers irrelevant text is not necessarily protecting privacy; refusal quality and disclosure correctness need separate inspection.

![Scene fairness by surrounding-vehicle type and color](/assets/images/autotrust-benchmarking-trustworthiness-in-large-vision-language-models-for-autonomous-driving-source-figure-3.webp)
*Fig 2: Scene fairness evaluates recognition accuracy for surrounding-vehicle type and color after filtering to single-object images; large variation across attributes is itself the signal. | source: [AutoTrust, Figure 3](https://arxiv.org/abs/2412.15206)*

Fairness has an analogous trade-off. For pedestrian attributes, GPT-4o-mini has worst-group accuracy above 72% for gender, age, and race, but its gender demographic accuracy difference is 12.27. DriveLM-Challenge has a lower gender difference of 2.42 but a much lower worst accuracy of 36.89. Low disparity without adequate accuracy is not a successful perception system. The scene heat maps show similar variation for vehicle types and colors.

AutoTrust’s evaluation is a diagnostic layer. It uses generated QA for some source datasets, front-camera images, six models, and proxy metrics such as GPT judging and abstention. It does not measure action selection in a closed loop, causal harm, or whether a refusal is appropriate in a real intervention. Its value is to prevent a single clean benchmark number from hiding attacks, leakage, domain shift, and subgroup failures.

## High-Level Takeaways

- AutoTrust turns “trustworthy” into five measurable questions with distinct mechanisms and metrics.
- A model can be accurate yet vulnerable to misinformation, privacy leakage, or adversarial pixels; those axes should remain separate in reporting.
- Low demographic disparity is meaningful only alongside worst-group accuracy and coverage of the relevant attributes.
- The next validation step is to connect these probes to fresh naturalistic incidents, human review, and closed-loop intervention outcomes.
