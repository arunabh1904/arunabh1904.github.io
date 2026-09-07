---
title: 'SafeAuto: Knowledge-Enhanced Safe Autonomous Driving with Multimodal Foundation Models'
date: '2025-02-28T21:53:47.000Z'
section: paper-shorts
postSlug: safeauto-knowledge-enhanced-safe-autonomous-driving-with-multimodal-foundation-models
legacyPath: /paper shorts/2025/02/28/safeauto-knowledge-enhanced-safe-autonomous-driving-with-multimodal-foundation-models.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – SafeAuto: Knowledge-Enhanced Safe Autonomous Driving with Multimodal Foundation Models"
---
## 2025 – SafeAuto

**arXiv:** [2503.00211](https://arxiv.org/abs/2503.00211)

**Code:** [AI-secure/SafeAuto](https://github.com/AI-secure/SafeAuto)

## Summary

> SafeAuto attaches three different safeguards to a multimodal language model driving stack. Position-Dependent Cross-Entropy (PDCE) makes digit-level control prediction respect numerical proximity; a Markov Logic Network (MLN) checks generated high-level actions against extracted traffic rules; and multimodal retrieval brings similar video, control, and environmental-predicate examples into context. On BDD-X, SafeAuto reports speed/course RMSE of 0.65/3.85; on DriveLM it reports 74.60% behavior accuracy and 0.84 m ADE. The results are offline prediction improvements, not a closed-loop safety guarantee.

## Core Insights

### Make a language loss care about numbers

A token model sees “12.45” as five categorical decisions. Ordinary cross-entropy therefore does not necessarily prefer 12.44 over 19.44, even though the former is numerically closer. SafeAuto replaces the one-hot target at each digit with a Gaussian-like soft distribution centered on the target digit and weights positions by the probability mass of the earlier digits. The leading positions matter more, while nearby digits receive a smoother penalty. Numbers are formatted to a fixed five-digit representation, such as `08.100`, so the loss sees comparable place structure across examples. During fine-tuning, σ is increased from 0.01 to 0.35 to stabilize the transition from sharp token targets to soft numeric supervision.

The overview makes the stack's three interventions visible at once: token loss at the top, rule verification in the middle, and retrieval below. The important intuition is that these are different error surfaces. PDCE changes the shape of numeric mistakes during training; it cannot tell whether braking at a red light is lawful.

![Figure 1 from SafeAuto: Knowledge-Enhanced Safe Autonomous Driving with Multimodal Foundation Models](/assets/images/safeauto-knowledge-enhanced-safe-autonomous-driving-with-multimodal-foundation-models-paper-figure.webp)
*Fig 1: SafeAuto combines position-dependent numeric loss, post-generation rule verification, and multimodal retrieval around one MLLM. | source: [SafeAuto: Knowledge-Enhanced Safe Autonomous Driving with Multimodal Foundation Models, Figure 1](https://arxiv.org/abs/2503.00211)*

### Put traffic rules after generation, where they can reject an action

SafeAuto represents MLLM suggestions, detected environmental conditions, and possible actions as predicates. Its MLN assigns weights to first-order rules such as “solid red light implies no accelerate, left pass, or yield,” infers the most probable safe assignment, and overwrites the original high-level answer when the two conflict. On BDD-X the system defines 16 action predicates, 20 environmental predicates, and 35 formulas; DriveLM uses 7, 29, and 29. A YOLOv8m detector is fine-tuned on LISA's 43,007 frames, which contain 113,888 traffic-light and 7,855 sign annotations, before those observations enter the logic layer.

That explicit correction is useful, but it is also the boundary of the claim. The detector and GPT-4o predicate extraction determine what the checker knows, and the MLN can only enforce rules whose predicates were observed. It is a post-safety verification interface, not a proof that an unseen object or an incorrectly extracted red light is safe.

### Retrieve by scenario structure, not just pixels

The RAG path aligns three modalities—eight-frame video, historical control values, and binary environmental predicates—with a text embedding of the action and justification. LanguageBind compresses video to a 1,024-dimensional vector; BDD-X contributes 28 historical control values (four signals across seven past frames); the predicate vector records conditions such as stop signs and signal state. Projected embeddings are trained to reproduce the pairwise similarity ranking of Sentence-T5-xl text embeddings. At inference, SafeAuto retrieves two BDD-X examples or one DriveLM example.

The result table shows the division of labor. On BDD-X, the final model reaches action BLEU4/CIDEr/METEOR of 38.6/337.4/35.5, versus RAGDriver's 34.3/260.8/30.7; its speed and course RMSE are 0.65 and 3.85, versus 0.69 and 4.48. On DriveLM, behavior accuracy is 74.60%, speed and steering accuracy are 81.61% and 81.90%, and ADE is 0.84 m, compared with DriveLM-Agent's 61.60%, 65.40%, 81.61%, and 1.51 m. Removing environmental predicates largely removes the retrieval benefit, which is why the binary rule context matters more than simply adding another visual neighbor.

![Figure 3 from SafeAuto: Knowledge-Enhanced Safe Autonomous Driving with Multimodal Foundation Models](/assets/images/safeauto-knowledge-enhanced-safe-autonomous-driving-with-multimodal-foundation-models-source-figure-3.webp)
*Fig 2: Across PDCE σ values, speed and course RMSE remain below the original CE baselines, showing the numeric loss is not tuned to one narrow setting. | source: [SafeAuto: Knowledge-Enhanced Safe Autonomous Driving with Multimodal Foundation Models, Figure 3](https://arxiv.org/abs/2503.00211)*

## High-Level Takeaways

- SafeAuto separates numeric precision, rule compliance, and contextual precedent into three interfaces around the same MLLM.
- PDCE preserves autoregressive language output while making nearby digit predictions cheaper than distant ones; it is a training correction, not a runtime safety check.
- The MLN can repair a red-light action only when perception and predicate extraction expose the relevant condition, so the reported gains should not be read as formal guarantees.
- Environmental predicates are the retrieval signal that turns similar-looking episodes into similar driving situations; the paper still needs closed-loop intervention and false-rejection measurements.
