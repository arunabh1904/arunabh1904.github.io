---
title: 'RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model'
date: '2024-02-16T16:57:18.000Z'
section: paper-shorts
postSlug: rag-driver-generalisable-driving-explanations-with-retrieval-augmented-in-context-learning
legacyPath: /paper shorts/2024/02/16/rag-driver-generalisable-driving-explanations-with-retrieval-augmented-in-context-learning.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2024 – RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning"
---
## 2024 – RAG-Driver

**arXiv:** [2402.10828](https://arxiv.org/abs/2402.10828)

## Summary

> RAG-Driver uses retrieved expert demonstrations as in-context evidence for a multimodal driving model. Its stated goal is to generate a control prediction together with driving explanations and justifications without repeatedly fine-tuning a large model for every domain. The paper reports state-of-the-art results on its evaluated explanation and control tasks, plus zero-shot generalization to unseen environments; the abstract does not provide the benchmark breakdown or a closed-loop safety result.

## Core Insights

The paper moves adaptation from model weights to the prompt. A retrieval step selects expert demonstrations that the multimodal language model can condition on when interpreting the current driving scene. That is useful when annotations are scarce or data domains differ, because the system can change its evidence set without a training run. It also creates a new deployment dependency: irrelevant or misleading retrieval can change both the explanation and the predicted control.

The retrieval key is deliberately hybrid. LanguageBind encodes an eight-frame, $224\times224$ video sequence into a 1,024-dimensional video vector; a second projector maps the 28-dimensional control signal into the same space, and triplet learning brings scenarios with similar action descriptions and justifications together. At inference, cosine search selects the two nearest driving experiences, then prefixes their video, explanation, justification, and control tokens before the current query. The overview therefore has a useful causal reading: memory changes the context seen by Vicuna-1.5 7B, while the decoder still predicts action explanation, action justification, or the next speed/course/acceleration/curvature signal.

![RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model source figure: RAG-Driver overview from current query and control signal through retrieval to multimodal prediction.](/assets/images/rag-driver-generalisable-driving-explanations-with-retrieval-augmented-in-context-learning-paper-figure.webp)
*Fig 1: The current video and control signal query a memory of driving experiences; two retrieved demonstrations are prefixed to the multimodal language model before it predicts an explanation, justification, or next control signal. | source: [RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model](https://arxiv.org/abs/2402.10828)*

![Figure 3 from RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model](/assets/images/rag-driver-generalisable-driving-explanations-with-retrieval-augmented-in-context-learning-source-figure-3.webp)
*Fig 2: Video Encoder architecture. Video is first split into patches concatenated in time, where these patches are linear projected to video embedding. | source: [RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model](https://arxiv.org/abs/2402.10828)*


The abstract frames expensive annotation, domain gaps, training cost, and catastrophic forgetting as the inherited constraints. It does not disclose the retrieval embedding, the number of demonstrations, the control representation, or an ablation that separates retrieval quality from in-context reasoning. The central claim would be stronger with a matched retrieval-free prompt, random demonstrations, and an oracle-retrieval condition evaluated on both explanation fidelity and action safety.

## High-Level Takeaways

- RAG-Driver treats a retrieved multimodal demonstration, not a gradient update, as the primary unit of driving adaptation.
- Its reported zero-shot result supports retrieval as a way to transfer explanation and control behavior, but not as proof that the retrieved evidence is causally used.
- At larger deployment scale, retrieval coverage and failure detection are likely to matter more than decoder fluency; random- and oracle-retrieval controls would falsify an apparent retrieval gain.
