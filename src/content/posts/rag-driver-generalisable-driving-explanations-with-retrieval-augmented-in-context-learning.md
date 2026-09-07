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

> RAG-Driver uses retrieved expert demonstrations as in-context evidence for a multimodal driving model. Its stated goal is to generate a control prediction together with driving explanations and justifications without repeatedly fine-tuning a large model for every domain. On BDD-X it reports strong explanation and control results, and on the unseen Spoken-SAX set it transfers without fine-tuning; the study remains open-loop and does not evaluate closed-loop safety.

## Core Insights

### Retrieval changes the examples seen by the planner

The paper moves adaptation from model weights to the prompt. A retrieval step selects expert demonstrations that the multimodal language model can condition on when interpreting the current driving scene. That is useful when annotations are scarce or data domains differ, because the system can change its evidence set without a training run. It also creates a new deployment dependency: irrelevant or misleading retrieval can change both the explanation and the predicted control.

The retrieval key is deliberately hybrid. LanguageBind encodes an eight-frame, $224\times224$ video sequence into a 1,024-dimensional video vector; a second projector maps the 28-dimensional control signal into the same space, and triplet learning brings scenarios with similar action descriptions and justifications together. At inference, cosine search selects the two nearest driving experiences, then prefixes their video, explanation, justification, and control tokens before the current query. The overview therefore has a useful causal reading: memory changes the context seen by Vicuna-1.5 7B, while the decoder still predicts action explanation, action justification, or the next speed/course/acceleration/curvature signal.

![RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model source figure: RAG-Driver overview from current query and control signal through retrieval to multimodal prediction.](/assets/images/rag-driver-generalisable-driving-explanations-with-retrieval-augmented-in-context-learning-paper-figure.webp)
*Fig 1: The current video and control signal query a memory of driving experiences; two retrieved demonstrations are prefixed to the multimodal language model before it predicts an explanation, justification, or next control signal. | source: [RAG-Driver, Figure 2](https://arxiv.org/abs/2402.10828)*

The video encoder makes retrieval depend on a short motion sequence rather than a single frame. Its eight sampled frames provide the visual embedding, while the control vector adds how the ego vehicle is moving. The retrieval key needs both because visually similar roads can require different maneuvers.

![Figure 3 from RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model](/assets/images/rag-driver-generalisable-driving-explanations-with-retrieval-augmented-in-context-learning-source-figure-3.webp)
*Fig 2: Video Encoder architecture. Video is first split into patches concatenated in time, where these patches are linear projected to video embedding. | source: [RAG-Driver, Figure 3](https://arxiv.org/abs/2402.10828)*


The benchmark protocol makes the claim concrete. BDD-X contributes 77 hours of US driving video, with 16,803 question-answer pairs for training and 2,123 for testing; a second test set, Spoken-SAX, contains 58 London driving questions and is never used for fine-tuning. Videos are uniformly sampled to eight 224 × 224 frames. On BDD-X, the full method reaches action CIDEr 260.8, BLEU-4 34.3, and METEOR 30.7, and justification CIDEr 109.1, BLEU-4 11.1, and METEOR 14.8. On the zero-shot Spoken-SAX set, it reaches action CIDEr 48.9 and justification CIDEr 17.1, versus 5.7 and 4.7 for the base model without retrieval. For control, the BDD-X course RMSE is 4.48° and speed RMSE is 0.69 m/s; the two-example hybrid retrieval condition reaches 88.69% and 85.54% within the 0.5 tolerance for course and speed.

The ablations explain what is doing the work. Visual-only retrieval is weaker than the hybrid video-plus-28-dimensional-control key, and a pretrained MLLM given retrieved examples only at inference produces unusable outputs; the model must be instruction-tuned on the retrieval format. Moving from one to two demonstrations improves action CIDEr from 257.2 to 260.9 and justification CIDEr from 99.3 to 109.1, while slightly worsening speed error. The retrieval engine uses a 1,024-dimensional video embedding, a projected control vector, triplet metric learning, and the two nearest examples. That is the mechanism behind the overview figure's “memory” arrow, rather than a generic prompt prefix.

The deployment boundary is measurable too: training the retrieval engine takes about 30 minutes on one A100, MLLM fine-tuning takes about six hours on eight A100s, and one retrieval-augmented round takes roughly four seconds on one A100. Vicuna-1.5's 4,096-token context limits the system to at most two examples, and the paper has no closed-loop simulator or vehicle evaluation. Retrieval therefore improves open-loop explanations and controls under the reported protocol, while failure detection for a misleading neighbor remains an unresolved safety requirement.

## High-Level Takeaways

- RAG-Driver treats a retrieved multimodal demonstration, not a gradient update, as the primary unit of driving adaptation.
- Its reported zero-shot result supports retrieval as a way to transfer explanation and control behavior, but not as proof that the retrieved evidence is causally used.
- Two demonstrations improve explanation scores but slightly worsen speed error in the ablation. Better retrieved language evidence does not guarantee that every control metric improves.
