---
title: 'Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving'
date: '2023-10-03T00:00:00.000Z'
section: paper-shorts
postSlug: driving-with-llms-fusing-object-level-vector-modality
legacyPath: /paper shorts/2023/10/01/driving-with-llms-fusing-object-level-vector-modality.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2023 – Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving"
---

**arXiv:** [2310.01957](https://arxiv.org/abs/2310.01957) · **Code:** [wayveai/driving-with-llms](https://github.com/wayveai/driving-with-llms)

## Summary

> Driving with LLMs asks whether a pretrained language model can turn structured driving state into actions and explanations. It grounds numeric object vectors in LLaMA-7B through caption pretraining, then learns question answering and control from an RL expert and GPT-generated supervision. The useful result is an interface between metric state and language; the experiments measure offline predictions in a simulator, with closed-loop driving left unresolved.

## Core Insights

### Removing pixels exposes the harder problem of grounding numbers

The input consists of four groups of vectors: route points, nearby vehicles, pedestrians, and ego state. They carry quantities such as relative position, speed, orientation, traffic-light state, and the previous action. These descriptors come from a custom two-dimensional simulator. The system therefore starts with structured scene information already available; its reported “perception” scores evaluate how well the model reads and expresses that information, rather than detecting objects in camera images.

This makes a useful experiment possible. If an LLM receives the right objects and distances but still miscounts agents or misreads a traffic light, the failure lies in the numeric-to-language interface. Upstream detection noise is a deployment concern, but it is not the uncertainty these experiments test.

![Object vectors enter the Vector Encoders and Vector-Former before the language model produces actions and a reason](/assets/images/driving-with-llms-fusing-object-level-vector-modality-source-figure-1.webp)
*Fig 1: Follow the lower path from structured state to language embeddings. The scene is already vectorized before the LLM sees it; the illustrated control loop does not establish closed-loop driving performance. | source: [Driving with LLMs, Figure 1](https://arxiv.org/abs/2310.01957)*

### Caption pretraining teaches the interface before asking it to explain actions

MLPs encode the four vector types, and cross-attention aggregates them into learned latent vectors. The ego feature is added to each latent, anchoring the representation to the controlled vehicle. A Vector-Former then combines latent processing with the question tokens to produce embeddings the language model can consume. LLaMA-7B supplies the pretrained language backbone; LoRA provides trainable adaptation during the second stage.

The first stage freezes the LLM and trains the vector interface to produce structured captions. A deterministic language generator supplies descriptions of positions, agents, and traffic conditions. Training uses 100,000 simulator-derived captioning QA pairs plus 200,000 uniformly sampled random vectors per epoch. Action and expert-attention labels are excluded at this stage: the immediate problem is making a number such as a relative distance recoverable through the language model, before learning what to do about it.

For the second stage, the generator includes actions and attention from a PPO expert. GPT constructs 16 question-answer pairs for each of 10,000 scenarios, and training also includes captioning and action-prediction examples. The vector modules and LoRA are optimized together. Questions that request control receive extra emphasis; accelerator, brake, and steering values are parsed from the generated text using regular expressions.

This supervision also limits what an explanation demonstrates. GPT sees the expert's action when generating its answer, so fluent justifications can be learned alongside action imitation. Agreement between an action and its explanation does not by itself show that the explanation describes the model's causal decision process.

### The action advantage coexists with weak traffic-light numerics

Table 1 evaluates 1,000 held-out simulator scenarios. Caption pretraining reduces pedestrian-count mean absolute error from 1.668 to 0.313 and normalized acceleration/brake-control error from 0.094 to 0.066. Its effect on steering is much smaller: both versions round to 0.014, with exact errors of 0.01441 and 0.01437.

| Held-out measure | Perceiver-BC | LLM without pretraining | LLM with pretraining |
| --- | ---: | ---: | ---: |
| Car-count MAE, lower is better | 0.869 | 0.101 | 0.066 |
| Traffic-light accuracy, higher is better | 0.900 | 0.758 | 0.718 |
| Traffic-light distance MAE, meters | 0.410 | 7.475 | 6.624 |
| Normalized acceleration/brake-control MAE | 0.180 | 0.094 | 0.066 |

The split matters. The pretrained LLM predicts actions more closely to the expert, yet its traffic-light distance error remains over six meters, and pretraining actually lowers traffic-light detection accuracy. A strong aggregate language or action result can coexist with a poor estimate of a quantity that matters for control.

Perceiver-BC uses the same vector interface and roughly the same 25 million trainable parameters, but it lacks the pretrained 7B backbone and the additional driving QA supervision. This is a practical reference point, not a controlled isolation of language reasoning. The authors explicitly acknowledge the training difference.

### Conversational competence has a measurable gap from executable driving

![The model explains braking at a red light and answers a hypothetical question about the light turning green](/assets/images/driving-with-llms-fusing-object-level-vector-modality-source-figure-4.webp)
*Fig 2: The second question changes a condition in language while retaining the observed scene. It demonstrates conditional dialogue about an action, rather than a measured rollout after the traffic signal changes. | source: [Driving with LLMs, Figure 4](https://arxiv.org/abs/2310.01957)*

GPT-3.5 grades answers against the structured observation and question. Pretraining raises its score from 7.48 to 8.39 out of ten; human grading of 230 sampled QA pairs rises from 6.63 to 7.71. The ordering agrees, but the evaluator is lenient: randomly shuffled answers still receive 3.88 from GPT versus 0.26 from humans. Since GPT also generates the training labels, the human check is especially useful, although it remains a small language-evaluation sample.

The paper leaves closed-loop evaluation to future work, citing inference time, numeric inaccuracies, and insufficient command precision. Its RL expert runs in closed loop to collect demonstrations; that does not mean the resulting LLM driver has been validated in the same way. The next decisive experiment would measure whether these predicted commands produce stable driving when their errors alter subsequent observations.

## High-Level Takeaways

- Structured vectors let an LLM consume metric scene state, but successful access to that state must be measured separately from fluent explanations.
- Caption pretraining substantially improves several grounding and action metrics; the traffic-light results show why each control-relevant quantity still needs its own check.
- The Perceiver comparison mixes pretrained knowledge, architecture, and extra QA supervision. It supports the combined system without identifying which ingredient causes the gain.
- The demonstrated capability is offline simulator QA and action prediction. Closed-loop control and faithful causal explanations remain open questions.
