---
title: 'PaLM-E: An Embodied Multimodal Language Model'
date: '2023-03-06T00:00:00.000Z'
section: paper-shorts
postSlug: palm-e-embodied-multimodal-language-model
legacyPath: /paper shorts/2023/03/06/palm-e-embodied-multimodal-language-model.html
tags: [Vision-Language-Action, Robotics]
field: 'Vision-Language-Action & Robotics'
summary: '2023 – PaLM-E: An Embodied Multimodal Language Model'
---

## 2023 – PaLM-E: An Embodied Multimodal Language Model

**arXiv:** [2303.03378](https://arxiv.org/abs/2303.03378)

## Summary

> PaLM-E turns sensor readings into the same embedding space as language tokens, then lets a decoder-only language model produce answers, plans, or robot skills. Its central result is transfer: a single model can train on internet vision-language data, language, and several embodied datasets while using a small fraction of embodied samples.

## Core Insights

### Continuous observations as tokens

A decoder-only language model normally consumes a sequence of word embeddings. PaLM-E adds an encoder $\phi$ for each continuous modality—state vectors, image patches, or object-centric scene slots—and projects the resulting vectors into PaLM's token embedding space. A special placeholder in the prompt is replaced by the projected sequence, so a prompt can interleave text and observations such as “What happened between `<img 1>` and `<img 2>`?” The model is still trained with next-token cross-entropy; the target remains text.

The insertion point matters. PaLM-E does not bolt a separate cross-attention tower onto the language model. It dynamically inserts the observation embeddings into the prefix and reuses the language model's positional machinery. For object-centric OSRT inputs, each object slot becomes several embeddings and receives a token such as `<obj 1>`. The generated plan can then refer to that entity, which helps when two objects share a color or shape and cannot be identified by a short noun phrase.

The model is a high-level policy. Its text output names skills from a small low-level vocabulary; a downstream controller executes one skill, the robot observes the new scene, and PaLM-E generates the next decision. That loop lets it replan after a failed grasp or a changed scene, but it leaves metric control, contact dynamics, and control frequency to the low-level policies. The reported PaLM-E sizes combine an 8B/62B/540B PaLM language model with a 4B/22B/22B ViT, producing 12B, 84B, and 562B variants.

### What the model can express

![PaLM-E capabilities across visual reasoning, grounded dialogue, perception, and planning](/assets/images/palm-e-embodied-multimodal-language-model-source-figure-2.webp)
*Fig 1: The large PaLM-E model answers visual questions, performs multimodal reasoning, and produces grounded descriptions and plans; these examples show the breadth of the shared output space rather than a single robot-control benchmark. | source: [PaLM-E, Figure 2](https://arxiv.org/abs/2303.03378)*

Figure 1 is a capability map. The model can answer a visual question or produce a textual plan because both tasks use the same autoregressive interface. The robot-facing interpretation still depends on the skill vocabulary and the controller attached to the generated text. This distinction prevents the general VLM examples from being read as evidence of direct torque or end-effector control.

### Transfer is the main experiment

![PaLM-E transfer across embodied domains with a shared model and mixed pretraining](/assets/images/palm-e-embodied-multimodal-language-model-source-figure-3.webp)
*Fig 2: The paper compares separate in-domain models with one model trained on a mixture of robotics and general vision-language data; the shared mixture improves several embodied tasks despite containing only 8.9% embodied samples. | source: [PaLM-E, Figure 3](https://arxiv.org/abs/2303.03378)*

The full mixture is dominated by WebLI, VQ2A, VQG, Conceptual Captions, and Object Aware data. The paper's sampling table assigns 3.1% to real mobile-manipulator data, 4.2% to Language Table, and 1.6% to TAMP, for 8.9% embodied data in total. This makes the transfer claim testable: the robot tasks are not simply a giant robot-only corpus hidden behind a language model.

In the TAMP environment, training uses only 1% of the full mixture—320 examples for each of the two planning tasks. With OSRT object-centric inputs, PaLM-E reaches 99.7, 98.2, 100.0, and 93.7% on the four embodied VQA tasks and 82.5 and 76.2% on the two planning tasks (Table 1). The OSRT-no-VQA control reaches 71.9 and 75.1% on the planning tasks, so object-centric geometry and co-training matter together. The table is a structured scene experiment, not a claim that the language model discovers all geometry from raw pixels without an encoder.

Language Table supplies a complementary long-horizon test. With the full mixture and a 12B model, Task 1 reaches 80.0% with 40 demonstrations, Task 2 reaches 57.5% with 40, and Task 3 reaches 50.0% with 80. The 84B model reaches 90.0, 53.8, and 64.4% at those same demonstration counts. The model sees a short-horizon language command every four seconds during data collection, so the result measures interactive subtask sequencing rather than a single one-shot trajectory.

### A robot control loop with explicit limits

For mobile manipulation, the model is trained on 2,912 sequences and prompted with a human instruction, the executed step history, and the current image. It emits a next skill until it says “terminate”; the skill is mapped to an RT-1-style low-level policy. The paper reports F1 scores of 0.91 for affordance prediction and 0.77 for failure detection for one PaLM-E-12B configuration in Table 4, with different encoder and freezing rows also reported. These checks explain why a high-level model can close the loop: it must know both whether an action is possible and whether the previous step failed.

The freezing ablations are central. In the TAMP experiments the LLM is frozen, and the encoder/projector learn to ground observations into an existing language space. In the broader mixture experiments, the paper compares frozen and fine-tuned language models; the mobile-manipulation table reports F1 values as high as 0.91 for both failure detection and affordance prediction, depending on the training mixture and freezing choice. The language-retention comparison reports relative natural-language-generation degradation of 87.3% for the 12B model versus 3.9% for the 562B model. These are relative degradation figures, not absolute task-accuracy losses. This makes scale part of the transfer story, while also making compute and data access part of the practical boundary.

PaLM-E's strongest evidence is therefore transfer into structured planning and affordance interfaces. It does not show that text plans replace a robust low-level policy, that object-centric encoders are unnecessary, or that the model is safe under arbitrary language and scene changes. A modern reproduction should report the encoder, freezing choice, low-level skill set, and replanning protocol beside the headline success rate.

## High-Level Takeaways

- PaLM-E inserts projected sensor and scene representations directly into a pretrained language model's token sequence.
- Co-training transfers semantic and visual knowledge into planning and affordance tasks even though only 8.9% of the mixture is embodied data.
- OSRT object slots help entity references and structured TAMP reasoning, but they are a geometric interface learned outside the language model.
- The deployed system is a high-level text policy around low-level skills, with replanning after each executed step.
- The right comparison is a complete control stack with matched encoders, freezing choices, skill vocabularies, and feedback—not a VLM score alone.
