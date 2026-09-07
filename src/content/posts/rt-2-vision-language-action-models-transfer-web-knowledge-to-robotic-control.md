---
title: 'RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control'
date: '2023-07-28T00:00:00.000Z'
section: paper-shorts
postSlug: rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control
legacyPath: /paper shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html
tags:
  - Robotics
  - Vision-Language-Action
field: 'Vision-Language-Action & Robotics'
summary: "2023 – RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control"
---

## 2023 – RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control

**arXiv:** [2307.15818](https://arxiv.org/abs/2307.15818)

**Project:** [robotics-transformer2.github.io](https://robotics-transformer2.github.io/)

## Summary

> RT-2 turns low-level robot control into another output language for a pretrained vision-language model. It discretizes end-effector actions into tokens, co-fine-tunes them with web-scale image-language tasks, and decodes the resulting sequence back into closed-loop motion.

## Core Insights

### The bridge is a shared output alphabet

RT-2 starts with a mismatch between what a VLM knows and what a robot needs. A VLM can answer a question about a picture; a robot needs a small Cartesian displacement and a gripper command at every control step. The paper's solution is to represent six-dimensional end-effector position and rotation, gripper extension, and episode termination as eight discrete fields. Each continuous dimension is uniformly quantized into 256 bins and serialized as an action string alongside ordinary text targets.

For PaLI-X, integer tokens already exist for the relevant range, so the action bins can be assigned to those tokens. PaLM-E lacks that convenient numeric vocabulary, so the authors repurpose 256 of its least frequently used tokens. When the prompt asks for a robot action, decoding is constrained to valid action tokens; ordinary VQA prompts retain the full language vocabulary. The same transformer can therefore train on a caption, a VQA answer, or an action sequence with the same next-token objective.

That bridge is powerful because it shares weights at the point where semantic concepts become predictions. It is also a real control compromise: a physical error is turned into a token error, and the model's probability over neighboring bins is not a calibrated metric loss.

![RT-2 evaluation scenes for unseen objects, backgrounds, and environments](/assets/images/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control-source-figure-3.webp)
*Fig 1: The controlled evaluation changes object categories, scene backgrounds, and kitchen environments separately, with easy and hard variants that expose where visual transfer survives a distribution shift. | source: [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control, Figure 3](https://arxiv.org/abs/2307.15818)*

The source figure is useful because it shows what “generalization” means in this paper. A new object, a new visual context, and a new physical workspace are different tests; they should not be collapsed into one open-world score. The robot still executes familiar pick, place, and move behaviors while the semantic and visual conditions change.

### Co-fine-tuning protects the visual prior

RT-2 adapts PaLI-X (5B and 55B) and PaLM-E (12B) VLMs. The robot data comes from 13 mobile manipulators collected over 17 months, with instructions built from a skill verb and the manipulated nouns. The web mixture includes about 10 billion WebLI image-text pairs across 109 languages, filtered to its top 10% by cross-modal similarity, plus captioning, VQA, and language data. For RT-2-PaLI-X, robot data is weighted to about 50% of the training mixture; for RT-2-PaLM-E it is about 66%.

The important control is not simply “pretrained versus unpretrained.” The authors compare a 5B PaLI-X model trained from scratch, fine-tuned from a VLM checkpoint using robot data only, and co-fine-tuned with robot and original web data interleaved. On the unseen-object, background, and environment average, the 5B variants score 9 from scratch, 42 with robot-only fine-tuning, and 44 with co-fine-tuning. At 55B, robot-only fine-tuning reaches 52 and co-fine-tuning 63. The paper attributes the gap to retaining concepts learned during VLM pretraining; the comparison also changes the data mixture, so it is evidence for the full recipe rather than a pure memory-retention measurement.

![RT-2-PaLI-X scaling and training ablation across unseen settings](/assets/images/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control-source-figure-8.webp)
*Fig 2: The ablation compares scratch training, robot-only fine-tuning, and co-fine-tuning at 5B and 55B; capacity and retained web data both raise the average unseen-condition score. | source: [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control, Figure 6](https://arxiv.org/abs/2307.15818)*

The bars make the interaction visible. A large randomly initialized model is not enough, and a pretrained model can still forget useful web concepts when robot actions are the only fine-tuning signal. Keeping the original tasks in the mixture is the paper's way of making low-level action learning coexist with semantic recognition.

### The generalization test is still a robot test

The main study uses a seven-degree-of-freedom mobile manipulator and about 6,000 evaluation trajectories. Seen instructions cover more than 200 tasks: picking, knocking over, placing upright, moving, opening or closing drawers, and moving objects into or out of receptacles. The held-out evaluation contains more than 280 pick-and-place scenarios divided into unseen objects, backgrounds, and environments, each with easy and hard cases.

| Model | Seen | Unseen objects | Unseen backgrounds | Unseen environments | Unseen average |
| --- | ---: | ---: | ---: | ---: | ---: |
| RT-1 | 92 | 31 / 43 | 71 / 9 | 26 / 14 | 32 |
| RT-2-PaLI-X 55B | 91 | 70 / 62 | 96 / 48 | 63 / 35 | 62 |
| RT-2-PaLM-E 12B | 93 | 84 / 76 | 75 / 71 | 36 / 33 | 62 |

Each slash is easy / hard. RT-2 keeps seen-task performance near RT-1 while roughly doubling the unseen average relative to RT-1 and MOO. The result supports a precise claim: web-scale visual and semantic pretraining transfers when the robot can reuse an existing physical skill. It does not show that the robot has acquired a new grasp, a new contact strategy, or a new dynamics model.

The open Language-Table experiment tests that interface on a different simulated robot. A 3B PaLI model predicts two discretized end-effector deltas at 5 Hz and reaches $90\pm10$ success, compared with $74\pm13$ for RT-1, $77\pm4$ for LAVA, and $72\pm3$ for BC-Zero. The simulated action format is smaller than the real-world eight-field string, so this result demonstrates transfer of the training recipe rather than a direct hardware comparison.

### Emergent behavior comes from recombination

The quantitative emergent suite runs each instruction five times in an A/B framework, evaluating four models under the same scene conditions. Symbol tasks ask the robot to move objects near numbers, logos, or matching cards. Reasoning tasks add visual relations, arithmetic, nutrition, color, and multilingual commands. Human-recognition tasks refer to celebrities or people with a visual attribute.

RT-2-PaLI-X-55B averages 82 on symbol understanding, 46 on reasoning, and 53 on person recognition, for 60 overall; RT-2-PaLM-E-12B averages 36, 43, and 43, for 40 overall; RT-1 averages 17. The PaLM-E model is better on the math subset, which the authors associate with its pretraining mixture. These are still learned motions applied to new semantic targets. For example, “put the strawberry into the correct bowl” asks the model to identify a relation and then reuse pick-and-place, while “use something to hammer a nail” can elicit a rock as the object choice without teaching a hammering motion.

A second experiment adds a natural-language Plan before the action tokens. After a few hundred gradient steps, the model can produce traces such as “Plan: pick the energy drink” followed by an action sequence. The paper presents this as qualitative evidence that a shared VLA can expose a planning-like text step; it is not a separate quantitative proof of multi-step reasoning.

### Runtime and limits matter

The 55B model runs at only 1–3 Hz when served by a multi-TPU cloud service; the 5B model runs at about 5 Hz. Network inference makes the shared VLM practical for the tested setup, but high-frequency control remains expensive. The output vocabulary constraint protects execution, while the action quantization and autoregressive decoding impose latency and resolution limits.

The paper is explicit about the boundary. RT-2 does not learn new physical motions from web data, and its demonstrations cover only a small set of skills. In the Language-Table failure cases, the policy can attend to the correct object but cannot predict the dynamics of a rolling pen or banana. It also struggles with grasping by a particular handle, wiping or tool use, dexterous folding, and long chains of indirection. The durable lesson is to use web data for semantic coverage while measuring contact and recovery separately.

## High-Level Takeaways

- Action tokens let a VLM share its language and vision parameters with a robot policy, but token likelihood is still an imperfect physical objective.
- Co-fine-tuning preserves useful web behavior better than the tested robot-only recipe; the result belongs to the whole data and initialization protocol.
- The strongest gains are semantic recombinations of demonstrated motions: symbols, relations, multilingual instructions, and object choice.
- A 55B model reaches 62% unseen-condition average in the reported suite while running at 1–3 Hz, making compute and latency part of the control design.
- Web knowledge expands where learned skills can be applied; it does not provide contact dynamics, new motions, or recovery coverage.
