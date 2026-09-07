---
title: 'Distilling Multi-modal Large Language Models for Autonomous Driving'
date: '2025-01-16T00:00:00.000Z'
section: paper-shorts
postSlug: distilling-multimodal-large-language-models-for-autonomous-driving
legacyPath: /paper shorts/2025/01/01/distilling-multimodal-large-language-models-for-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2025 – Distilling Multi-modal Large Language Models for Autonomous Driving"
---

## Summary

> DiMA uses a multimodal language model as a training-time source of structured scene knowledge while keeping a vision planner in the runtime loop. A shared scene encoder turns multi-view images into BEAM tokens for bird’s-eye view, ego, agents, and map; the language branch learns planning, visual QA, distillation, masked reconstruction, future prediction, and scene editing, while the planner learns trajectory constraints. On the full nuScenes validation split under the paper’s standardized evaluation, DiMA-VAD-Tiny reduces average trajectory L2 from VAD-Tiny’s 0.78 m to 0.51 m and average collision rate from 0.30% to 0.06%. The long-tail gains are strongest on overtaking and three-point turns, but the benchmark is open-loop and uses manually selected scenarios, so the result is evidence for efficient representation transfer rather than closed-loop safety.

**arXiv:** [2501.09757](https://arxiv.org/abs/2501.09757)

## Core Insights

### A shared scene tokenizer lets the teacher shape the planner

DiMA starts from a practical deployment constraint: an MLLM can reason about unusual traffic situations, but calling it for every trajectory is slow and expensive. The authors decompose a vision-based planner into a scene encoder and a planning transformer. The encoder is not just a frozen visual front end. It learns structured BEAM representations—bird’s-eye-view, ego, agent, and map tokens—and passes those same scene components to the MLLM as a trainable tokenizer.

The language branch is trained for trajectory estimation and visual question answering, while the planner is trained for the usual waypoint and collision constraints. A distillation loss aligns the planner with penultimate-layer ego representations from the MLLM. The MLLM also receives three surrogate objectives: reconstruct masked BEV tokens, predict future BEV tokens at later times, and edit the scene by adding or removing an agent and predicting the ego response. These objectives are designed to force scene representations to carry spatial, temporal, and interaction information instead of only matching a language answer.

![DiMA’s BEAM-token and distillation architecture](/assets/images/distilling-multimodal-large-language-models-for-autonomous-driving-source-figure-2.webp)
*Fig 1: A multi-view scene encoder produces BEAM tokens for the planning transformer and the MLLM; language, distillation, and surrogate losses shape the shared representation during training. | source: [Distilling Multi-modal Large Language Models for Autonomous Driving, Figure 2](https://arxiv.org/abs/2501.09757)*

The training sources make the supervision concrete. NuScenes provides about 28,000 open-loop planning samples with 22,000/6,000 training/validation examples. DriveLM contributes a 4,000-sample subset with 300,000 QA pairs, and the authors generate additional QA from numerical annotations with Llama-3-70B. The MLLM can be removed after training when the vision branch is used for planning; an optional dual branch combines both at inference for a slower, language-aware mode.

### The useful comparison is matched by evaluation protocol

Planning predicts two waypoints per second for three seconds. The paper reports both the standardized evaluation proposed by PARA-Drive and the older VAD evaluation, because averaging time steps and handling invalid frames differently changes the numbers. Under standardized full-validation evaluation, VAD-Tiny reports 0.78 m average L2 and 0.30% average collision rate. DiMA-VAD-Tiny reports 0.51 m and 0.06%; DiMA-VAD-Base reports 0.47 m and 0.06%. These are open-loop metrics, not a claim about an interactive vehicle.

![DiMA on long-tail overtaking and three-point-turn scenarios](/assets/images/distilling-multimodal-large-language-models-for-autonomous-driving-source-figure-1.webp)
*Fig 2: On manually selected nuScenes long-tail cases, DiMA lowers trajectory L2 for overtaking from VAD’s 1.06 m to 0.66 m and for a zero-shot three-point turn from 1.57 m to 1.05 m; the latter appears only in validation. | source: [Distilling Multi-modal Large Language Models for Autonomous Driving, Figure 1](https://arxiv.org/abs/2501.09757)*

The targeted split contains 689 validation scenes where the ego vehicle must turn. DiMA-VAD-Tiny reaches 0.81 m average L2 versus VAD-Tiny’s 1.08 m, while DiMA-VAD-Base reaches 0.71 m. In the long-tail table, overtaking improves from 1.14 m for VAD-Tiny to 0.69 m for DiMA-VAD-Tiny, and the zero-shot three-point-turn case improves from 1.55 m to 1.17 m. The visual figure is persuasive because it shows the path, but the table is the proper source for aggregate comparisons.

Under VAD evaluation, the runtime trade-off is also explicit. DiMA-VAD-Tiny retains 16.8 FPS and 59.5 ms latency, while DiMA-VAD-Base is 4.5 FPS and 226 ms. The optional DiMA-Dual VAD-Tiny is 3.5 FPS and 286 ms. The LLM-free vision branch is therefore the deployment story; the MLLM branch is a diagnostic or optional language interface.

### Ablations show what the student actually receives

The eight-step VAD-Tiny ablation separates architecture from supervision. The baseline has 0.60 m average L2 and 0.29% average collision under VAD evaluation. Adding language training with only BEV tokens gives 0.62 m/0.26%, while adding all BEAM tokens improves to 0.52 m/0.21%. Explicit MLLM-to-planner distillation reaches 0.48 m/0.19%. Masked reconstruction reduces the pair to 0.42 m/0.18%, future prediction to 0.39 m/0.16%, and all three surrogate tasks to 0.38 m/0.15%.

This sequence is the paper’s strongest mechanistic evidence. The gain does not come from a single “LLM magic” switch. It accumulates as the student receives richer scene structure, an aligned planning representation, and tasks that teach it to preserve context across space and time. Scene editing is especially interesting because it creates a counterfactual agent and asks whether the ego trajectory should change; that is closer to interaction reasoning than a captioning loss.

The boundary is equally important. The teacher’s generated QA and hidden representations can contain its own biases, and the long-tail cases are a selected validation subset rather than a random estimate of rare-event safety. Open-loop nuScenes L2 and collision metrics do not capture closed-loop compounding error, human interaction, or distribution shift. DiMA makes a convincing case for using language models as offline teachers and structured representation builders; it does not show that the teacher’s reasoning is necessary for every gain or that the distilled planner is safe in deployment.

## High-Level Takeaways

- DiMA’s practical idea is to use an MLLM while training, then discard it for efficient vision-only planning.
- BEAM tokens and the ablation sequence explain the gain better than a single teacher-versus-student number.
- Long-tail results are encouraging, but standardized versus VAD evaluation must stay attached to every comparison.
- The decisive next test is closed-loop, held-out rare-event evaluation with equal-volume non-LLM supervision and a real latency budget.
