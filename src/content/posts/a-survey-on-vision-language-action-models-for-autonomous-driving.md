---
title: 'A Survey on Vision-Language-Action Models for Autonomous Driving'
date: '2025-06-30T00:00:00.000Z'
section: paper-shorts
postSlug: a-survey-on-vision-language-action-models-for-autonomous-driving
legacyPath: /paper shorts/2025/06/30/a-survey-on-vision-language-action-models-for-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – A Survey on Vision-Language-Action Models for Autonomous Driving"
---
## 2025 – A Survey on VLA Models for Autonomous Driving

**arXiv:** [2506.24044](https://arxiv.org/abs/2506.24044)

**Awesome list:** [Awesome-VLA4AD](https://github.com/JohnsonJiang1996/Awesome-VLA4AD)

### Method and reported result

This VLA-for-autonomous-driving survey provides a taxonomy and a curated bibliography. It treats VLA4AD as a distinct line: models that connect visual perception and language understanding to driving actions rather than stopping at explanation or QA.

## Summary

> The survey is useful as a map. It shows how the field moved from end-to-end perception-to-control, to VLMs that explain scenes, to VLA systems that generate actions with language-aware reasoning.

## Core Insights

The survey formalizes the building blocks of VLA4AD systems, compares more than 20 representative models, and reviews datasets and benchmarks. It separates architectures by how they connect perception, reasoning, and action, then closes with recurring deployment problems: robustness, real-time efficiency, safety, and formal verification.

This is not a method paper, so the value is coverage and vocabulary. It helps place papers like Talk2BEV, DriveLM, OpenDriveVLA, DiffVLA, UniDriveVLA, and DriveVLA-W0 on the same timeline without pretending they solve the same subproblem.

![Figure 2 from the VLA4AD survey showing the architecture blocks for vision-language-action autonomous driving systems](/assets/images/a-survey-on-vision-language-action-models-for-autonomous-driving-paper-figure.png)
_Figure 2 summarizes the VLA4AD architecture blocks, connecting visual inputs, language reasoning, and action generation. From the [survey paper](https://arxiv.org/abs/2506.24044), via arXiv HTML._

**What to look at:**
- The survey separates VLM-as-explainer, modular VLA, end-to-end VLA, and augmented VLA styles.
- Architecture blocks give a vocabulary for comparing otherwise incompatible systems.
- The bibliography is useful for tracking the fast-moving 2024-2026 driving VLA cluster.

**Taxonomy slice:**

| Category | Typical role | Examples to connect |
| -------- | ------------ | ------------------- |
| VLM as explainer | Describe or answer questions about a driving scene | Talk2BEV, DriveLM-style reasoning tasks |
| Modular VLA | Use language reasoning as an intermediate signal | Hybrid VLM plus planner systems |
| End-to-end VLA | Map scene inputs and instructions toward actions | OpenDriveVLA and related action models |
| Augmented VLA | Add tools, chains of thought, or world models | DiffVLA and DriveVLA-W0-style extensions |

### Chronology of the curated list

The table uses each paper's initial arXiv submission date, not the Awesome list's year column or this site's publication date. It gives a reproducible reading order across the two driving fields; it does not, by itself, establish that a later paper was influenced by an earlier one.

| Initial arXiv date | Paper | Main interface or evidence change |
| ------------------ | ----- | -------------------------------- |
| 2023-10-02 | [DriveGPT4](/paper%20shorts/2023/10/02/drivegpt4-interpretable-end-to-end-autonomous-driving-via-large-language-model.html) | One multimodal interface for explanations and low-level controls |
| 2023-11-22 | [ADriver-I](/paper%20shorts/2023/11/22/adriver-i-a-general-world-model-for-autonomous-driving.html) | Interleaved vision-action world-model rollout |
| 2024-02-16 | [RAG-Driver](/paper%20shorts/2024/02/16/rag-driver-generalisable-driving-explanations-with-retrieval-augmented-in-context-learning.html) | Retrieved demonstrations as driving evidence |
| 2024-08-19 | [CoVLA](/paper%20shorts/2024/08/19/covla-comprehensive-vision-language-action-dataset-for-autonomous-driving.html) | Video, language, and trajectory data construction |
| 2024-10-30 | [EMMA](/paper%20shorts/2024/10/01/emma-end-to-end-multimodal-model-for-autonomous-driving.html) | Tokenized multimodal driving inputs and outputs |
| 2025-02-28 | [SafeAuto](/paper%20shorts/2025/02/28/safeauto-knowledge-enhanced-safe-autonomous-driving-with-multimodal-foundation-models.html) | Explicit loss, rule-checking, and retrieval safety interfaces |
| 2025-03-12 | [SimLingo](/paper%20shorts/2025/03/12/simlingo-vision-only-closed-loop-autonomous-driving-with-language-action-alignment.html) | Camera-only language-action alignment |
| 2025-03-14 | [DynRsl-VLM](/paper%20shorts/2025/03/14/dynrsl-vlm-enhancing-autonomous-driving-perception-with-dynamic-resolution-vision-language-models.html) | Dynamic-resolution visual evidence for VLM reasoning |
| 2025-03-25 | [ORION](/paper%20shorts/2025/03/25/orion-a-holistic-end-to-end-autonomous-driving-framework-by-vision-language-instructed-action-generation.html) | Long-history reasoning connected to a generative planner |
| 2025-03-30 | [OpenDriveVLA](/paper%20shorts/2025/03/30/opendrivevla-towards-end-to-end-autonomous-driving-with-large-vision-language-action-model.html) | Hierarchical 2D and 3D spatial-token alignment |
| 2025-04-18 | [LangCoop](/paper%20shorts/2025/04/18/langcoop-collaborative-driving-with-language.html) | Language as a vehicle-to-vehicle communication packet |
| 2025-05-19 | [TS-VLM](/paper%20shorts/2025/05/19/ts-vlm-text-guided-softsort-pooling-for-vision-language-models-in-multi-view-driving-reasoning.html) | Query-conditioned multi-view pooling |
| 2025-05-22 | [DriveMoE](/paper%20shorts/2025/05/22/drivemoe-mixture-of-experts-for-vision-language-action-model-in-end-to-end-autonomous-driving.html) | Separate sparse routing of camera and action experts |
| 2025-05-23 | [FutureSightDrive](/paper%20shorts/2025/05/23/futuresightdrive-thinking-visually-with-spatio-temporal-cot-for-autonomous-driving.html) | Predicted visual future as a planning trace |
| 2025-05-26 | [DiffVLA](/paper%20shorts/2025/05/26/diffvla-vision-language-guided-diffusion-planning-for-autonomous-driving.html) | Vision-language-conditioned diffusion trajectories |
| 2025-05-29 | [Impromptu VLA](/paper%20shorts/2025/05/29/impromptu-vla-open-weights-and-open-data-for-driving-vision-language-action-models.html) | Curated corner-case VLA data and diagnostics |
| 2025-05-30 | [S4-Driver](/paper%20shorts/2025/05/30/s4-driver-scalable-self-supervised-driving-multimodal-large-language-model-with-spatio-temporal-visual-representation.html) | Self-supervised sparse 3D visual representation |
| 2025-06-09 | [ReCogDrive](/paper%20shorts/2025/06/09/recogdrive-a-reinforced-cognitive-framework-for-end-to-end-autonomous-driving.html) | VLM cognition feeding a diffusion planner |
| 2025-06-16 | [AutoVLA](/paper%20shorts/2025/06/16/autovla-a-vision-language-action-model-for-end-to-end-autonomous-driving-with-adaptive-reasoning-and-reinforcement-fine-tuning.html) | Adaptive reasoning plus tokenized feasible trajectories |
| 2025-06-23 | [Drive-R1](/paper%20shorts/2025/06/23/drive-r1-bridging-reasoning-and-planning-in-vlms-for-autonomous-driving-with-reinforcement-learning.html) | RL for visual reasoning-plan alignment |
| 2025-10-30 | [Alpamayo-R1](/paper%20shorts/2025/10/30/alpamayo-r1-bridging-reasoning-and-action-prediction-for-generalizable-autonomous-driving-in-the-long-tail.html) | Causal driving traces with diffusion action prediction |

## High-Level Takeaways

- This survey informs how to partition a driving-VLA research portfolio across perception-language alignment, world modeling, action generation, datasets, and closed-loop evaluation. Its comparison unit is not one token or trajectory but a system interface: visual representation, language/reasoning backbone, action head, and deployment loop.
- The taxonomy is useful only if it predicts which interfaces transfer across papers. A controlled benchmark that fixes sensors, backbone, data, latency, and action space would test that causal value. As the field expands, inconsistent action definitions and mostly open-loop metrics will age the taxonomy faster than model names. The survey's organizing claim would fail if capability and safety differences were explained better by data quality or evaluation protocol than by the proposed architecture categories.
- The survey gives a shared vocabulary for a field where "VLA" can mean anything from QA to closed-loop trajectory generation.
- Use this paper as the index card for the VLA-for-driving literature, then read the individual method papers for the actual design tradeoffs.
