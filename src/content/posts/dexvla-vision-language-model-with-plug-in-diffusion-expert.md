---
title: 'DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control'
date: '2025-02-01T05:00:00.000Z'
section: paper-shorts
postSlug: dexvla-vision-language-model-with-plug-in-diffusion-expert
legacyPath: /paper shorts/2025/02/01/dexvla-vision-language-model-with-plug-in-diffusion-expert.html
tags:
  - Other
field: Robotics
summary: DexVLA paired VLM reasoning with a diffusion policy expert for long-horizon dexterous robot control.
---
## 2025 - DexVLA

**arXiv:** [2502.05855](https://arxiv.org/abs/2502.05855)

**Plain-language summary:** DexVLA separates high-level reasoning from low-level control. A VLM-style module processes images and instructions, producing reasoning and action tokens. A diffusion policy expert then turns that guidance into continuous robot actions.

This hybrid design is useful for dexterous, long-horizon tasks where pure language-model action generation may be too coarse and pure diffusion may lack semantic planning.

![Vision-language-action stack schematic](/assets/images/robot-vla-stack-schematic.svg)

**What to look at:**
- The VLM handles high-level reasoning tokens and action guidance.
- A diffusion policy expert generates continuous low-level actions.
- The curriculum separates general motor skill, embodiment adaptation, and task tuning.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Architecture | VLM plus diffusion expert | Splits planning from precise control. |
| Training | Three-stage embodied curriculum | Moves from general motor skills to task specialization. |
| Best fit | Dexterous long-horizon tasks | Where pure VLM action output is too coarse. |

**Why it mattered:** DexVLA shows how robotics can borrow from both sides of modern AI: language models for task structure and diffusion models for continuous trajectory generation.

**Take-home message:** The strongest robot policies may be coordinated systems, not monoliths. Let the VLM plan; let the control expert execute.
