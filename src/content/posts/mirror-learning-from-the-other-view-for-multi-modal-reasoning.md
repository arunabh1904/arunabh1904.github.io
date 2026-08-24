---
title: 'MIRROR: Learning from the Other View for Multi-Modal Reasoning'
date: '2026-07-23T09:00:00.000Z'
section: paper-shorts
postSlug: mirror-learning-from-the-other-view-for-multi-modal-reasoning
legacyPath: /paper shorts/2026/07/23/mirror-learning-from-the-other-view-for-multi-modal-reasoning.html
tags:
  - Multimodal Reasoning
  - Reinforcement Learning
  - Distillation
field: 'Alignment & Post-Training'
topics:
  - multimodal
  - learning
summary: '2026 – MIRROR: Learning from the Other View for Multi-Modal Reasoning'
---

## 2026 – MIRROR: Learning from the Other View for Multi-Modal Reasoning

**arXiv:** [2607.21552](https://arxiv.org/abs/2607.21552)

## Summary

> A geometry problem can be solvable from its text but difficult from its diagram, or the reverse. MIRROR turns this disagreement into supervision. For each problem, it evaluates text-dominant, image-dominant, and combined image-plus-text views, selects the currently strongest view as a teacher, and regularizes students operating on the weaker restricted views toward that teacher.

## Core Insights

The method keeps student rollouts on-policy. It applies ordinary outcome-reward GRPO to the student trajectory, then adds a reverse-KL term computed by rescoring those same tokens under an exponential-moving-average teacher conditioned on the selected view. On a curated 2,000-example geometry dataset, MIRROR improves Qwen3-VL-4B-Instruct beyond single-view and mixed-view GRPO. The result is evidence that paired views need a directed transfer objective; merely placing them in the same RL mixture does not make the successful reasoning path move across modalities.

ODA-Data begins with 97,000 geometry problems from ODA-Math-460k. After difficulty filtering, Gemini-3-Pro-Preview generates and verifies TikZ diagrams and removes from the image-dominant prompt any relations already visible in the diagram. The authors then retain examples that Qwen3-VL-4B-Instruct solves under one view but not the other, yielding about 2,000 paired problems split 85:15 for training and validation. This filtering makes ODA-Val a diagnostic test of modality asymmetry, not a representative sample of general geometry.

![MIRROR selects the strongest view of each problem as a teacher for students operating on restricted text or image views](/assets/images/mirror-reciprocal-reasoning.png)
*Text, image, and combined views expose different bottlenecks. MIRROR selects the best teacher per problem rather than fixing one transfer direction. org/abs/2607.21552). source: [paper](https://arxiv.org/abs/2607.21552)*

![Figure 4 from MIRROR: Learning from the Other View for Multi-Modal Reasoning](/assets/images/mirror-learning-from-the-other-view-for-multi-modal-reasoning-source-figure-4.webp)
*Figure 4 Net solvability gain after training. Each bar shows the change in the fraction of solved examples relative to the base model across 3 random seeds. Each random seed samples 309 problems which is the same size as the validation set. Error bars indicate the min and max values across seeds. MIRROR produces the largest net increase. source: [MIRROR: Learning from the Other View for Multi-Modal Reasoning](https://arxiv.org/abs/2607.21552)*

![Figure 1 from MIRROR: Learning from the Other View for Multi-Modal Reasoning](/assets/images/mirror-learning-from-the-other-view-for-multi-modal-reasoning-source-figure-1.webp)
*Figure 1 Modality-Informed Reciprocal Reasoning Optimization ( MIRROR ). MIRROR exploits view asymmetry by selecting the strongest-performing view of each problem as a teacher and regularizing weaker student views towards the teacher distribution, improving performance without external supervision. source: [MIRROR: Learning from the Other View for Multi-Modal Reasoning](https://arxiv.org/abs/2607.21552)*


For each candidate teacher view, the policy samples 16 rollouts and estimates success. The best view is selected per problem, with ties broken randomly. The student then generates from either the text-dominant or image-dominant prompt. The auxiliary objective compares each sampled student token with the probability assigned to that token by the selected teacher view. Because the teacher only rescores student-generated trajectories, it supplies dense guidance without introducing off-policy teacher states.

| Method | ODA-Val image pass@16 | ODA-Val text pass@16 | GeoInt pass@1 | MathVerse mean |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL-4B base | 42.57 | 80.22 | 58.62 | 41.31 |
| Best single-view GRPO result | 48.78 | 83.16 | 63.02 | 45.22 |
| Mixed-modality GRPO | 45.68 | 81.66 | 61.68 | 44.25 |
| MIRROR | **57.06** | **86.10** | **66.15** | **46.53** |

The adaptive teacher matters because no fixed view dominates. A text teacher produces 51.03 image pass@16, an image teacher 49.10, and the combined teacher 52.28; adaptive selection reaches 57.06. MIRROR also raises the fraction of ODA-Train problems solvable under both restricted views from 42.5% for the base model to 60.7%, compared with 53.6% after standard GRPO.

Stability depends on slowing the teacher. With current-policy teacher scores, entropy rises from about 0.3 to 3.9, reference-policy KL reaches 0.33, and reward falls from 0.34 to 0.21 by roughly step 165. An EMA teacher with decay 0.99 keeps entropy and reference KL near 0.29 and 0.02 while reward reaches 0.48. The reverse-KL coefficient is also narrow: 0.01 ranks best in the reported sweep, while 0.1 collapses training.

## High-Level Takeaways

- MIRROR informs whether paired multimodal examples should enter RL as independent prompts or as linked views of one latent task. Its atomic unit is a student-generated reasoning token, but the supervision unit is the problem-view pair: the policy chooses which view currently carries the strongest evidence and transfers that distribution toward weaker views. For datasets with verified equivalent representations, the reported results favor explicit directional transfer.
- The cost is larger than the 2,000-example headline suggests. The reported jobs use 64 H200 GPUs, approximately 20 minutes per training step, and at least 200 steps—about 4,267 H200 GPU-hours per model. MIRROR uses about 37.5% more FLOPs per update than mixed-modality GRPO, although a matched-cumulative-compute comparison still favors it. At 10× domain breadth, teacher selection rollouts and trustworthy view construction become the likely bottlenecks.
- The missing control is domain transfer without synthetic view filtering. A matched-compute study should construct paired views for charts, scientific figures, and spatial instructions; compare adaptive teachers with uncertainty-weighted soft teachers; and hold the number of student rollouts fixed. The claim should be rejected if gains disappear on naturally occurring paired views or if a cheaper consistency loss matches accuracy without the extra teacher rollouts.
- MIRROR combines on-policy distillation with multimodal consistency, treating disagreement across equivalent views as a training signal rather than only an evaluation failure.
- ODA-Data is geometry-only, synthetically diagrammed, judged by another model, and filtered specifically for view-dependent failures. MathVerse uses an external model judge, and the paper does not establish that the approach transfers to noisy or non-equivalent views.
- Paired modalities become useful supervision when training specifies who teaches whom; a mixed RL batch alone does not reliably transfer reasoning across views.
