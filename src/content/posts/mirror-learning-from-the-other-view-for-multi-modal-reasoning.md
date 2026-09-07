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

![MIRROR selects the strongest view of each problem as a teacher for students operating on restricted text or image views](/assets/images/mirror-learning-from-the-other-view-for-multi-modal-reasoning-source-figure-1.webp)
*Fig 1: Text, image, and combined views expose different bottlenecks, so MIRROR selects the strongest view per problem as the teacher instead of fixing one transfer direction. | source: [MIRROR, Figure 1](https://arxiv.org/abs/2607.21552)*

ODA-Data begins with 97,000 geometry problems from ODA-Math-460k. After difficulty filtering, Gemini-3-Pro-Preview generates and verifies TikZ diagrams and removes from the image-dominant prompt any relations already visible in the diagram. The authors then retain examples that Qwen3-VL-4B-Instruct solves under one view but not the other, yielding about 2,000 paired problems split 85:15 for training and validation. This filtering makes ODA-Val a diagnostic test of modality asymmetry, not a representative sample of general geometry.

For each candidate teacher view, the policy samples 16 rollouts and estimates success. The best view is selected per problem, with ties broken randomly. The student then generates from either the text-dominant or image-dominant prompt. The auxiliary objective compares each sampled student token with the probability assigned to that token by the selected teacher view. Because the teacher only rescores student-generated trajectories, it supplies dense guidance without introducing off-policy teacher states.

| Method | ODA-Val image pass@16 | ODA-Val text pass@16 | GeoInt pass@1 | MathVerse mean |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL-4B base | 42.57 | 80.22 | 58.62 | 41.31 |
| Best single-view GRPO result | 48.78 | 83.16 | 63.02 | 45.22 |
| Mixed-modality GRPO | 45.68 | 81.66 | 61.68 | 44.25 |
| MIRROR | **57.06** | **86.10** | **66.15** | **46.53** |

The adaptive teacher matters because no fixed view dominates. A text teacher produces 51.03 image pass@16, an image teacher 49.10, and the combined teacher 52.28; adaptive selection reaches 57.06. MIRROR also raises the fraction of ODA-Train problems solvable under both restricted views from 42.5% for the base model to 60.7%, compared with 53.6% after standard GRPO.

Stability depends on slowing the teacher. With current-policy teacher scores, entropy rises from about 0.3 to 3.9, reference-policy KL reaches 0.33, and reward falls from 0.34 to 0.21 by roughly step 165. An EMA teacher with decay 0.99 keeps entropy and reference KL near 0.29 and 0.02 while reward reaches 0.48. The reverse-KL coefficient is also narrow: 0.01 ranks best in the reported sweep, while 0.1 collapses training.

The solvability result exposes what aggregate pass rates hide. Figure 4 measures the fraction of sampled ODA-Train problems with at least one successful rollout before and after training. Standard GRPO improves solvability, but MIRROR reaches a net gain of 8.95%, larger than the text-only, image-only, and combined-view baselines. This is evidence that reciprocal teaching changes which problems are reachable, not only the average score on problems already in reach.

![Net solvability gain for text-only, image-only, combined-view, and MIRROR training](/assets/images/mirror-learning-from-the-other-view-for-multi-modal-reasoning-source-figure-4.webp)
*Fig 2: Each bar measures the change in the fraction of solved examples relative to the base model across three random seeds; MIRROR produces the largest net gain. | source: [MIRROR, Figure 4](https://arxiv.org/abs/2607.21552)*

### Where adaptive transfer may stop helping

MIRROR is most useful when views are verified equivalents and one view is measurably stronger on the same problem. The paper's 2,000-example geometry set is also expensive: each job uses 64 H200 GPUs for about 20 minutes per step, at least 200 steps, and about 37.5% more FLOPs per update than mixed-modality GRPO. The unresolved test is whether the gain survives naturally paired charts, scientific figures, and spatial instructions without synthetic filtering under matched cumulative compute. If a cheaper consistency loss matches the adaptive teacher, or if the views are not truly equivalent, the transfer objective has lost its premise.

## High-Level Takeaways

- MIRROR's supervision unit is a paired problem view plus the student's sampled trajectory; the strongest view supplies token-level teacher probabilities to the weaker restricted view.
- Adaptive teacher choice matters because text, image, and combined views are asymmetric per problem; a mixed RL batch does not specify a transfer direction.
- The EMA teacher and reverse-KL coefficient are part of the result: decay 0.99 stabilizes long runs, while the reported 0.1 coefficient collapses training and 0.01 performs best.
- The method improves both pass@16 and net solvability, but its data construction and 64-H200 training cost leave open whether the principle transfers beyond curated geometry.
