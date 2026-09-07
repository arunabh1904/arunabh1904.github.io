---
title: 'ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation'
date: '2025-03-25T15:18:43.000Z'
section: paper-shorts
postSlug: orion-a-holistic-end-to-end-autonomous-driving-framework-by-vision-language-instructed-action-generation
legacyPath: /paper shorts/2025/03/25/orion-a-holistic-end-to-end-autonomous-driving-framework-by-vision-language-instructed-action-generation.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation"
---
## 2025 – ORION

**arXiv:** [2503.19755](https://arxiv.org/abs/2503.19755)

## Summary

> ORION makes long-horizon visual context and language reasoning part of the trajectory interface. QT-Former compresses scene and history queries into a reasoning stream, an LLM emits a planning token, and a VAE-based generative planner maps that token into multimodal trajectories. On the 220 short routes in Bench2Drive's base set, the paper reports 77.74 Driving Score and 54.62% Success Rate, 14.28 score points and 19.61 percentage points above its stated prior best. The strongest comparison is protocol-specific: the work is simulator closed-loop, with no real-world latency or seed study that establishes deployment readiness.

## Core Insights

### QT-Former makes history an interface

ORION starts from a mismatch between two representations. Vision encoders describe scenes as visual tokens, while the language model reasons over a sequence and the planner needs numeric trajectories. QT-Former inserts a learned bridge. Scene queries summarize the current view, perception queries predict traffic state and motion, and history queries read a FIFO memory bank of prior observations. The resulting planning token lets the language model express a decision without forcing the trajectory decoder to interpret the entire visual history directly.

The history study shows why this compression is selective. With no history queries, the planning-only setting reaches 65.10 Driving Score and 38.83% Success Rate. Eight queries raise these to 68.09 and 39.09; 16 queries reach 74.10 and 44.66. At 32 queries, performance falls to 62.46 and 37.73. The model benefits from a bounded temporal summary, then loses the benefit when the interface becomes too crowded. That is a concrete design boundary rather than a general claim that longer context is better.

![Comparison of driving paradigms and the reasoning-to-action connection](/assets/images/orion-a-holistic-end-to-end-autonomous-driving-framework-by-vision-language-instructed-action-generation-source-figure-1.webp)
*Fig 1: The source comparison places ORION's differentiable reasoning-to-action connection between language reasoning and multimodal trajectory generation. | source: [ORION, Figure 1](https://arxiv.org/abs/2503.19755)*

The figure's important distinction is where the interface lives. A text-only planner can explain an action without constraining the trajectory, while a direct trajectory head can act without exposing the reasoning space. ORION uses the planning token as a shared object so the language and action losses can shape the same decision path.

### A latent planner closes the action gap

The planner uses a VAE to align the latent distribution of driving state and trajectory. Its decoder produces multimodal trajectories from the planning token and vehicle state. The authors also replace it with a diffusion planner that predicts 20 trajectory modes, which tests whether the improvement comes from generative multimodality alone. In the reported comparison, the diffusion variant reaches 71.97 Driving Score, 46.54% Success Rate, 0.73 m open-loop average L2, and 0.96% average collision rate. The VAE variant reaches 77.74, 54.62%, 0.68 m, and 0.47% average collision rate. Its reported ability score is 54.72 versus 46.68 for diffusion.

The result suggests that the latent interface is doing more than adding samples. ORION argues that the VAE's compact distribution is easier to align with the reasoning representation and more stable for multimodal trajectories. That interpretation is supported by the planner swap, although it still depends on the specific training and sampling implementations.

The language data supplies the bridge's supervision. Chat-B2D contains 2.11 million training VQA pairs and 0.12 million validation pairs generated from Bench2Drive scenes with Qwen2VL-72B. Joint VQA and planning training reaches 77.74 Driving Score and 54.62% Success Rate, compared with 74.10 and 44.66 in the planning-only row. The same joint row reports CIDEr 65.77, BLEU 52.49, and ROUGE-L 77.58, so the language task is evaluated alongside the action task rather than treated as an explanation written after the fact.

![ORION architecture from visual history through reasoning to trajectory generation](/assets/images/orion-a-holistic-end-to-end-autonomous-driving-framework-by-vision-language-instructed-action-generation-source-figure-2.webp)
*Fig 2: QT-Former aggregates scene and history queries, the language model produces reasoning and a planning token, and the generative planner emits candidate trajectories. | source: [ORION, Figure 2](https://arxiv.org/abs/2503.19755)*

Follow the data path in the figure: history is compressed before the LLM, the planning token is the handoff, and the VAE turns that handoff into continuous motion. This explains why the auxiliary VQA task can help planning, while also showing the risk: errors in the shared token can affect both language quality and action quality.

### The benchmark result has a clear boundary

Bench2Drive's base set contains 220 short routes and evaluates the policy in a closed-loop simulator. ORION reports 77.74 Driving Score, 54.62% Success Rate, 0.68 m open-loop average L2, 0.47% average collision rate, and 54.72 ability. Those numbers combine route completion, collisions, efficiency, and comfort under the benchmark's controller and discount factors. They do not report real-vehicle latency, seed variance, or a matched test of whether the generated reasoning itself changes action quality.

The practical choice is to preserve a compact history-to-planning interface and train it jointly with action-grounded language tasks. The decision should be revisited if a frozen language model with the same visual history and planner matches the closed-loop score, or if the gain disappears when route length and compute are expanded beyond the base-set simulator.

## High-Level Takeaways

- The temporal bottleneck is a bounded set of history queries: 16 is the reported sweet spot in the planning-only ablation.
- QT-Former and the planning token give language reasoning a differentiable path into trajectory generation.
- The VAE planner beats the diffusion replacement on the reported Bench2Drive and open-loop metrics, supporting the chosen latent interface under this budget.
- The untested part is whether the reasoning token itself changes the trajectory; a no-history and frozen-language control with latency and seed reporting would separate that effect from the planner and simulator.
