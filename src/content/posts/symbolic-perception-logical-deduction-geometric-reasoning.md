---
title: "From Symbolic Perception to Logical Deduction: Guiding Language Models in Geometry"
date: "2026-09-09T00:00:00.000Z"
section: paper-shorts
postSlug: symbolic-perception-logical-deduction-geometric-reasoning
legacyPath: /paper shorts/2026/09/09/symbolic-perception-logical-deduction-geometric-reasoning.html
tags: ["Geometric Reasoning", "Neuro-Symbolic AI"]
field: "Language Models"
summary: "2026 – From Symbolic Perception to Logical Deduction: Guiding Language Models in Geometry"
---

## 2026 – From Symbolic Perception to Logical Deduction: Guiding Language Models in Geometry

**Paper:** [arXiv:2609.10335](https://arxiv.org/abs/2609.10335) · [PDF](https://arxiv.org/pdf/2609.10335)

**Source license:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source figures are reproduced with attribution and converted to WebP.

## Summary

> This framework gives DeepSeek-R1 parsed geometric relations and selected theorem deductions before it writes a solution. It reaches 74.30% on ZhongkaoGeo-L2 versus 67.87% for the reported DeepSeek-R1 baseline, and 78.31% with three-run majority voting. The result supports structured assistance, but the baseline input differences prevent attributing the entire gain to theorem reasoning alone.

## Core Insights

### Convert diagrams into facts before generating the explanation

A geometric diagram carries relations that a caption can miss: points lying on the same line, cyclic order on a circle, angle decomposition, and symbol-to-object assignments. The system detects line segments with a fine-tuned YOLO pose model, uses geometric processing for circles and intersections, detects symbols and text, and reads annotations with Texteller. It assigns each symbol to a nearby compatible primitive and constructs higher-order predicates.

The symbolic module expands simple relations and applies a theorem library. It uses argument ordering, geometric types, and constraints such as parallelism to prune combinations. Without pruning, even a seven-point example can produce 143,640 ordered choices of four line segments. The final prompt contains the problem, parsed relations and point coordinates, and relevant deductions. DeepSeek-R1 then generates the human-readable solution.

Read the pipeline as three different responsibilities. Parsing determines the input facts; deduction adds consequences; language generation turns that structured material into an answer.

![Diagram formalization, theorem reasoning, and language-model answer generation; source Figure 1](/assets/images/symbolic-geometric-reasoning-source-figure-1.webp)
*Fig 1: The framework separates visual parsing from symbolic deductions and final explanation. Errors in the first stage can still contaminate every later deduction. | source: [Paper, Figure 1](https://arxiv.org/abs/2609.10335)*

[View full-size figure](/assets/images/symbolic-geometric-reasoning-source-figure-1.webp)

A worked appendix example shows why the deductions help. The unassisted model attempts a lengthy coordinate derivation and eventually guesses an incorrect angle. The augmented prompt supplies the relevant straight-line angle relation, allowing the solution to subtract a 45-degree triangle angle and a 70-degree corresponding angle from 180 degrees. The benefit is a constrained reasoning path, not additional prose length.

### Read the benchmark with its input conditions

The paper curates 89 problems in ZhongkaoGeo-L1, 83 in L2, and 105 in L3. L1 uses older examinations, L2 uses 2024 and early-2025 material, and L3 uses official 2025 examinations. The first two use strict answer accuracy; L3 uses examination scoring rubrics across subquestions. Recent source dates reduce one contamination concern but do not establish that every evaluated model was trained before those problems existed.

| System | L1 accuracy | L2 accuracy |
| --- | ---: | ---: |
| DeepSeek-R1 baseline | 88.39% | 67.87% |
| Gemini 2.5 Pro | 94.38% | 73.49% |
| Structured framework | 92.13% | 74.30% |
| Framework with majority of three runs | 93.26% | 78.31% |

The implementation section states that ordinary LLM baselines receive problem text, multimodal baselines receive image and text, and the proposed system supplies structured geometric information. Consequently, the DeepSeek-R1 comparison includes better input access as well as theorem guidance. Majority voting also uses additional inference. These are useful system-level results, not a matched-cost measurement of a single reasoning module.

On L3, the parser identifies all tested relation types correctly in 81 problems versus 23 for Qwen2.5-VL-72B in the reported parsing comparison. Yet parsing, theorem coverage, and the final generator remain separate failure points. A language explanation built from verified intermediate deductions is not itself a mechanically checked proof. The strongest next ablation would hold parsed facts and inference budget fixed while adding or removing the theorem deductions.

## High-Level Takeaways

- Specialized parsing and symbolic constraints can supply information a text-only reasoner otherwise lacks.
- Distinguish better perception, a smaller theorem search space, and extra inference attempts when interpreting accuracy gains.
- Use an independent proof checker if correctness must be certified; readable deductions and benchmark accuracy do not provide that guarantee.
