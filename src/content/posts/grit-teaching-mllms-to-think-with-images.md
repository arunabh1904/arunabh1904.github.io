---
title: 'GRIT: Teaching MLLMs to Think with Images'
date: '2025-05-21T00:00:00.000Z'
section: paper-shorts
postSlug: grit-teaching-mllms-to-think-with-images
legacyPath: /paper shorts/2025/05/21/grit-teaching-mllms-to-think-with-images.html
tags: [Vision-Language Models, Visual Reasoning]
field: 'Alignment & Post-Training'
summary: '2025 – GRIT: Teaching MLLMs to Think with Images'
---

## 2025 – GRIT: Teaching MLLMs to Think with Images

**arXiv:** [2505.15879](https://arxiv.org/abs/2505.15879)

## Summary

> GRIT trains multimodal models to interleave language reasoning with bounding-box coordinates that identify the image regions used along the way. Its GRPO-GR reinforcement-learning objective rewards final answers and grounded output format without requiring annotated reasoning chains or box labels. The paper reports effective training with as few as 20 image-question-answer examples.

## Core Insights


![Figure 1 from GRIT: Teaching MLLMs to Think with Images](/assets/images/grit-teaching-mllms-to-think-with-images-source-figure-1.webp)
*Fig 1: A language-only trace is compared with a trace that interleaves coordinate boxes and text; the boxes make the regions referenced by the reasoning visible in the rendered output. | source: [GRIT, Figure 1](https://arxiv.org/abs/2505.15879)*

![Figure 2 from GRIT: Teaching MLLMs to Think with Images](/assets/images/grit-teaching-mllms-to-think-with-images-source-figure-2.webp)
*Fig 2: GRPO-GR samples a group of completions, scores format and valid boxes together with optional counting and answer signals, then uses the group-normalized advantage for the model update. | source: [GRIT, Figure 2](https://arxiv.org/abs/2505.15879)*


The method changes the reasoning interface. A language-only chain can sound coherent while drifting away from the image. GRIT requires the model to name coordinates as it reasons, which gives the reward function a structural target and the reader an evidence trail.

Boxes improve inspectability, but they do not prove causal use. A model can learn plausible regions and plausible text together without depending on the selected pixels. Counterfactual masking remains necessary to test whether the grounded trace caused the answer.

### A box is a generated token, not a second image lookup

GRIT's output is a pair: a reasoning chain $c$ that interleaves text and coordinate tuples, followed by an answer $a$. Once the model emits a box, the later tokens do not receive a cropped or re-encoded image. They receive the coordinate tokens and must interpret their own grounding action. That makes the representation cheap and keeps generation in one stream, but it also makes the claim narrower than visual tool use: the box can organize the model's reasoning without forcing a new pixel-level observation.

GRPO-GR samples four completions for each input and normalizes their rewards within the group. The reward combines a format score for the <think>/<rethink> structure and valid boxes, an optional count reward for counting examples, and a GPT-4o answer-accuracy score with a small BLEU-1 term. This avoids requiring annotated reasoning traces or box labels. The format reward can tell the policy to emit a box; it cannot by itself tell whether the box names the region that supports the sentence, which is why the paper adds a cross-modal correlation evaluation against randomly drawn boxes.

The experiment is unusually small and unusually explicit about its boundary. Qwen2.5-VL-3B and InternVL3-2B train for 200 steps on 20 image-question-answer triplets from VSR and TallyQA, using eight A100 GPUs for about 12 hours. Evaluation covers those sources plus GQA, MathVista, MME, and OVDEval; it reports answer accuracy across the sets and grounding IoU where annotated boxes are available. GRIT beats direct-query, chain-of-thought, one-shot, and few-shot SFT baselines on the combined tests, while remaining below human-written traces on the cross-modal correlation measure. Scaling from 20 to 500 to 7,000 examples improves in-domain scores more than out-of-domain scores; the paper therefore points toward diversity as the next bottleneck rather than treating twenty examples as a general data law.

### Decision test and boundary

GRIT is a good fit when the product needs an inspectable region reference and can score answers or counts with a verifier. Its coordinate tokens make the reasoning trace auditable without a second crop-and-read loop, but the format reward can be satisfied by a plausible box. Use the reported cross-modal correlation and masking tests as a grounding check; treat the 20-example result as a demonstration of sample efficiency for this recipe, not as evidence that arbitrary visual reasoning needs only 20 examples.
