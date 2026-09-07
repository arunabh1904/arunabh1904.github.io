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

The method changes the reasoning interface. A language-only chain can sound coherent while drifting away from the image. GRIT requires the model to name coordinates as it reasons, which gives the reward function a structural target and the reader an evidence trail.

The coordinates are emitted as tokens inside the same sequence as the language. At inference, GRIT does not crop and re-encode the image after each box; the coordinates point back to the model's existing visual representation and can be rendered as highlighted regions for inspection. That makes the interface cheaper than a visual tool loop, but it also narrows the claim. A plausible box can make a trace look grounded without proving that the selected pixels caused the answer.

![Figure 1 from GRIT: Teaching MLLMs to Think with Images](/assets/images/grit-teaching-mllms-to-think-with-images-source-figure-1.webp)
*Fig 1: A language-only trace is compared with a trace that interleaves coordinate boxes and text; the boxes make the regions referenced by the reasoning visible in the rendered output. | source: [GRIT, Figure 1](https://arxiv.org/abs/2505.15879)*

### A box is a generated token, not a second image lookup

GRIT's output is a pair: a reasoning chain $c$ that interleaves text and coordinate tuples, followed by an answer $a$. Once the model emits a box, the later tokens do not receive a cropped or re-encoded image. They receive the coordinate tokens and must interpret their own grounding action. That makes the representation cheap and keeps generation in one stream, but it also makes the claim narrower than visual tool use: the box can organize the model's reasoning without forcing a new pixel-level observation.

GRPO-GR samples four completions for each input and normalizes their rewards within the group. The reward combines a format score for the <think>/<rethink> structure and valid boxes, an optional count reward for counting examples, and a GPT-4o answer-accuracy score with a small BLEU-1 term. This avoids requiring annotated reasoning traces or box labels. The format reward can tell the policy to emit a box; it cannot by itself tell whether the box names the region that supports the sentence, which is why the paper adds a cross-modal correlation evaluation against randomly drawn boxes.

![Figure 2 from GRIT: Teaching MLLMs to Think with Images](/assets/images/grit-teaching-mllms-to-think-with-images-source-figure-2.webp)
*Fig 2: GRPO-GR samples a group of completions, scores format and valid boxes together with optional counting and answer signals, then uses the group-normalized advantage for the model update. | source: [GRIT, Figure 2](https://arxiv.org/abs/2505.15879)*

The experiment is unusually small and unusually explicit about its boundary. Qwen2.5-VL-3B and InternVL3-2B train for 200 steps on 20 image-question-answer triplets from VSR and TallyQA, using eight A100 GPUs for about 12 hours. Evaluation covers those sources plus GQA, MathVista, MME, and OVDEval; it reports answer accuracy across the sets and grounding IoU where annotated boxes are available. GRIT beats direct-query, chain-of-thought, one-shot, and few-shot SFT baselines on the combined tests, while remaining below human-written traces on the cross-modal correlation measure.

The counting ablation shows why the small dataset is not just a format trick. The full recipe trains on 10 VSR and 10 TallyQA examples with the grounded-target-counting reward. The ablation replaces them with 20 VSR examples and removes that reward; grounding IoU falls from 0.387 to 0.349 on the in-domain set and from 0.437 to 0.378 out of domain, even though in-domain answer accuracy rises from 51.8 to 53.8. The reward therefore teaches a behavior that answer accuracy alone does not capture.

Scaling from 20 to 500 to 7,000 examples improves answer accuracy, but the gain is larger in-domain than on GQA and MathVista-mini, and the curve begins to flatten. The paper's own interpretation is that more diverse supervision may matter more than simply repeating the same mixture. This is the evidence boundary around the headline “20 samples”: it demonstrates that the format can be acquired cheaply, not that arbitrary visual reasoning generalizes from twenty examples.

### The boundary is semantic grounding

GRIT is a good fit when a task has a verifier for the answer and, ideally, for the referenced region. Its coordinate tokens make the reasoning trace auditable without a second crop-and-read loop, but the format reward can be satisfied by a plausible box. The cross-modal correlation test compares generated boxes with random boxes through GPT-4o, and it excludes OVDEval because that set is primarily a grounding task. A stronger test would mask or perturb the referenced regions and measure whether the answer changes for the right reason; the paper shows correlation and grounding IoU, not a complete causal account.

## High-Level Takeaways

- GRIT changes the atomic output from a text-only reasoning chain to text interleaved with coordinate tuples; the model keeps one generation stream instead of invoking a crop-and-read tool after every box.
- GRPO-GR rewards format, optional counting, and answer accuracy, so it avoids dense reasoning and box annotations while leaving semantic grounding partly outside the reward.
- The counting ablation lowers grounding IoU even when one in-domain accuracy number improves, showing why a final-answer metric cannot stand in for process grounding.
- Twenty examples demonstrate data efficiency for this recipe; the 20/500/7,000 sweep shows weaker out-of-domain gains and points to diversity as the unresolved scaling variable.
