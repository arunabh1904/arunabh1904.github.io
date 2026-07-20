---
title: Visual Instruction Tuning (LLaVA)
date: '2023-04-01T04:00:00.000Z'
section: paper-shorts
postSlug: visual-instruction-tuning-llava
legacyPath: /paper shorts/2023/04/01/visual-instruction-tuning-llava.html
tags:
  - Other
field: 'Vision-Language Models'
summary: "2023 – Visual Instruction Tuning (LLaVA)"
---
## 2023 – Visual Instruction Tuning (LLaVA)

**arXiv:** [2304.08485](https://arxiv.org/abs/2304.08485)

**Project:** [llava-vl.github.io](https://llava-vl.github.io/)

**Summary:** LLaVA connects a CLIP-style visual encoder to Vicuna with a learned projection layer, then teaches the combined model to answer image-conditioned instructions. The important move is not a new vision backbone. It is the recipe: use GPT-4 to turn image captions and boxes into instruction-following conversations, then fine-tune the multimodal model on that data.

That made image understanding feel like chat. A model could describe an image, answer questions, and follow open-ended visual instructions instead of only producing class labels or retrieval scores.

## Paper Insights

LLaVA connects a CLIP-style vision encoder to an LLM and instruction-tunes the combined model for visual dialogue. The data move is the key: use GPT-4 to generate image-grounded instruction-following conversations from captions and visual context. Training first aligns visual features to the language model, then tunes for multimodal chat and reasoning. The paper demonstrates that instruction tuning transfers from text-only assistants to visual assistants. The caveat is synthetic supervision: generated data can teach useful behavior, but it may also preserve language priors or miss fine visual details.

![Figure 1: LLaVA network architecture from Visual Instruction Tuning (LLaVA)](/assets/images/visual-instruction-tuning-llava-paper-figure.png)
_Figure 1: LLaVA network architecture. From the [Visual Instruction Tuning (LLaVA) paper](https://arxiv.org/abs/2304.08485), via arXiv HTML._

**What to look at:**
- CLIP image encoder plus Vicuna language model joined by a learned projection layer.
- GPT-4-generated visual instruction data is the enabling data trick, not a new visual backbone.
- Use ScienceQA/chat behavior as a signal, but remember this is still synthetic-instruction tuning.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Interface | Image-to-chat assistant | Turns visual representations into dialogue. |
| Training signal | Visual instruction tuning | Synthetic QA data teaches the model how to answer about images. |
| Artifact | Project page and code | The LLaVA recipe became a common open baseline. |

## Decision Lens

LLaVA informs whether a pretrained visual representation needs architectural reinvention or primarily an instruction-following interface. CLIP patch features pass through a learned projector into the language model, and a two-stage curriculum first aligns that interface and then trains image-grounded dialogue. The expensive asset is not a new encoder but synthetic visual instruction data generated from captions and image context.

The paper establishes that text-style instruction tuning transfers surprisingly well to a visual assistant, but it does not prove that the model's answers are grounded rather than polished expressions of language priors. The key missing ablation pairs equivalent conversations with counterfactual or withheld visual evidence and compares human, synthetic, and caption-only supervision. At ten times the synthetic-data scale, teacher bias and templated reasoning could harden into confident hallucination. The central claim is falsified if performance survives image corruption or swapping, because then the instruction interface has improved conversation without improving perception.

**Context:** LLaVA made the VLM stack modular: vision encoder, projector, LLM, instruction data. That template became the default starting point for many open multimodal assistants.

**Takeaway:** The jump from CLIP to LLaVA is the jump from representation to interface. Once vision features were wired into an instruction-tuned LLM, VLMs became conversational systems.
