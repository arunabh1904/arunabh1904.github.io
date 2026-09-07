---
title: Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)
date: '2023-10-05T00:00:00.000Z'
section: paper-shorts
postSlug: improved-baselines-with-visual-instruction-tuning-llava-1-5
legacyPath: /paper shorts/2023/10/05/improved-baselines-with-visual-instruction-tuning-llava-1-5.html
tags:
  - Multimodal AI
field: 'Vision-Language Models'
summary: '2023 – Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)'
---

## 2023 – Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)

**Paper:** [arXiv 2310.03744](https://arxiv.org/abs/2310.03744) · **Conference:** [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Improved_Baselines_with_Visual_Instruction_Tuning_CVPR_2024_paper.html) · **Code:** [LLaVA](https://github.com/haotian-liu/LLaVA)

## Summary

> LLaVA-1.5 improves visual instruction tuning through a combination of clearer answer-format supervision, broader VQA data, a two-layer MLP connector, and higher image resolution. The paper's value is in separating these changes: a short-answer benchmark and a visual conversation reward different behavior, and a better connector cannot recover detail that disappeared when the image was resized. Its high-resolution extension makes that last limitation especially concrete.

## Core Insights

### Teach the model which kind of answer the question expects

Adding academic VQA data to [the original LLaVA](/paper%20shorts/2023/04/01/visual-instruction-tuning-llava.html) creates an instruction ambiguity. A question about an image might expect one word in a benchmark or a detailed explanation in a conversation. If the training prompt does not distinguish these cases, the model can learn a general preference for terse answers, losing some of the behavior that made a visual assistant useful.

The paper adds explicit formatting instructions for short answers and multiple-choice letters. The purpose is to condition the response style on the request, rather than make all responses equally short or equally elaborate. Its examples show that the learned behavior can transfer to unseen format requests. On VizWiz's unanswerable questions, explicitly requesting “Unanswerable” when information is insufficient raises the reported result from 11.1% to 67.8%. This is a protocol-sensitive improvement: the model must both recognize insufficient information and express that judgment in the expected form.

The stepwise ablation in Table 2 shows why answer formatting should be separated from architecture:

| Successive change | GQA | MME | MM-Vet |
| --- | ---: | ---: | ---: |
| Original LLaVA plus VQAv2 | 47.0 | 1197.0 | 27.7 |
| Add answer-format prompts | 46.8 | 1323.8 | 26.3 |
| Replace linear connector with MLP | 47.3 | 1355.2 | 27.8 |
| Add open-knowledge VQA and OCR data | 50.0 | 1377.6 | 29.6 |
| Add region-level VQA | 50.3 | 1426.5 | 30.8 |
| Increase image resolution to 336 | 51.4 | 1450.0 | 30.3 |

Formatting helps MME substantially while slightly hurting the other two metrics in its row. The next MLP row improves all three. Higher resolution later improves GQA and MME but does not improve MM-Vet in that comparison. These are measurements of particular capabilities under particular protocols, so a recipe change can help one without moving every score in the same direction.

### A nonlinear connector maps patches; it does not compress them

The connector changes from a linear projection to a two-layer MLP. Each CLIP patch feature still becomes a visual token for the language model. Nonlinearity increases the flexibility of the feature mapping, while the language model itself is updated during visual instruction tuning. This differs from a learned query bottleneck such as [BLIP-2's](/paper%20shorts/2023/01/30/blip-2-bootstrapping-language-image-pretraining.html), where learned queries select information before a frozen language model reads it.

For a 14-pixel patch size, moving from a 224×224 image to 336×336 increases the spatial grid from 16×16 to 24×24: 256 versus 576 patch tokens, an illustrative 2.25-fold increase. The MLP does not remove that increase. Higher resolution can preserve more detail, but the decoder must process more visual context. Choosing a small connector and choosing a small visual-token budget are separate decisions.

Later ablation rows add GQA data, ShareGPT conversations, and a 13B language model. The final 13B result is 63.3 on GQA, 1531.3 on MME, and 36.1 on MM-Vet. The source marks GQA results after its data is introduced because GQA training images were seen during training; those scores should not be presented as zero-shot transfer to an unseen visual dataset. More broadly, the final gain belongs to the complete recipe. The isolated MLP improvement is the smaller difference between its adjacent rows.

### Higher-resolution tiles need both detail and whole-image context

The LLaVA-1.5-HD extension avoids enlarging a pretrained ViT's individual input grid. It splits a larger image into crops at the encoder's native resolution, encodes them independently, and merges their feature maps before passing the visual sequence to the language model. The appendix's HD implementation uses the 224-pixel CLIP encoder, rather than simply feeding arbitrary sizes into the standard model's 336-pixel encoder.

![LLaVA-1.5-HD splitting an image into local tiles and retaining a downsampled global view](/assets/images/improved-baselines-with-visual-instruction-tuning-llava-1-5-source-figure-2.webp)
*Fig 1: The upper path preserves local detail through separately encoded tiles; the lower path supplies a downsampled view of the whole scene. Both feature sequences condition the language model. | source: [Paper, Figure 2](https://arxiv.org/abs/2310.03744)*

Figure 1 shows why the lower path matters. A tile can preserve a person's hand or the edge of the ironing board, but its encoder cannot see how that crop relates to the entire car. The downsampled image restores an overall scene view. The language model receives both kinds of evidence; it must combine them, since the separate tile encoders never jointly attend across crop boundaries.

The appendix adds three details that the diagram leaves implicit. Features belonging only to padding are discarded, reducing wasted visual tokens. A row-end token records the shape of the merged spatial grid before flattening. The input resolution is selected from a predefined set supporting up to six tiles, balancing retained detail against unnecessary padding and compute. Thus the method is flexible across supported shapes, not an unlimited-resolution interface with constant cost.

The HD variant uses the existing alignment setup and proceeds to instruction tuning without a separate high-resolution pretraining stage. That is the efficiency claim: reuse a native-resolution encoder over tiles. It does not mean that extra tiles have no training or inference cost, and the paper lists longer high-resolution training among its limitations.

### The base language model changes what visual knowledge becomes usable

![Normalized benchmark results for different language-model backbones in LLaVA-1.5](/assets/images/improved-baselines-with-visual-instruction-tuning-llava-1-5-source-figure-3.webp)
*Fig 2: Each axis is normalized to the best tested language-model variant on that benchmark. The shapes expose capability differences that an overall score can hide. | source: [Paper, Figure 3](https://arxiv.org/abs/2310.03744)*

In Figure 2, compare MMBench with its Chinese version, MMBench-CN. Vicuna-1.5 and LLaMA-2-Chat are close on the former but separate on the latter. Their shared model family does not imply identical language-conditioning behavior. The paper connects the difference to their language instruction-tuning data, including multilingual ShareGPT conversations. That explanation is plausible and source-supported, but the comparison does not isolate every difference in their training histories.

The TextVQA axis also matters: this evaluation involves both visual text recognition and processing supplied OCR text. A backbone advantage there cannot be assigned wholly to the vision encoder. Since every axis is normalized separately, a radius of 0.9 means 90% of that axis's best variant score, not 90% task accuracy; areas of the plotted shapes are not calibrated overall performance measures.

### Data efficiency and hallucination remain conditional

The final standard model uses 558K alignment examples and 665K instruction examples, approximately 1.2 million combined. These describe the multimodal training stages, not the full cost or data behind the pretrained CLIP and Vicuna components. The paper reports approximately six hours of alignment and twenty hours of instruction tuning on eight A100 GPUs. The frequently repeated “one day” description is therefore a rounded account of this training run.

Randomly retaining half of the instruction samples preserves more than 98% of the reported full-data performance overall. This suggests redundancy in this particular mixture; it does not show that arbitrary half-datasets retain rare capabilities. The full mixture still gives the best overall coverage, and different tasks respond differently to subsampling.

The hallucination discussion adds another cause beyond bad synthetic labels. Even a correct detailed training answer can ask for information that is not visible after the model's input has been downsampled. The model is then rewarded for producing details it cannot reliably recover from its own observation. Better resolution can reduce that mismatch, while clearer prompts control how much detail is requested. Neither intervention proves that all hallucination has been removed.

## High-Level Takeaways

- Explicit response-format instructions let short-answer VQA supervision coexist with more detailed visual conversation.
- The two-layer MLP improves feature mapping but retains the patch-token count; resolution and compression remain distinct costs.
- The HD extension combines local tiles with a global view, removes padding features, and preserves row structure before flattening.
- Adjacent ablation rows isolate modest changes; final checkpoint scores combine data, prompting, resolution, connector, and language-model choices.
- Some apparent hallucination comes from a mismatch between detailed supervision and the visual evidence that survives preprocessing.
