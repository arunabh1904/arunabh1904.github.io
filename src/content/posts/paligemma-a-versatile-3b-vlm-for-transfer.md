---
title: 'PaliGemma: A Versatile 3B VLM for Transfer'
date: '2024-07-10T00:00:00.000Z'
section: paper-shorts
postSlug: paligemma-a-versatile-3b-vlm-for-transfer
legacyPath: /paper shorts/2024/07/10/paligemma-a-versatile-3b-vlm-for-transfer.html
tags:
  - Other
field: 'Vision-Language Models'
summary: "2024 – PaliGemma: A Versatile 3B VLM for Transfer"
---

## 2024 – PaliGemma: A Versatile 3B VLM for Transfer

**arXiv:** [2407.07726](https://arxiv.org/abs/2407.07726)

## Summary

> PaliGemma is a transferable base VLM built from a SigLIP-So400m image encoder, a Gemma-2B decoder, and a linear connector. Its main contribution is a disciplined training contract: prefix-LM multimodal pretraining at 224 pixels, short continued runs at 448 and 896 pixels, then task-specific transfer of the whole model. The shared image-in/text-out interface covers captioning, VQA, OCR, detection, segmentation, and video, but the released checkpoints are bases for adaptation rather than ready-made chat models.

## Core Insights

### One text interface can express many visual tasks

![Figure 1 from PaliGemma showing the SigLIP image encoder feeding a Gemma decoder language model](/assets/images/paligemma-a-versatile-3b-vlm-for-transfer-paper-figure.png)
*Fig 1: PaliGemma’s architecture: a SigLIP image encoder feeds a Gemma decoder LM. | source: [PaliGemma, Figure 1](https://arxiv.org/abs/2407.07726)*

Figure 1 looks simple because the interface does most of the work. The SigLIP-So400m encoder turns an image into visual tokens; a zero-initialized linear projection maps them into Gemma-2B’s token dimension; a SentencePiece tokenizer maps a task prefix into text tokens; and the decoder generates a suffix. The suffix may be a caption or answer, a sequence of normalized location tokens for detection, or vector-quantized mask tokens for referring-expression segmentation. The model therefore shares a language-like output contract across tasks instead of adding a separate prediction head for every benchmark.

At 224, 448, and 896 pixels, the image contributes 256, 1,024, and 4,096 tokens respectively. For video or multi-image transfers, PaliGemma encodes each image separately and concatenates the visual tokens; sixteen 224-pixel frames therefore occupy the same 4,096-token visual budget as one 896-pixel image. The choice is useful but expensive: resolution buys detail by increasing both input information and sequence length.

The decoder uses prefix-LM masking. Image and prefix tokens attend bidirectionally, so the visual tokens can read the question or task before the answer is generated. The suffix remains autoregressive and receives the next-token loss. The paper’s ablation finds that extending the autoregressive loss to the prefix supplies extra signal but lowers average transfer performance; the model benefits from asking “what should I output?” rather than from learning to predict its own task prompt.

### Stages separate broad knowledge from usable specialization

PaliGemma starts from public unimodal checkpoints: SigLIP-So400m and raw pretrained Gemma-2B. Stage 1 then trains the complete multimodal model at 224 pixels on one billion examples, with a broad mixture of captioning, OCR, question answering, detection, and segmentation-related sequences. Unlike the common recipe of freezing the visual encoder, Stage 1 updates all components. The authors’ rationale is that captioning and structured tasks provide spatial and relational signals that contrastive pretraining alone does not.

Stage 2 keeps the same basic task mixture but upweights resolution-sensitive tasks and lengthens their suffixes. The 448-pixel checkpoint sees another 50 million examples; the 896-pixel checkpoint sees another 10 million. OCR can request all text in reading order, and detection or segmentation can request all annotated objects, so the model learns to spend the longer context on the cases that need it. Stage 3 fine-tunes all parameters for a target task. The base model is deliberately optimized for transfer density rather than zero-shot usability: a task prefix organizes the pretraining mixture, but the released checkpoint still needs a task-specific suffix format and examples.

The paper also adds 1,024 location tokens (loc0000 through loc1023) for normalized box coordinates and 128 segmentation tokens (seg000 through seg127). They are initialized from a small Gaussian rather than by copying nearby word embeddings. Matching average embedding norms improves the first few steps, but after training the standard initialization gives better perplexity and referring-segmentation transfer. The connector ablation points in the same direction: a simple linear projection reaches a 77.2 average transfer score when all weights are tuned, slightly above the 77.1 from a one-hidden-layer MLP, and is much better when only the connector is trainable.

![Figure 7 from PaliGemma showing the effect of freezing or resetting model parts during Stage 1](/assets/images/paligemma-a-versatile-3b-vlm-for-transfer-source-figure-7.webp)
*Fig 2: The left panel tests whether vision and language weights are tuned, frozen, or reset. The right panel follows pretraining perplexity, exposing effects of a frozen visual encoder that can be less obvious after downstream transfer. | source: [PaliGemma, Figure 7](https://arxiv.org/abs/2407.07726)*

Figure 2 is a diagnostic of the training contract. The left plot compares tuning (T), freezing (F), and resetting (R) the ViT and Gemma components. Resetting pretrained parts causes the largest loss; freezing the language model is especially damaging, while freezing the image encoder is close to full tuning after transfer but worsens spatial-task perplexity during pretraining. The right plot tracks captioning and detection perplexity as pretraining continues, showing where a frozen ViT changes the learning dynamics even when downstream transfer can remain close to full tuning. The lesson is not that every component must always be updated; it is that the base checkpoint is valuable because the multimodal stage can refine pretrained interfaces instead of relearning vision from pixels.

### Resolution helps because it adds detail and capacity

| Task or signal | 224 px | 448 px | 896 px | What changes |
| --- | ---: | ---: | ---: | --- |
| COCO caption CIDEr | 141.9 | 144.6 | — | Global captioning improves modestly |
| VQAv2 accuracy | 83.2 | 85.6 | — | General visual questions benefit |
| TextVQA accuracy | 55.5 | 73.2 | 76.5 | Small text needs the extra visual tokens |
| DocVQA ANLS | 43.7 | 78.0 | 84.8 | Documents are strongly resolution-sensitive |
| RefCOCO testA | 75.7 | 77.9 | 78.7 | Localization gains continue but taper |

The resolution ablation is unusually useful because it separates image detail from model capacity. For resolution-sensitive tasks, feeding a 224-pixel image into a 448-pixel sequence recovers roughly half of the gain: the model gets the longer sequence but not new visual information. Native Stage 2 pretraining is best. Windowing a 448-pixel image into four 224-pixel crops is a fallback when continued pretraining is impossible, but it trails a native 448 checkpoint and saves at most about 5% training time because the Gemma computation is unchanged.

Transfer is also more forgiving of data size than the base-model training cost suggests. With one simple recommended hyperparameter setting, most tasks come within 10% of their full-data score using 4,000 examples and within 20% using 256; in many cases 64 examples are enough to prototype. The exceptions include tasks such as SciCap and RefCOCO, where label smoothing, dropout, or more epochs matter. That result supports PaliGemma as a reusable starting point, not as evidence that few-shot fine-tuning is stable for every output vocabulary.

The reported suite spans almost 40 tasks. At 448 pixels, PaliGemma reaches 144.6 COCO CIDEr, 85.6 VQAv2, 88.5 augmented ChartQA, 73.2 TextVQA, 78.0 DocVQA, and 77.9 RefCOCO testA; the 896 model reaches 76.5 TextVQA, 84.8 DocVQA, and 78.7 RefCOCO testA. Stage 1 takes slightly under three days on TPUv5e-256, each Stage 2 run about fifteen hours, and the paper reports 55% MFU or 5,189 tokens per second per device. These numbers describe a broad transfer recipe with per-task tuning and should not be read as one universal zero-shot score.

### The checkpoint carries an adaptable visual interface

PaliGemma’s breadth comes from learning an interface that survives specialization. Coordinates, masks, answers, and captions share a decoder, while task examples determine which output contract is useful. Resolution then becomes a joint choice about visual evidence and token capacity, particularly for text and small objects. The low-data transfer results suggest that much of this interface is already learned before specialization, but the reported breadth still depends on per-task fine-tuning. A base checkpoint’s transfer potential and its immediate chat behavior are separate properties.

## High-Level Takeaways

- PaliGemma maps SigLIP image tokens and a task prefix into Gemma-2B, then emits captions, answers, boxes, or masks as text.
- Prefix-LM masking lets image tokens read the task before suffix generation, while suffix-only loss and task prefixes make the transfer interface stable.
- Native 448 and 896 checkpoints matter for text-rich and small-object tasks; windowing is a fallback with a measurable quality gap.
- The 3B model is a transferable base: broad results and low-data adaptation are the evidence, while chat behavior still requires a separate transfer recipe.
