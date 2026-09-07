---
title: 'BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation'
date: '2022-01-28T00:00:00.000Z'
section: paper-shorts
postSlug: blip-bootstrapping-language-image-pretraining
legacyPath: /paper shorts/2022/01/28/blip-bootstrapping-language-image-pretraining.html
tags: [Vision-Language Models, Multimodal Generation]
field: 'Vision-Language Models'
summary: '2022 – BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation'
---

## 2022 – BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

**arXiv:** [2201.12086](https://arxiv.org/abs/2201.12086)

**Code:** [salesforce/BLIP](https://github.com/salesforce/BLIP)

## Summary

> BLIP makes one vision-language pre-training system serve both understanding and generation. Its multimodal mixture of encoder-decoder (MED) shares most of the text stack while changing attention masks for contrastive alignment, image-text matching, and captioning. A separate captioner writes synthetic descriptions for web images, and an independently tuned filter removes mismatched original and synthetic captions before a fresh pre-training run. In the paper’s experiments, this combination improves retrieval, captioning, VQA, and zero-shot video transfer; its main assumptions are compatible task heads and a filter that can reject its captioner’s mistakes.

## Core Insights

### One MED changes role by mask

![Figure 2 from BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](/assets/images/blip-bootstrapping-language-image-pretraining-source-figure-2.webp)
*Fig 1: The same MED alternates among contrastive alignment, pair matching, and caption generation by changing its attention masks; colors mark the parameter groups shared across these roles. | source: [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation, Figure 2](https://arxiv.org/abs/2201.12086)*

BLIP starts with a ViT-B/16 image encoder and a BERT-base text transformer. The image encoder produces a sequence of patch features plus a global image token. The same text-side parameters then serve three interfaces. A unimodal text encoder and image encoder produce global embeddings for image-text contrastive learning (ITC). An image-grounded text encoder adds cross-attention and uses an `[Encode]` token for image-text matching (ITM). An image-grounded text decoder swaps bidirectional self-attention for causal self-attention, uses `[Decode]`, and predicts a caption with an autoregressive language-modeling (LM) loss.

The objectives ask different questions of the same pair. ITC asks whether the image and sentence occupy compatible global positions; a momentum encoder supplies targets and soft labels account for possible positives among the negatives. ITM asks for fine-grained compatibility and mines harder negative pairs from the batch. LM asks the model to turn visual evidence into a coherent sequence, using cross-entropy with label smoothing of 0.1. Each image therefore makes one expensive vision pass and three text passes, with the masks deciding which evidence each objective can use.

The sharing boundary is deliberate. The text encoder and decoder share embeddings, cross-attention, and feed-forward layers, but keep separate self-attention layers because one reads both directions and the other predicts left to right. In the paper’s 14M-image ablation, sharing everything gives 224M parameters and COCO retrieval TR@1 77.3; sharing everything except self-attention gives 252M parameters and TR@1 78.4, with caption CIDEr rising from 125.9 to 127.8. Keeping no layers shared needs 361M parameters and does not improve those scores. The result is a compact architectural lesson: share the transformations that carry multimodal content, and leave the directional language operation task-specific.

### CapFilt uses one model to create data and another copy to judge it

![Figure 4 from BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](/assets/images/blip-bootstrapping-language-image-pretraining-source-figure-4.webp)
*Fig 2: CapFilt compares an original web caption and a generated caption for the same image; green text survives the image-text filter and red text is removed. | source: [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation, Figure 4](https://arxiv.org/abs/2201.12086)*

The data problem is noisy alt text. BLIP pre-trains on about 14M images from COCO, Visual Genome, Conceptual Captions 3M and 12M, and SBU; an expanded experiment adds 115M images from LAION for a 129M-image mixture. The authors first fine-tune two copies of MED on COCO. The captioner is the image-grounded decoder and writes one synthetic caption for each web image. The filter is the image-grounded encoder, fine-tuned with ITC and ITM. Its ITM head then accepts or rejects each original web caption and synthetic caption according to whether it matches the image. Only the accepted pairs, together with human-annotated pairs, form the dataset for a fresh BLIP pre-training run.

Figure 2 makes the division of labor concrete. A web caption such as “from bridge near my house” can be rejected even when it is grammatical, while a generated description such as “a flock of birds flying over a lake at sunset” can be kept when it names visible content. Synthetic text is therefore a proposal, not a label; original alt text is a proposal too. The filter applies the same image-text matching test to both.

The ablations explain why generation diversity and judge independence matter. Here, the noise ratio is the fraction of synthetic captions that the image-text matching filter rejects as unmatched. Nucleus sampling with cumulative probability threshold $p=0.9$ produces a 25% rejection rate, compared with 19% for beam search, yet its cleaned data yields COCO retrieval TR@1/IR@1 of 80.6/63.1 versus 79.6/61.9 for beam search. The authors attribute the gain to more varied captions that add information beyond common phrases. When the captioner and filter share parameters, the rejection rate falls to 8% and the downstream scores are lower than with decoupled copies; the paper connects this to confirmation bias, because a judge too close to its generator is less likely to reject the generator’s errors.

The improvement is also a data-quality effect rather than just extra steps. When the original web text is replicated to match the bootstrapped dataset’s samples per epoch, the noisy control reaches COCO retrieval 78.3/60.5 while CapFilt reaches 80.6/63.1. Table 13 makes the retraining comparison more precise: continuing the old model gives NoCaps CIDEr 104.5 versus 105.1 for a fresh model, but COCO caption CIDEr 129.9 versus 129.7, respectively. The continuation result is therefore mixed rather than uniformly weaker; the cleaner distribution changes what the model learns, so simply showing it more often is not an equivalent intervention.

### The gains travel across tasks, with clear measurement boundaries

| Evaluation | Reported BLIP result | What the comparison means |
| --- | --- | --- |
| COCO retrieval, 14M pre-training images | TR@1 80.6 and IR@1 63.1, versus ALBEF’s 77.6 and 60.7; +2.7 average recall@1 | Fine-tuned image-text ranking on the same nominal pre-training scale |
| Flickr30K retrieval after COCO fine-tuning | Zero-shot TR@1 94.8 and IR@1 84.9 | Transfer to a different image-text collection, rather than a new Flickr training run |
| Captioning | 14M BLIP reaches NoCaps CIDEr 105.1 and COCO CIDEr 129.7; 129M BLIP reaches 106.3 and 131.4 | Caption quality under the paper’s COCO fine-tuning and decoding recipe |
| VQA and NLVR2 | 14M BLIP reaches VQA test-dev/test-std 77.54/77.62 and NLVR2 dev/test-P 82.67/82.30 | Answer generation and two-image visual reasoning use downstream task-specific heads |
| Zero-shot video transfer | MSRVTT retrieval R@1/R@5/R@10 43.3/65.6/74.7 with median rank 2; MSRVTT-QA 19.2 and MSVD-QA 35.2 | Eight frames are concatenated for retrieval and 16 for QA; the transfer ignores temporal order |

The broad result is real but easy to overstate. BLIP’s 14M model already beats the matched ALBEF retrieval row, while the 129M and ViT-L rows change both data scale and visual capacity. Captioning, retrieval, VQA, and video use different fine-tuning sets, metrics, and inference arrangements, so their scores should not be read as one common measure of “multimodal understanding.” The video result is especially a test of whether still-image features survive frame concatenation, not a demonstration of temporal modeling.

### Decision test and boundary

BLIP is a sensible fit when one model must support pair ranking, image-text matching, and generation, and when a web corpus contains enough caption noise for a learned filter to pay for itself. The source evidence supports the combination of a shared MED, diverse synthetic captions, and decoupled captioner/filter copies. It does not establish that MED is best for a single task, that the filter generalizes unchanged beyond its COCO fine-tuning distribution, or that synthetic captions are faithful without the filter. [BLIP-2](/paper%20shorts/2023/01/30/blip-2-bootstrapping-language-image-pretraining.html) takes a later route to the same interface problem by freezing the large visual and language endpoints and learning a smaller bridge.

## High-Level Takeaways

- MED unifies retrieval, matching, and captioning by changing attention masks while sharing the multimodal transformations that can be shared safely.
- CapFilt improves noisy web supervision by generating candidate captions and filtering both candidates and original alt text with image-text matching.
- Nucleus sampling helps because the filter can remove its extra noise; decoupling the captioner and filter avoids confirmation bias.
- The reported gains cover COCO retrieval, captioning, VQA, and zero-shot video transfer, but each result uses its own task recipe and metric.
- Still-image features transfer to video in BLIP’s simple frame-concatenation test, while temporal structure remains outside the model.
