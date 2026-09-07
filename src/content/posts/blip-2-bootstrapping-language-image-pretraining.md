---
title: 'BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Models'
date: '2023-01-30T00:00:00.000Z'
section: paper-shorts
postSlug: blip-2-bootstrapping-language-image-pretraining
legacyPath: /paper shorts/2023/01/30/blip-2-bootstrapping-language-image-pretraining.html
tags: [Vision-Language Models, Multimodal Generation]
field: 'Vision-Language Models'
summary: '2023 – BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Models'
---

## 2023 – BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Models

**arXiv:** [2301.12597](https://arxiv.org/abs/2301.12597)

## Summary

> BLIP-2 bridges a frozen image encoder and a frozen language model with a trainable Querying Transformer (Q-Former). The Q-Former uses 32 learned queries to compress a variable-length image representation into a fixed visual interface. Its first stage learns image-text representations with a frozen encoder; its second stage projects those query features into a frozen OPT or FlanT5 model for generation. In the paper's overview table, BLIP-2 reaches 65.0 zero-shot VQAv2 accuracy versus Flamingo-80B's 56.3 with 188M versus 10.2B trainable parameters, the source of the 8.7-point and 54× headline comparison. A detailed ViT-g/FlanT5-XXL row reports 65.2 on VQAv2 validation and 65.0 on test-dev with 108M trainable parameters. The bottleneck makes endpoint reuse practical, but frozen encoders and language models retain their blind spots.

## Core Insights

### A fixed query interface carries the visual evidence across a frozen boundary

![Figure 1 from BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](/assets/images/blip-2-bootstrapping-language-image-pretraining-source-figure-1.webp)
*Fig 1: Overview of BLIP-2’s framework: a two-stage Q-Former bridges a frozen image encoder and a frozen large language model. | source: [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models, Figure 1](https://arxiv.org/abs/2301.12597)*

The left half of the figure is a compression problem. The frozen image encoder may emit hundreds of patch features, but the Q-Former always returns the same 32 query vectors. The right half is an interface problem: a fully connected layer maps those vectors into the frozen LLM's embedding space, where they act as soft visual prompts. The LLM never needs to update its text weights to accept an image, and the bridge never has to expose every patch token to the decoder.

The fixed interface is also the main inductive bias. If a visual detail is not selected by the queries, the frozen language model cannot recover it later from the raw image. A stronger encoder can improve the evidence available to the queries, while a stronger LLM can improve how that evidence is verbalized; neither removes the information bottleneck.

### Stage one teaches the queries what language will need

The Q-Former has 188M parameters, is initialized from BERT-base weights, and inserts cross-attention to the frozen image features every other transformer block. Each query has width 768, so the output $Z$ is $32\times768$. For comparison, a ViT-L/14 encoder supplies $257\times1024$ features. The Q-Former therefore has to select and summarize rather than pass the whole image sequence forward.

The first stage jointly optimizes three objectives over image-text pairs:

- **Image-text contrastive learning (ITC)** aligns the image representation with the text representation. A unimodal attention mask prevents the queries and text from seeing one another, so the image side cannot solve the contrastive task by copying text.
- **Image-text matching (ITM)** predicts whether an image-text pair matches. Bidirectional query-text attention and hard negatives force the queries to retain fine-grained compatibility signals.
- **Image-grounded text generation (ITG)** generates text from the image. A multimodal causal mask lets text attend to the queries and its preceding text, while the queries cannot look ahead at the target text. The information needed for generation must therefore pass through the query bottleneck.

The three masks are the important detail. ITC prevents leakage, ITM lets the two modalities interact to judge a pair, and ITG makes the queries carry everything the decoder will need. Representation pretraining is therefore not a warm-up added for convenience; it establishes what information the fixed visual interface is expected to preserve.

![Figure 2: Effect of vision-language representation learning on vision-to-language generative learning](/assets/images/blip-2-bootstrapping-language-image-pretraining-source-figure-5.webp)
*Fig 2: Zero-shot VQAv2 accuracy for ViT-g/OPT-6.7B with and without first-stage representation learning; removing that stage causes the score to collapse during generative training. | source: [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models, Figure 5](https://arxiv.org/abs/2301.12597)*

The plot gives the clearest evidence for the two-stage design. With representation learning, the red curve stays around the mid-50s as generative training proceeds. Without it, the blue curve falls from roughly 29 at 16k iterations to roughly 15 by 48k and never recovers. In other words, the frozen OPT model can learn to emit text from visual prompts, but without an aligned query representation it gradually forgets how to use those prompts. The first stage prevents the second stage from becoming a fragile attempt to discover the modality bridge through language loss alone.

### Stage two turns the query vectors into soft visual prompts

The second stage attaches the pretrained Q-Former to a frozen LLM through a learned fully connected projection. For decoder-only OPT, the projected queries are prepended to the text and the LLM is trained with a language-modeling loss. For encoder-decoder FlanT5, the queries and a text prefix enter the encoder and the suffix is the decoder target. The paper tests both families so that the bridge is not tied to one particular language-model interface.

Pretraining uses the BLIP mixture of 129M images, including COCO, Visual Genome, Conceptual Captions 3M and 12M, SBU, and 115M images from LAION-400M. CapFilt supplies synthetic captions for web images; the authors retain the top two captions per image according to CLIP ViT-L/14 similarity and sample one at each step. The representation stage runs for 250k steps and the generative stage for 80k steps. The largest ViT-g/FlanT5-XXL configuration takes less than six days for the first stage and less than three days for the second on one 16×A100 (40G) machine.

### The headline result is strong, but the parameter denominators need context

| Source table and configuration | Trainable parameters | Reported result |
| --- | ---: | --- |
| Table 1 overview, BLIP-2 versus Flamingo-80B | 188M versus 10.2B | Zero-shot VQAv2: 65.0 versus 56.3; BLIP-2 is ahead by 8.7 points |
| Table 2, ViT-g + FlanT5-XXL | 108M (12.1B total) | VQAv2: 65.2 validation, 65.0 test-dev; OK-VQA 45.9; GQA 44.7 |
| Table 1 overview, BLIP-2 ViT-g | 188M | Flickr30K zero-shot retrieval: text retrieval R@1 97.6, image retrieval R@1 89.7 |

The paper reports 188M for the Q-Former in the architecture section and in the overview comparison, while the detailed model rows report 103–108M trainable parameters for particular configurations. Those counts are presented in different tables and should not be silently combined. The 54× headline uses the overview comparison (10.2B divided by 188M); the detailed ViT-g/FlanT5-XXL row supports the 65.2 validation result and its 108M trainable count. In every case, the comparison is about vision-language pretraining parameters and does not equate total inference cost, data curation, or engineering effort.

Fine-tuning changes the trainable boundary. For VQA, the authors update the Q-Former and image encoder while keeping the LLM frozen, and feed the question into the Q-Former so its cross-attention can focus on relevant image regions. For retrieval, they select 128 candidates using image-text similarity and rerank them with ITM scores. This is why the zero-shot bridge and downstream fine-tuning numbers answer different questions.

### Decision test and boundary

Use BLIP-2 when strong unimodal checkpoints should be reused and the multimodal bridge must have a fixed token budget. Compare against the same frozen encoder and LLM with a simpler projector, remove first-stage objectives as an ablation, and vary the number of queries before attributing gains to scale. The paper's limits are concrete: in-context VQA examples do not improve performance because pretraining samples contain only one image-text pair, and instructed generations can contain incorrect knowledge, stale product information, or a wrong reasoning path. Freezing the endpoints preserves their capabilities and their blind spots; it does not turn a fluent answer into verified visual understanding.

## High-Level Takeaways

- BLIP-2 learns a small Q-Former bridge instead of updating a full vision-language stack; 32 learned queries turn variable image features into a fixed interface.
- ITC, ITM, and ITG give the queries complementary pressure: align globally, check fine-grained matching, and retain information needed for generation.
- Representation learning is causally important in the paper's ablation: without it, zero-shot VQAv2 collapses as generative training continues.
- The 8.7-point and 54× headlines use the overview table's 65.0, 188M, and 10.2B values; the detailed best row reports 65.2 validation with 108M trainable parameters.
- Frozen endpoints make training efficient and modular, while limiting correction of missing, stale, or incorrectly interpreted visual evidence.
