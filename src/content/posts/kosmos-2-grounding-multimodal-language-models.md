---
title: 'Kosmos-2: Grounding Multimodal Language Models to the World'
date: '2023-06-26T00:00:00.000Z'
section: paper-shorts
postSlug: kosmos-2-grounding-multimodal-language-models
legacyPath: /paper shorts/2023/06/26/kosmos-2-grounding-multimodal-language-models.html
tags: [Vision-Language Models, Visual Grounding]
field: 'Vision-Language Models'
summary: '2023 – Kosmos-2: Grounding Multimodal Language Models to the World'
---

## 2023 – Kosmos-2: Grounding Multimodal Language Models to the World

**arXiv:** [2306.14824](https://arxiv.org/abs/2306.14824)

## Summary

> KOSMOS-2 adds visual grounding to a causal multimodal language model by serializing a phrase and its box as one Markdown-like hyperlink. A web-scale pipeline turns noun phrases into grounded spans, and a 1,024-token coordinate vocabulary lets the model learn those links with next-token prediction. The same interface accepts a user’s box, emits a box, or grounds generated text. This unifies language and spatial reference, while leaving localization limited by quantization, web-derived labels, and the paper’s 0.5-IoU evaluations.

## Core Insights


### GRIT turns web captions into phrase-box links

![Figure 3: The pipeline for constructing grounded image-text pairs](/assets/images/kosmos-2-grounding-multimodal-language-models-source-figure-3.png)
*Fig 1: The pipeline of constructing web-scale grounded image-text pairs. | source: [KOSMOS-2: Grounding Multimodal Large Language Models to the World, Figure 3](https://arxiv.org/abs/2306.14824)*

KOSMOS-2 starts with the data interface, not a new detector head. GRIT is built from a subset of COYO-700M and LAION-2B image-text pairs. First, spaCy extracts noun chunks from a caption such as “a dog in a field of flowers.” A pretrained grounding model such as GLIP proposes boxes for those chunks; non-maximum suppression removes overlapping proposals, and only matches with confidence above 0.65 are kept. Pairs with no retained box are discarded.

The second step makes the text more useful than a list of isolated nouns. The pipeline traverses sentence-dependency children to expand “a dog” into “a dog in a field of flowers,” then drops expressions contained inside a longer retained expression. The box detected for “a dog” is assigned to the expanded phrase. Figure 1 shows why this matters: the final pair links a visually localized dog to the complete referring expression, while “a field of flowers” and “flowers” are removed as redundant substrings.

The resulting GRIT corpus contains 90,614,680 images, 137,349,210 objects, and 114,978,233 text spans, with an average expression length of 4.7 words. These are automatically constructed links, so the model inherits the detector’s vocabulary and confidence decisions. The scale supplies many grounding examples, but the source paper does not turn every association into a human-verified annotation.

### Location tokens make the box part of the language stream

![Figure 1: Grounding and referring in KOSMOS-2](/assets/images/kosmos-2-grounding-multimodal-language-models-source-figure-1.webp)
*Fig 2: KOSMOS-2 is a multimodal large language model with capabilities of multimodal grounding and referring. It can understand multimodal input, follow instructions, perceive object descriptions such as bounding boxes, and ground language to the visual world. | source: [KOSMOS-2: Grounding Multimodal Large Language Models to the World, Figure 1](https://arxiv.org/abs/2306.14824)*

For an image of width $W$ and height $H$, KOSMOS-2 divides each axis into $P=32$ bins. Each bin is represented by a location token, giving $32\times32=1{,}024$ possible location tokens. A box uses its discretized top-left and bottom-right corners: `<box><loc1><loc2></box>`. A phrase is wrapped as `<p> text span </p>`, so the combined form resembles a Markdown hyperlink: `<p> text span </p><box><loc1><loc2></box>`. Multiple boxes use a delimiter inside the box span.

This makes grounding an ordinary next-token prediction problem. The model receives image embeddings from the KOSMOS-1 vision encoder and resampler, then predicts text and location tokens under the same causal objective. A `<grounding>` token tells it when a response should connect language to the visual world. A user can also provide a box in the input; the model reads its location tokens and generates a description. In Figure 2, the model writes “a campfire” together with its box and links “It” to a second region, so referring and grounding share one sequence interface.

The 1.6B-parameter model is initialized from KOSMOS-1. Its vision encoder has 24 layers with hidden size 1,024; the 24-layer MAGNETO multimodal transformer has hidden size 2,048, 32 attention heads, and an 8,192-unit feed-forward layer. Image resolution is 224×224 with 14×14 patches, and the model uses 64 image embeddings. The new location-token embeddings start randomly, and all parameters are updated. Training mixes 185K text-corpus tokens, 215K original and grounded image-caption tokens, and 19K interleaved-data tokens per 419K-token batch for 60K steps—about 25B tokens—on 256 V100 GPUs, taking about one day. Instruction tuning adds LLaVA-Instruct, Unnatural Instructions, FLANv2, and grounded expression-box examples.

### The same serialization supports two grounding directions

![Figure 4: Input format for phrase grounding and referring expression comprehension](/assets/images/kosmos-2-grounding-multimodal-language-models-source-figure-4.webp)
*Fig 3: Input format of evaluation on (1) phrase grounding and (2) referring expression comprehension. | source: [KOSMOS-2: Grounding Multimodal Large Language Models to the World, Figure 4](https://arxiv.org/abs/2306.14824)*

Phrase grounding gives the model a caption and asks for boxes around marked phrases. The evaluation includes the preceding words as context: “A man in a blue hard hat and `<p>orange safety vest</p>`” is less ambiguous than the phrase alone when several people appear. Referring expression comprehension reverses the direction: the input is only a marked expression such as “A man in a blue hard hat and orange safety vest,” and the model must emit its box. In both tasks, location tokens are converted back to coordinates; a prediction is correct only when IoU exceeds 0.5, and a malformed sequence such as a box with one location token is counted as a negative.

On Flickr30K Entities, zero-shot KOSMOS-2 reaches test R@1/R@5/R@10 of 78.7/80.1/80.1, versus VisualBERT’s 71.3/85.0/86.5 with fine-tuning. The near-equality of KOSMOS-2’s top-1 and top-10 scores is a consequence of generating a small set of valid locations directly rather than producing many detector proposals and reranking them. Referring expression comprehension reaches 52.32 on RefCOCO validation, 50.73 on RefCOCO+ validation, and 60.57 on RefCOCOg validation; the test splits are 57.42/47.26, 42.24, and 61.65 respectively. RefCOCO+ removes spatial relations, while RefCOCOg contains longer expressions and spatial relations. The lower scores on the shorter game-generated RefCOCO and RefCOCO+ language show that grounding depends on the expression distribution, not only on box prediction.

The reverse direction is referring expression generation. Given a box, the model is prompted with `<p> It </p><box><loc1><loc2></box> is` and generates a description. On RefCOCOg, zero-shot KOSMOS-2 scores METEOR/CIDEr 12.2/60.3; two and four few-shot examples raise this to 13.8/62.2 and 14.1/62.3. The model slightly exceeds the fine-tuned SLR baseline’s CIDEr 59.2 without reranking and remains below SLR+Rerank’s 66.2. Grounding therefore gives a useful bidirectional interface, while the result still depends on prompting and the language distribution.

### Adding grounding preserves much of the original model’s behavior

The paper evaluates KOSMOS-2 without instruction tuning on the perception-language tasks used for KOSMOS-1. With the prompt “An image of” and beam size 5, Flickr30K captioning reaches CIDEr 66.7 versus KOSMOS-1’s 65.2. With greedy decoding and the standard VQAv2 prompt, VQA accuracy is 45.6 versus 46.7, a small decrease. The model thus acquires grounding while retaining broadly comparable still-image behavior under the same 224×224 input.

Language-task changes are similarly mixed. Relative to KOSMOS-1, KOSMOS-2 scores 72.0 versus 72.1 on StoryCloze, 49.4 versus 50.0 on HellaSwag, 69.1 versus 69.8 on Winograd, 55.6 versus 54.8 on Winogrande, and 72.9 versus 72.9 on PIQA. BoolQ improves from 56.4 to 62.0 and COPA from 63.0 to 67.0, while CB falls from 44.6 to 30.4. These comparisons suggest that the extra location vocabulary and grounded data do not impose a uniform language penalty, but they also do not improve every task.

### Decision test and boundary

KOSMOS-2 is a good fit when a response must carry both a description and an explicit region reference, or when a user should be able to point to an image region instead of describing it in words. Its design keeps the spatial answer inside the language stream, which makes grounding available to captioning, VQA, dialogue, and few-shot referring. The limits are equally concrete: 32-bin coordinates quantize each axis, the web-scale links come from an automatic detector and parser, and a 0.5-IoU threshold can hide fine localization errors. Evaluate the exact expression style and serialization validity on the target domain, and keep the no-instruction-tuning boundary visible when comparing the perception results.

## High-Level Takeaways

- GRIT creates scale by linking automatically detected boxes to noun phrases and dependency-expanded referring expressions.
- A 1,024-token coordinate vocabulary turns bounding boxes into ordinary next-token targets and lets the same stream consume or emit spatial references.
- Phrase grounding, referring comprehension, and box-to-description generation use one serialization in opposite directions.
- KOSMOS-2 improves zero-shot grounding while keeping KOSMOS-1-like captioning and language performance broadly comparable, with task-specific tradeoffs.
- Quantized coordinates, detector-derived labels, malformed sequences, and the 0.5-IoU rule define the boundary of the reported grounding results.
