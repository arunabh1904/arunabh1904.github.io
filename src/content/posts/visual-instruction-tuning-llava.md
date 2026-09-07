---
title: Visual Instruction Tuning (LLaVA)
date: '2023-04-17T00:00:00.000Z'
section: paper-shorts
postSlug: visual-instruction-tuning-llava
legacyPath: /paper shorts/2023/04/01/visual-instruction-tuning-llava.html
tags:
  - Other
field: 'Vision-Language Models'
summary: "2023 – Visual Instruction Tuning (LLaVA)"
---

## 2023 – Visual Instruction Tuning (LLaVA)

**arXiv:** [2304.08485](https://arxiv.org/abs/2304.08485) · **Project:** [LLaVA](https://llava-vl.github.io/)

## Summary

> LLaVA turns a pretrained visual encoder and a language model into an image-conditioned assistant with a linear connector and synthetic instruction data. Its strongest idea is how that data is made: text-only GPT-4 receives captions and object boxes, then writes questions and answers as though it had seen the image. The resulting model learns a much more useful conversational interface, while the paper's evaluations expose the distance between fluent visual reasoning and reliable perception.

## Core Insights

### A simple connector leaves room to study the supervision

LLaVA takes CLIP ViT-L/14 grid features and projects each feature into Vicuna's embedding dimension. The projected visual sequence and the language instruction then condition an autoregressive answer. The linear projector changes the representation of each patch; it does not compress the whole image into one vector or run a separate question-answering model. The language model must learn to use the resulting visual tokens when deciding what to say next.

![LLaVA's visual features and instruction tokens entering the language model](/assets/images/visual-instruction-tuning-llava-paper-figure.png)
*Fig 1: Image features pass through the learned projection W, while instruction tokens enter through the language path. Both condition the generated answer. | source: [Paper, Figure 1](https://arxiv.org/abs/2304.08485)*

Follow the two arrows into the model in Figure 1. The image arrives through the vision encoder and projector; the question arrives through ordinary language embeddings. Asking a different question does not change the CLIP features in this architecture. It changes how the language model uses them. This is the useful separation behind the recipe: the visual representation can remain fixed while the model learns whether the user wants a description, an object count, or an explanation of something unusual.

The two training stages teach different parts of that interface. First, 595K filtered CC3M image-caption pairs train only the projection matrix, with both CLIP and Vicuna frozen. Predicting captions gives the projector an initial mapping that the existing language model can use. Second, instruction tuning updates both the projector and Vicuna while keeping CLIP frozen. The paper's “end-to-end” label for this stage therefore does not mean that every component is updated. The loss covers the assistant's answer tokens, including the stopping behavior, rather than teaching the model to predict the user's questions.

### A teacher can write visual instructions without receiving pixels

The instruction-data pipeline gives text-only GPT-4 two descriptions of each COCO image: captions for scene meaning, and labeled bounding boxes for object identity and location. A few manually written examples demonstrate the desired outputs. GPT-4 then generates 158K samples: 58K conversations, 23K detailed descriptions, and 77K complex-reasoning examples. These samples cover roughly 80K unique images, so instruction count and image count are different quantities.

This works because the teacher's job is to transform existing annotations into useful interactions. A caption about a person beside a bicycle can support questions about the scene; object boxes can support some relative-position questions. The teacher adds conversational structure and language reasoning to that evidence. It does not obtain visual details omitted from the annotations. That distinction explains both the efficiency of the method and its risk: a plausible story can enter the training answers even when the original image does not establish it.

The data ablation is more informative than the presence of GPT-4 alone. On LLaVA-Bench (COCO), conversation-only tuning yields a relative score of 73.8; the full mixture reaches 85.1. Detailed descriptions and reasoning examples improve the model's conversational score too, from 76.5 to 83.1. Under this evaluation, richer answer supervision helps beyond reproducing one answer format. However, changing the mixture also changes the available training samples, so these rows do not isolate reasoning format from data quantity.

### The headline score is a judge-relative comparison

LLaVA-Bench (COCO) contains 90 questions over 30 held-out COCO images. A text-only GPT-4 reference answers using ground-truth captions and boxes. GPT-4 then judges that reference response against the candidate model's response, with textual visual information supplied to the judge. The reported 85.1 is the relative score under this procedure. It is neither 85.1% question accuracy nor evidence that LLaVA has reached 85.1% of a pixel-reading GPT-4 system's general capability.

| Evaluation | Reported result | What the comparison establishes |
| --- | --- | --- |
| COCO instruction benchmark, no instruction tuning | 21.5 relative score | Caption alignment alone provides a poor instruction-following interface in this setup. |
| Same benchmark, conversation data only | 73.8 | Conversational supervision produces a large gain. |
| Same benchmark, full instruction mixture | 85.1 | The mixed supervision performs best among these variants. |
| LLaVA-Bench in the wild | 67.3 ± 2.0 overall | Performance is weaker on the separate, more varied 24-image, 60-question set. |
| ScienceQA, specialized LLaVA | 90.92% accuracy | This is task-specific fine-tuning, separate from the general chatbot evaluation. |
| ScienceQA, LLaVA plus GPT-4 judge | 92.53% accuracy | The headline result belongs to a combined system. |

The in-the-wild examples make the perceptual boundary concrete. Identifying a restaurant or a particular yogurt in a crowded refrigerator requires reading small details and connecting them to knowledge. A model can produce a convincing scene description while missing exactly the detail needed for the question. The benchmark is useful for exposing that failure, but its small size and shared GPT-4 generation/judging machinery limit how broadly the scores should be interpreted.

### ScienceQA separates visual assistance from answer adjudication

For ScienceQA, LLaVA is trained on the benchmark's own training split to produce a reason followed by an answer. The reported configuration uses features before CLIP's final layer and twelve training epochs. Its 90.92% accuracy already differs from the generic chat setting; the 92.53% result adds another operation. When LLaVA and text-only GPT-4 disagree, GPT-4 sees the question and both outcomes and decides again.

The paper notes that some questions with an image can still be answered from language knowledge. GPT-4 can therefore correct an answer without itself perceiving the image. Conversely, the visual model can supply information missing from GPT-4's text-only context. This is a useful form of complementary inference, but the combined accuracy cannot be attributed entirely to LLaVA's visual grounding.

The durable result is that a relatively small instruction dataset makes pretrained vision and language components usable through a shared conversational interface. Stronger claims about perceptual accuracy need evidence that isolates the image's contribution. The paper's own examples and evaluation setup give reasons to preserve that distinction, even when the response sounds remarkably capable.

## High-Level Takeaways

- A linear patch projector can support visual conversation when the language model is also instruction-tuned; the CLIP encoder remains frozen in both training stages.
- Captions and boxes let a text-only teacher generate visual instructions, but they also define what visual evidence the teacher can actually access.
- The 85.1 result is a GPT-4-relative judged score on 90 questions, not a general visual-accuracy percentage.
- ScienceQA's 92.53% belongs to LLaVA combined with GPT-4 adjudication; the specialized LLaVA model alone reaches 90.92%.
