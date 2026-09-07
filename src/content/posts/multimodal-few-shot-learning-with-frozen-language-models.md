---
title: 'Multimodal Few-Shot Learning with Frozen Language Models'
date: '2021-06-25T00:00:00.000Z'
section: paper-shorts
postSlug: multimodal-few-shot-learning-with-frozen-language-models
legacyPath: /paper shorts/2021/06/25/multimodal-few-shot-learning-with-frozen-language-models.html
tags: [Vision-Language Models, Multimodal Generation]
field: 'Vision-Language Models'
summary: '2021 – Multimodal Few-Shot Learning with Frozen Language Models'
---

## 2021 – Multimodal Few-Shot Learning with Frozen Language Models

**arXiv:** [2106.13884](https://arxiv.org/abs/2106.13884)

## Summary

> Frozen gives a pretrained autoregressive language model a visual interface without updating its weights. A vision encoder maps each image to two continuous vectors in the language model’s token-embedding space, and caption training teaches the frozen decoder to continue from that prefix. The resulting model can answer some visual questions, retrieve outside knowledge, and bind new visual categories from a few interleaved image-text examples. Its transfer is striking for a proof of concept, while its exact-match scores and five-way binding results show how much information a tiny prefix loses.

## Core Insights

### The image becomes a learned prefix

![Frozen examples of open-ended multimodal generation](/assets/images/multimodal-few-shot-frozen-lm-paper-figure-1.png)
*Fig 1: Frozen conditions open-ended completions on images and text together; the examples also show that several decoding seeds may be needed to avoid repetition or unrelated language-model continuations. | source: [Frozen, Figure 1](https://arxiv.org/abs/2106.13884)*

Frozen starts with a 7B autoregressive Transformer trained on the C4 text corpus. Its weights, including the token embedding and self-attention layers, stay fixed. An NF-ResNet-50 vision encoder produces one pooled image vector. A learned linear map expands that vector to $D\times n$ values and reshapes them into $n$ embeddings of the same width $D$ as a language token. The authors test prefix lengths of one, two, and four and find two works best in their configuration.

Caption training updates the visual pathway only. The frozen language model still computes the next-token likelihood, and gradients flow through its attention operations into the image encoder and projection. The learned prefix therefore has to express an image in a coordinate system the language model already knows how to continue. Fine-tuning the language model is an available control, but it generalizes worse because the roughly three million Conceptual Captions pairs are much smaller than the text-only pretraining corpus.

![Frozen inference interface for VQA, outside-knowledge questions, and few-shot classification](/assets/images/multimodal-few-shot-learning-with-frozen-language-models-source-figure-3.webp)
*Fig 2: The same prefix interface supports an image question, an image plus a knowledge-seeking question, or an ordered support set of images and labels before a new query. | source: [Frozen, Figure 3](https://arxiv.org/abs/2106.13884)*

The interface is more flexible than the training example suggests. During training, the model sees one image followed by its caption. At inference, image prefixes and text embeddings can be interleaved in an arbitrary sequence. A few support examples can therefore teach a new task format or associate a new word with a visual category without gradient updates. This is prefix tuning with a dynamic, image-conditioned prefix rather than a fixed learned prompt.

### Transfer comes from language priors and the visual bridge together

The paper trains on Conceptual Captions with early stopping, usually just after one epoch, using batch size 128, Adam with $\beta_1=0.9$ and $\beta_2=0.95$, and a constant learning rate of $3\times10^{-4}$. Images are padded to square and resized to 224×224. The evaluation deliberately includes blind baselines: they receive blacked-out images but still train a prefix, so any score they achieve measures what the language model can infer from the prompt alone.

| Evaluation | Frozen result | Comparison or protocol |
| --- | ---: | --- |
| VQAv2 | 29.5 zero-shot / 35.7 one-shot / 38.2 four-shot | open-ended generation; four examples close about half the zero-to-full-training gap |
| OKVQA | 5.9 zero-shot / 9.7 one-shot / 12.6 four-shot | Frozen never trains on OKVQA; answers require outside knowledge |
| Open-ended miniImageNet, 2-way | 53.4 with one inner-shot; 58.9 with five | chance is 50%; generated class name and EOS must match |
| Open-ended miniImageNet, 5-way | 20.2 with one inner-shot; 22.3 with three | chance is 20%; five new names remain near chance |
| Fast-VQA | 1.6 zero-shot → 7.9 with five inner-shots | exact-match answers with novel names such as “dax” and “blicket” |

On VQAv2, the frozen language model outperforms a model trained from scratch and a language-model-finetuned control for transfer from captioning. Four in-context examples raise accuracy from 29.5 to 38.2, while a baseline trained directly on VQA reaches 48.4. The gap is the point: multimodal prompting works without the target task’s gradient updates, but it is not a replacement for task-specific supervision. The blind model reaches 26.2 zero-shot and 33.3 four-shot, so the visual prefix contributes, yet the language prior accounts for a substantial fraction of the result.

OKVQA tests whether the language model can use an image to retrieve facts learned from text. Frozen sees no OKVQA training data and rises from 5.9 to 12.6 with four examples. The paper’s qualitative airplane example makes the division of labor visible: the vision encoder identifies the airplane, while the language model supplies “the Wright brothers” from its text-only knowledge. The smaller 400M language model reaches only 4.0 zero-shot and 6.6 four-shot, which supports a scale effect but does not separate parameter count from all other model differences.

### Few-shot binding has a narrow operating range

Open-ended miniImageNet replaces familiar class names with nonsense words and gives the model a support set before a query image. In the two-way setting, one inner-shot raises Frozen to 53.4% and five varied inner-shots to 58.9%, above the 50% chance level. Repeating one exemplar is less useful than showing different exemplars. The real-name control is easier because the language model may already associate the original category words with visual descriptions. In the five-way setting, however, 20.2–22.3% is effectively chance. Binding two new names and selecting between them is a much weaker claim than learning a general visual vocabulary.

Fast-VQA makes the same distinction in a question-answering setting. A support set teaches new names for known visual categories, and the model must use those names in a question about a new image. Frozen rises from 1.6% to 7.9% on the synthetic-name version and from 3.7% to 10.5% when the real category names are used. The blind baseline also improves with more textual support, especially on the real-name version. The gains therefore show multimodal integration, but they are not purely visual: task format and linguistic reminders help too.

### Decision test and boundary

Frozen is the right baseline when the question is whether a large text model’s prompting behavior can be reached through a small learned visual interface. Compare it with a blind prefix, a finetuned decoder, and a richer visual token stream under the same open-ended exact-match protocol. Its two-vector prefix is inexpensive and modular, but it cannot preserve arbitrary spatial detail, five-way binding is near chance, and the paper’s strongest examples use curated seeds. The contribution is the frozen-decoder interface and the evidence that language-only few-shot behavior can transfer across the modality boundary.

## High-Level Takeaways

- Frozen trains only an image-conditioned prefix while the language model remains fixed.
- Caption training transfers language-model prompting, outside knowledge, and some visual concept binding.
- A few examples help on VQA and two-way category binding, but five-way binding remains near chance.
- Blind baselines and exact-match protocols show that language priors and prompt format materially affect the gains.
