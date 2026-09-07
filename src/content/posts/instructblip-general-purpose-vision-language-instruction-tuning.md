---
title: 'InstructBLIP: General-Purpose Vision-Language Instruction Tuning'
date: '2023-05-11T00:00:00.000Z'
section: paper-shorts
postSlug: instructblip-general-purpose-vision-language-instruction-tuning
legacyPath: /paper shorts/2023/05/11/instructblip-general-purpose-vision-language-instruction-tuning.html
tags: [Vision-Language Models, Instruction Tuning]
field: 'Vision-Language Models'
summary: '2023 – InstructBLIP: General-Purpose Vision-Language Instruction Tuning'
---

## 2023 – InstructBLIP: General-Purpose Vision-Language Instruction Tuning

**arXiv:** [2305.06500](https://arxiv.org/abs/2305.06500)

**Code:** [salesforce/LAVIS](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)

## Summary

> InstructBLIP asks the visual connector to read the task instruction before it compresses an image for a frozen language model. Starting from BLIP-2, it fine-tunes the Q-Former on a diverse instruction mixture while keeping the image encoder and LLM fixed. The authors convert 26 datasets into one natural-language interface, hold out half of them for evaluation, and report stronger zero-shot transfer than BLIP-2 and larger Flamingo models. The gains depend on instruction-aware visual selection and balanced sampling.

## Core Insights

### The instruction reaches the visual bottleneck

![Figure 3: Model architecture of InstructBLIP](/assets/images/instructblip-general-purpose-vision-language-instruction-tuning-source-figure-3.png)
*Fig 1: The instruction joins the queries before the Q-Former compresses frozen image features; a projection turns those task-conditioned vectors into soft prompts for the frozen LLM. | source: [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning, Figure 3](https://arxiv.org/abs/2305.06500)*

On the general zero-shot image-to-text path compared here, BLIP-2 uses a Q-Former to turn image features into a fixed set of task-agnostic visual vectors before the instruction reaches the LLM. Its supervised VQA fine-tuning is a separate recipe that also feeds the question into the Q-Former, so the contrast here describes BLIP-2’s zero-shot bridge rather than every BLIP-2 configuration. InstructBLIP adds instruction tokens to the Q-Former. The learnable queries interact with those tokens through self-attention and with the frozen image encoder through cross-attention. A linear projection then sends the resulting $K$ vectors as soft visual prompts to the frozen language model. The number of vectors and the language-model interface stay fixed; what changes is which visual evidence the bottleneck selects.

This matters when one image supports incompatible requests. A caption instruction can favor global appearance, an OCR request can favor scene text, and a spatial question can require a different subset of regions. If the Q-Former first emits one task-agnostic summary, the frozen LLM must recover all of those distinctions from the same compressed signal. InstructBLIP lets the instruction participate before compression, so the language model receives a task-conditioned visual prompt rather than a generic image summary.

The paper initializes from BLIP-2 and updates only the Q-Former during instruction tuning. The image encoder and LLM remain frozen, so the trainable change is localized to the visual connector; this boundary does not isolate feature selection from the instruction templates, dataset mixture, or sampling recipe. The four frozen LLM variants are FlanT5-XL (3B), FlanT5-XXL (11B), Vicuna-7B, and Vicuna-13B; Vicuna models require the authors to repeat BLIP-2-style pre-training because the released BLIP-2 checkpoints do not include Vicuna. Training runs for at most 60K steps, validates every 3K steps, and completes in about 1.5 days on 16 A100 (40G) GPUs.

### The evaluation tests task and dataset transfer separately

![Figure 2: Tasks and datasets used for vision-language instruction tuning](/assets/images/instructblip-general-purpose-vision-language-instruction-tuning-source-figure-2.webp)
*Fig 2: The training mixture spans 11 task families; yellow marks datasets used for instruction tuning and white marks held-out evaluation datasets. | source: [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning, Figure 2](https://arxiv.org/abs/2305.06500)*

The authors gather 26 public datasets across 11 categories: captioning, captioning with reading comprehension, visual reasoning, image question answering, knowledge-grounded question answering, question answering with reading comprehension, question generation, video QA, visual conversational QA, image classification, and LLaVA-Instruct-150K. Each task is converted into 10–15 natural-language templates. For datasets that normally reward short answers, some templates explicitly ask for a short or brief response; OCR tokens are appended to instructions for scene-text tasks.

The split has two meanings. Thirteen held-in datasets supply training data and held-in evaluation; 13 held-out datasets are never exposed during instruction tuning. Some held-out datasets reuse a task category seen during training, while four categories—visual reasoning, video QA, visual conversational QA, and image classification—are held out at the task level. The authors select splits to avoid evaluation examples leaking across the training cluster, so the result tests both dataset shift and entirely new task forms.

Dataset balance is part of the method. If dataset $d$ has $S_d$ training examples, the default sampling probability is

$$
p_d = \frac{\sqrt{S_d}}{\sum_{i=1}^{D}\sqrt{S_i}}.
$$

The authors then lower the weight of multiple-choice A-OKVQA and raise the weight of open-ended OKVQA. This square-root rule prevents the largest sources from dominating, while the manual correction recognizes that similarly sized datasets can require different amounts of training. At inference, open-ended tasks generate normally; classification and multiple-choice tasks rank the candidate vocabulary by log-likelihood. Video QA uses four uniformly sampled frames, processes each frame through the image encoder and Q-Former separately, and concatenates the resulting visual features before the LLM.

### Instruction tuning beats task labels on unseen tasks

![Figure 4: Instruction tuning versus multitask training](/assets/images/instructblip-general-purpose-vision-language-instruction-tuning-source-figure-4.webp)
*Fig 3: With the same BLIP-2 FlanT5-XL backbone, instruction tuning separates held-in performance from transfer to held-out datasets; the plotted averages use the paper’s stated dataset groups. | source: [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning, Figure 4](https://arxiv.org/abs/2305.06500)*

The chart separates memorizing task formats from generalizing a task interface. On held-in data, plain multitask training with instructions only at evaluation reaches a 92.5 average and training with a dataset identifier reaches 93.7, close to InstructBLIP’s 93.8. On held-out data, however, the same methods reach only 46.3 and 46.8, while InstructBLIP reaches 52.9; BLIP-2 without tuning is 46.1. A dataset name tells the model which training source it saw. A natural-language instruction gives it a reusable description of the requested operation, and the Q-Former can use that description while selecting visual evidence.

The ablation in Table 2 tests the two proposed ingredients directly. For FlanT5-XL, the full model’s held-in average is 94.1. Removing instruction-aware visual features reduces it to 89.8 and lowers GQA from 48.4 to 45.9, ScienceQA with image context from 70.4 to 63.4, IconQA from 50.0 to 45.8, VizWiz from 32.7 to 25.1, and iVQA from 53.1 to 47.5. Removing data balancing gives a held-in average of 92.6 and a less uniform degradation: GQA 46.8, ScienceQA 66.0, IconQA 49.9, VizWiz 31.8, and iVQA 51.1. The larger drops on spatial and temporal tasks are consistent with the instruction telling the Q-Former what evidence matters.

Table 1 reports new zero-shot best results on all 13 held-out datasets for the model variants tested. InstructBLIP FlanT5-XL scores 84.5 on Flickr30K, 48.4 on GQA, 64.8 on VSR, 50.0 on IconQA, 46.6 on TextVQA, 46.6 on Visual Dialog, 56.6 on HatefulMemes, 32.7 on VizWiz, 70.4 on image-context ScienceQA, 43.4 on MSVD-QA, 25.0 on MSRVTT-QA, and 53.1 on iVQA; the table uses CIDEr for NoCaps and Flickr30K, MRR for Visual Dialog, AUC for HatefulMemes, and top-1 accuracy for the remaining tasks. The paper reports a 15.0% average relative improvement over BLIP-2 FlanT5-XL, up to 47.1% relative improvement on MSRVTT-QA over the previous best despite no temporal video training, and a 24.8% average relative improvement over Flamingo-80B on six shared datasets.

### Fine-tuning confirms the initialization effect, not universal visual competence

When the model is fine-tuned on an individual task, the visual encoder remains frozen and the input resolution stays at 224×224. This reduces the trainable parameter count from about 1.2B for approaches that update the visual side to 188M. In Table 3, InstructBLIP FlanT5-XXL reaches 90.7 on ScienceQA with image context and 73.3 on OCR-VQA; its A-OKVQA direct-answer scores are 57.1 validation and 54.8 test, and its multiple-choice scores are 81.0 and 76.7. It sets the reported best on ScienceQA, OCR-VQA, and A-OKVQA, while PaLM-E with 562B parameters remains ahead on OKVQA. FlanT5 variants are stronger on multiple-choice tasks, whereas Vicuna variants are generally stronger on open-ended generation, reflecting the capabilities of the frozen LLMs rather than a change in image encoder.

The paper’s qualitative examples show broad behavior—scene reasoning, knowledge-grounded descriptions, metaphorical interpretation, and multi-turn dialogue—but the broader-impact section makes the boundary explicit. A frozen LLM can still hallucinate or reproduce bias, and instruction-aware extraction cannot recover visual detail the frozen encoder never represented. Long answers can also be less useful than a short answer that directly satisfies the request. The held-out gains therefore support a better interface for routing visual evidence, not a guarantee of correctness on arbitrary images or instructions.

### Decision test and boundary

Choose InstructBLIP when one visual backbone must serve many task descriptions and the failure is that a generic visual summary does not contain the right evidence for the requested operation. Reproduce the split between seen and unseen task categories when evaluating a new domain: held-in scores can look strong even when dataset- or task-level transfer is weak. Keep the fixed 224×224 image path, query budget, frozen endpoints, sampling mixture, and inference protocol visible in the comparison, because each changes what the reported zero-shot result measures. [BLIP-2](/paper%20shorts/2023/01/30/blip-2-bootstrapping-language-image-pretraining.html) supplies the frozen-endpoint bridge that InstructBLIP makes instruction-aware.

## High-Level Takeaways

- InstructBLIP puts the instruction inside the Q-Former, so visual features are selected for the requested task before reaching the frozen LLM.
- The 26-dataset mixture tests both dataset shift and four task categories held out entirely from instruction tuning.
- Square-root sampling and manual task weights keep large datasets from dominating and keep different response types trainable together.
- Instruction tuning matches multitask training on held-in data but reaches a higher held-out average because the interface transfers beyond dataset identifiers.
- The gains come with frozen 224×224 visual input and frozen endpoints, so they improve routing of available evidence rather than remove hallucination or detail limits.
