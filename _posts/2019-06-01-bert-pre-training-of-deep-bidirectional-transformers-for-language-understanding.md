---
layout: post
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
date: 2019-06-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Natural Language Processing
---

## 2019 – BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

**arXiv:** [1810.04805](https://arxiv.org/abs/1810.04805)

**GitHub:** [google-research/bert](https://github.com/google-research/bert)

**Project page / blog:** [Google AI Blog – "Open Sourcing BERT"](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)

**Conference:** NAACL 2019
![BERT Model](/assets/images/bert.png)


**Summary (abstract in plain English):**
BERT is a Transformer encoder stack with 12 layers (BERT-base) or 24 layers (BERT-large) pre-trained on the BooksCorpus and English Wikipedia (3.3B tokens).
Two self-supervised objectives drive pre-training:
1. **Masked Language Modeling (MLM):** randomly mask or corrupt 15% of tokens and predict the originals, forcing the model to use bidirectional context.
2. **Next Sentence Prediction (NSP):** decide whether a candidate sentence B truly follows sentence A in the corpus, encouraging inter-sentence reasoning.
After this unsupervised phase, a small task-specific layer is added and the entire network fine-tuned, enabling state-of-the-art results on a wide range of NLP benchmarks with only minutes to hours of training.

**Novel insights:**
- True bidirectionality via MLM surpasses left-to-right GPT and shallow concatenations like ELMo.
- A single encoder backbone serves QA, NLI, sentiment and NER tasks by attaching lightweight output heads.
- [CLS] pooling and segment embeddings became standard for classification and sentence-pair modelling.
- BERT sparked an ecosystem—RoBERTa, ALBERT, DistilBERT—that iterates on objectives, data and efficiency.

**Evals / Benchmarks:**

| Benchmark | Metric | Prev SOTA | BERT-large | Δ ↑ |
| --------- | ------ | --------- | ---------- | --- |
| GLUE | Avg. score | 72.8 | 80.5 | +7.7 |
| SQuAD 1.1 | F1 | 88.5 | 93.2 | +4.7 |
| SQuAD 2.0 | F1 | 78.0 | 83.1 | +5.1 |
| MNLI | Acc. | 82.1 | 86.7 | +4.6 |

Training required about four days on 16 Cloud TPU v3 chips; fine-tuning QA tasks took roughly 30 minutes on a single TPU.

**Critiques & limitations:**
- **What I liked:** Elegant, task-agnostic framework that demonstrated scaling benefits early and accelerated community progress through open-sourced weights and code.
- **What I didn’t like:** Pre-training demanded substantial compute; NSP was later deemed unnecessary; a fixed 512-token context and quadratic attention limit long-document reasoning; the model remains static after 2018.

**Take-home message:** BERT introduced a simple yet powerful pre-training recipe that unified NLP tasks under one bidirectional Transformer encoder, inspiring many successful successors.
