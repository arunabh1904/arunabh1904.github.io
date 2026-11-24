---
title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks'
date: '2020-05-22T09:00:00.000Z'
section: paper-shorts
postSlug: retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks
legacyPath: /paper shorts/2020/05/22/retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks.html
tags:
  - Retrieval
  - Question Answering
field: "Language Models"
summary: "2020 – Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
---

## 2020 – Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**Paper:** [Lewis et al., arXiv:2005.11401v4](https://arxiv.org/abs/2005.11401v4) · **Implementation:** [Hugging Face Transformers RAG](https://huggingface.co/docs/transformers/model_doc/rag)

## Summary

> Lewis et al. combine a dense Wikipedia index with a pretrained sequence-to-sequence model. Each retrieved passage gives the generator a separate view of the question, and the model mixes those predictions. This improves several knowledge-intensive benchmarks and makes some facts adjustable by replacing the index. The experiments do not establish that a returned answer is entailed by its retrieved passages.

## Core Insights

### Where the two variants mix evidence

Dense Passage Retrieval embeds a question and scores candidate passages by inner product. The index contains about 21 million nonoverlapping 100-word passages from December 2018 Wikipedia. For each of the top-ranked passages, BART-large reads the question concatenated with that passage. The resulting generator predictions are combined using retriever probabilities.

![Two rows show the same retrieval scores combined with either complete answer probabilities or separate next-token probabilities.](/assets/images/rag-latent-document-mixtures.svg)
*Fig 1: An original schematic of the two latent-document models. RAG-Sequence mixes after scoring an answer; RAG-Token mixes before choosing each next token. | source: [Lewis et al., Figure 1](https://arxiv.org/html/2005.11401v4#S1.F1)*

[Open the figure at full size](/assets/images/rag-latent-document-mixtures.svg)

Suppose one passage identifies a person and another gives a date. RAG-Sequence evaluates the probability of the whole answer under each passage, then averages those probabilities with the retrieval weights. RAG-Token averages the next-token distributions first, so the effective contributing passage can change over the course of an answer. It still retrieves once for the input, not again for every token. This distinction also changes decoding: RAG-Token can use a single beam search over the mixed token probabilities; RAG-Sequence generates candidates per passage and estimates their scores across passages.

The training loss scores the target answer after marginalizing over documents. It updates BART and the query encoder; the document encoder and index stay fixed to avoid rebuilding the index during training. This fine-tuning does not require document labels, although DPR initialization used retrieval supervision on Natural Questions and TriviaQA.

### Improvements depend on the benchmark and split

In [Table 1](https://arxiv.org/html/2005.11401v4#S4.T2.fig1), RAG-Sequence scored 44.5 exact match on Natural Questions against 41.5 for the DPR reader. RAG-Token reached 45.5 on WebQuestions against 41.1 for DPR. The TriviaQA result depends on the split: RAG-Sequence reached 68.0 on TQA-Wiki against 60.5 for T5-11B+SSM, but scored 56.8 on the standard open QA split against DPR's 57.9. The variants also trade places on generation tasks: RAG-Sequence had better ROUGE-L on open MS-MARCO, while RAG-Token had higher question BLEU-1 on Jeopardy.

The index-swap test used questions about world leaders and Wikipedia snapshots from 2016 and 2018. Accuracy was 70% for 2016 questions with the 2016 index and 68% for 2018 questions with the 2018 index; mismatched snapshots did worse. This is evidence for replacing some task-relevant knowledge without updating the generator weights. It is not a test of general freshness or citation accuracy. The generator sometimes answered correctly even without a retrieved passage containing the answer string, so exact match alone could not tell whether it used the index or its own weights.

## High-Level Takeaways

- In the original RAG model, retrieval scores enter the generator's probability model. Supplying a bundle of retrieved passages in one prompt is a different implementation.
- Sequence-level and token-level mixing share the same retrieved candidates, but differ in how they score an answer and how they decode.
- Measure retrieval recall, answer accuracy, and claim support separately. A right answer does not establish a reliable evidence trail.
