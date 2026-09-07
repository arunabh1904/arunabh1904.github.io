---
title: Language Models are Few-Shot Learners
date: '2020-05-28T00:00:00.000Z'
section: paper-shorts
postSlug: language-models-are-few-shot-learners
legacyPath: /paper shorts/2020/05/01/language-models-are-few-shot-learners.html
tags:
  - Other
field: 'Language Models'
summary: "2020 – Language Models are Few-Shot Learners"
---

## 2020 – Language Models are Few-Shot Learners (GPT-3)

**Paper:** [arXiv:2005.14165](https://arxiv.org/abs/2005.14165) · NeurIPS 2020

## Summary

> GPT-3 tests whether a large next-token model can perform new tasks from examples in its input, with no task-specific parameter updates. Across eight model sizes, larger models generally use those examples more effectively. The most revealing results separate knowing an answer from recognizing the requested format: demonstrations can teach a completion model to answer a question or fill one blank, but do not reliably solve every reasoning task. The 175B model trains on 300B tokens; its broad transfer comes with uneven results, benchmark contamination concerns, and substantial model size.

## Core Insights

### The task changes through the context while the parameters stay fixed

In conventional fine-tuning, task examples change the weights before evaluation. GPT-3's few-shot setting puts examples directly before a new query. The same forward computation reads the demonstrations and continues the text; no optimizer updates the model for that task.

This creates two different kinds of adaptation. Pretraining changes parameters over a large text distribution. In-context adaptation changes the model's current input and activations. Calling both “learning” can obscure that distinction: the examples in one prompt do not permanently become new training data in the weights.

The source evaluation diagram makes the boundary explicit. Zero-shot supplies a task description when appropriate; one-shot adds one demonstration; few-shot adds several. The demonstrations tell the model both what relationship to reproduce and what a valid response should look like.

![GPT-3 source Figure 2.1 comparing zero-shot, one-shot, few-shot, and fine-tuning](/assets/images/gpt3-paper-figure-2-1-eval-strategies.png)
*Fig 1: Demonstrations become context in few-shot evaluation, while fine-tuning uses examples to update parameters. More prompt examples consume input space without creating a separate trained checkpoint. | source: [GPT-3, Figure 2.1](https://arxiv.org/abs/2005.14165)*

A simple translation prompt can end after an unfinished target label:

```text
English: good morning
French: bonjour

English: thank you
French: merci

English: good evening
French:
```

This illustrates the paper's interface without conflating the original 2020 model with a later instruction-tuned API checkpoint. The next-token prediction problem stays the same; the prefix specifies a useful continuation.

### Larger models gain more from the same contextual evidence

The scale experiment includes eight models from 125M to 175B parameters, all trained for 300B tokens with a 2,048-token context. The architecture follows GPT-2 with alternating dense and locally banded sparse attention layers. The comparison therefore increases model capacity without also increasing the token budget for each larger model.

Figure 1.2 is more informative than a single best score. It tests removing inserted symbols from words and plots accuracy against the number of demonstrations. The large model's curve rises much more than the smaller models' curves. A natural-language instruction helps especially when examples are scarce; enough demonstrations let the large model infer much of the task without that instruction.

![GPT-3 source Figure 1.2 showing symbol-removal accuracy against contextual examples at three model sizes](/assets/images/gpt3-source-figure-1-2-context-scaling.png)
*Fig 2: Larger models extract more benefit from additional demonstrations on this symbol-removal task. Instructions help at low example counts; the plotted curves describe one diagnostic, not a universal capacity threshold. | source: [GPT-3, Figure 1.2](https://arxiv.org/abs/2005.14165)*

Read the slopes as sensitivity to usable evidence in the prompt. The figure does not show an inner gradient update or prove a particular meta-learning algorithm inside the Transformer. It shows that scale changes how effectively the pretrained model conditions on examples.

### Examples can resolve an output-format ambiguity

LAMBADA asks for the final word of a passage. An ordinary language model can assign probability to many plausible continuations, including ones that continue the sentence rather than supply exactly the missing word. The evaluation task is narrower than unrestricted text prediction.

The few-shot prompt presents fill-in-the-blank examples, making that restriction visible. GPT-3's accuracy moves from 76.2% zero-shot to 86.4% few-shot. One-shot actually performs worse, at 72.5%; the authors suggest one example may be insufficient to identify the format. Small models also fail to gain as much from the few-shot presentation.

This is a useful limit on the interpretation of benchmark gains. Demonstrations may reveal a capability by clarifying how to express it, rather than supplying all the knowledge needed for the answer. The zero-shot and few-shot settings also use different formats, so the improvement is not a controlled estimate of demonstration count alone.

### Factual recall and task adaptation contribute differently across benchmarks

The question-answering experiments are **closed-book**: GPT-3 receives no retrieved reference passage. Answers depend on information represented in its parameters plus the task examples in context. “Open-domain QA” describes the breadth of questions, not permission for this model to search a knowledge base.

| Reported QA score | Zero-shot | One-shot | Few-shot |
| --- | ---: | ---: | ---: |
| Natural Questions | 14.6 | 23.0 | 29.9 |
| WebQuestions | 14.4 | 25.3 | 41.5 |
| TriviaQA | 64.3 | 68.0 | 71.2 |

The strong TriviaQA starting point and much larger WebQuestions gain suggest different mixtures of stored knowledge and sensitivity to the question distribution or answer format. That is an interpretation of the pattern, not a direct measurement of either component. The paper also reports test-server evaluation for the few-shot TriviaQA result, while other settings generally use available development evaluation.

Breadth does not imply uniform competence. With 32 examples per task, GPT-3 scores 71.8 on SuperGLUE, above the paper's fine-tuned BERT-Large comparison at 69.0 but below its fine-tuned state of the art at 89.0. Word-in-Context remains around chance at 49.4. A prompt can specify the requested operation without making the model reliably execute it.

### The training mixture and contamination audit qualify what scale means

The available training datasets total roughly 499B tokens; the models consume 300B tokens. Those quantities differ because data sources are sampled with deliberately unequal weights. Filtered Common Crawl supplies most examples, while smaller curated sources are revisited: Wikipedia reaches about 3.4 epochs, whereas Common Crawl is seen for less than one. The recipe trades some repetition for higher average data quality.

Benchmark isolation is less clean. A filtering bug leaves some detected overlaps in the training data, and retraining is considered too costly. The authors construct cleaned benchmark subsets, usually using 13-gram overlap, and compare their scores with the full sets. Most differences are small, while PIQA and Winograd receive specific contamination qualifications.

That analysis is informative without being definitive. Shared background passages can trigger overlap even when question–answer pairs are absent; removing trivial examples can also make a cleaned subset harder. The audit cannot turn uncontrolled web pretraining into guaranteed unseen-task evaluation.

## High-Level Takeaways

- In-context learning conditions a frozen model on demonstrations; it does not perform task-specific weight updates.
- Larger models generally use contextual examples more effectively, with task-dependent gains.
- Some improvements come from clarifying output format, as LAMBADA's one-word completion experiment illustrates.
- GPT-3's QA results are closed-book, and strong recall on one benchmark does not establish broad reasoning reliability.
- Training uses 300B sampled tokens from a larger, reweighted corpus; contamination analysis remains part of interpreting the results.
