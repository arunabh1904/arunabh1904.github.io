---
title: "Neural Machine Translation by Jointly Learning to Align and Translate"
date: '2014-09-01T00:00:00.000Z'
section: paper-shorts
postSlug: neural-machine-translation-by-jointly-learning-to-align-and-translate
legacyPath: /paper shorts/2014/09/01/neural-machine-translation-by-jointly-learning-to-align-and-translate.html
tags:
  - Attention
  - Sequence Modeling
field: Language Models
summary: "2014 \u2013 Neural Machine Translation by Jointly Learning to Align and Translate"
---

## 2014 – Neural Machine Translation by Jointly Learning to Align and Translate

**Paper:** [Bahdanau, Cho, and Bengio, arXiv:1409.0473v7](https://arxiv.org/abs/1409.0473v7) · **Code:** [GroundHog](https://github.com/lisa-groundhog/GroundHog)

## Summary

> Bahdanau attention lets a recurrent translation decoder retrieve a different weighted view of the source sentence for each output word. On WMT14 English–French, RNNsearch-50 reaches 26.75 BLEU against 17.82 for the fixed-context RNN encoder–decoder, with better behavior on long sentences. It still trails the phrase-based Moses system on the full test set; unknown words remain a substantial limitation.

## Core Insights

A fixed-context encoder–decoder compresses an entire source sentence into one vector before translation begins. RNNsearch instead retains a sequence of annotations. A bidirectional recurrent encoder gives each source position a representation of its surrounding words; a learned alignment network scores those annotations against the decoder's previous state.

The source alignment plot makes the retrieval operation visible.

![Source Figure 3a shows English-to-French soft alignment, including noun-adjective reordering.](/assets/images/bahdanau-source-figure-3a.png)
*Fig 1: Panel (a) of the source alignment figure. Each target word receives a distribution over source positions. The reordering around “European Economic Area” shows why translation needs access beyond a monotonic scan. | source: [Bahdanau et al., Figure 3a](https://arxiv.org/html/1409.0473v7#S4.F3)*

[Open the figure at full size](/assets/images/bahdanau-source-figure-3a.png)

At output step $i$, a small feed-forward network produces scores $e_{ij}=a(s_{i-1},h_j)$. Softmax normalizes them across source positions; the decoder receives the weighted context $c_i=\sum_j\alpha_{ij}h_j$. The alignment is learned through translation likelihood, without separate word-alignment labels. It can spread weight over several words rather than choosing a single hard correspondence.

For example, French word order can require retrieving the noun before returning to its modifiers. The source annotations remain available while the output order changes. This is the durable architectural change: the decoder chooses what to retrieve when it needs it. The later [Transformer](/paper%20shorts/2017/06/01/attention-is-all-you-need.html) changes the sequence-processing machinery, but the distinction between one compressed context and addressable source representations remains useful.

### The benchmark separates the gain from the remaining gap

Training uses selected WMT14 parallel data, reduced from 850 million to 348 million words, with 30,000-word vocabularies on both sides. Development combines news-test-2012 and 2013; evaluation uses 3,003 news-test-2014 sentences. The authors train versions limited to 30- or 50-word sentences, using gated recurrent units, Adadelta, minibatches of 80, and beam-search decoding.

On all test sentences, the 50-word RNNsearch model scores 26.75 BLEU versus 17.82 for its fixed-context comparator. Longer training raises RNNsearch to 28.45, still below Moses at 33.30. The often-cited phrase-based parity applies to the subset without unknown words: the longer-trained model reaches 36.15 versus Moses's 35.63. That subset also disallows unknown-token generation, and Moses uses additional monolingual data.

The length breakdown supports the bottleneck hypothesis, but this is not a perfectly isolated attention toggle: RNNsearch also uses a bidirectional encoder. The comparison establishes the usefulness of the complete retrieval-based architecture under the reported recipe.

Attention also creates work. Every target step scores every source annotation, making alignment cost proportional to the product of source and target lengths. My reading is that the paper trades a fixed representational bottleneck for an explicit retrieval budget; later efficiency work changes that budget without making the original information problem disappear.

## High-Level Takeaways

- A changing context vector lets the decoder retrieve source evidence at each output step.
- Translation supervision trains the soft alignment jointly with the recurrent encoder and decoder.
- The large BLEU gain over the fixed-context baseline does not imply parity with Moses on the complete test set.
- Retaining source positions improves access but introduces a source-by-target scoring cost.
