---
title: Sequence to Sequence Learning with Neural Networks
date: '2014-09-10T00:00:00.000Z'
section: paper-shorts
postSlug: sequence-to-sequence-learning-with-neural-networks
legacyPath: >-
  /paper
  shorts/2014/09/01/sequence-to-sequence-learning-with-neural-networks.html
tags:
  - Other
field: 'Language Models'
summary: "2014 – Sequence to Sequence Learning with Neural Networks"
---

## 2014 – Sequence to Sequence Learning with Neural Networks

**Paper:** [arXiv:1409.3215](https://arxiv.org/abs/1409.3215) · NeurIPS 2014

## Summary

> An encoder LSTM compresses a source sentence into a fixed state, and a decoder LSTM generates its translation. The surprising engineering result is that reversing the source words makes this much easier to train, even though it does not shorten the average source-to-target dependency. Five reversed LSTMs reach 34.81 BLEU on WMT'14 English–French, exceeding the 33.30 phrase-based baseline; a single reversed model reaches 30.59. The paper establishes a practical neural sequence interface while showing how input order, ensembles, search, and vocabulary limits shape the result.

## Core Insights

### One state connects sequences of different lengths

A word-by-word classifier would need an alignment between the input and output positions. Translation cannot rely on that: a short source phrase can require a longer translation, and the order of its words may change. The encoder–decoder interface removes the need to assign each output to an input timestep in advance.

The encoder reads the whole source sentence and passes its final state to a separate decoder. The decoder predicts a distribution over the next target word, conditioned on that state and the target prefix. It continues until it predicts an end-of-sentence token. Formally,

$$
p(y_{1:T'}\mid x_{1:T})=\prod_{t=1}^{T'}p(y_t\mid v,y_{<t}),
$$

where $v$ is the encoder state. Training maximizes the probability of the correct translation; inference searches among possible continuations. The two LSTMs have separate parameters, so reading and generating need not use identical transitions.

![Original seq2seq Figure 1 showing source encoding followed by target generation](/assets/images/seq2seq-source-figure-1-encoder-decoder.png)
*Fig 1: The encoder finishes reading before the decoder begins generating. Each target prediction depends on the transferred state and earlier target words; the end token determines output length. | source: [Seq2seq, Figure 1](https://arxiv.org/abs/1409.3215)*

The drawing makes the information constraint visible: the decoder has no attention connection back to the individual source positions. The actual model nevertheless has a substantial state. Four layers with 1,000 LSTM cells each use 8,000 real numbers for the sentence representation, counting hidden and cell states. “Fixed vector” does not mean a tiny embedding.

### Reversal creates easy early connections without reducing the average distance

Suppose source words $a,b,c$ correspond roughly to target words $\alpha,\beta,\gamma$. Reading $a,b,c$ before generating $\alpha,\beta,\gamma$ leaves the earliest source word relatively far from the first target decision. Reading $c,b,a$ puts $a$ next to that decision. The network can establish some useful correspondences over short paths before it has learned to carry all information across long ones.

The subtle point is that the average distance between corresponding words is unchanged. Some paths become shorter and others longer. The authors attribute the improvement to reducing the *minimum* time lag and making optimization easier, while acknowledging that they do not have a complete explanation.

Section 3.3 reports perplexity falling from 5.8 to 4.7 and decoded BLEU rising from about 25.9 to 30.6. The final Table 1 comparison gives 26.17 versus 30.59 at beam size 12. This is evidence that the way a sequence presents its dependencies can matter substantially even when the information and model family remain the same.

Only the source is reversed; target sentences stay in their normal order during training and testing. Reversal also improves long-sentence behavior, beyond the early target words it directly brings closer. That observation motivates the authors' interpretation of better memory utilization.

### The headline depends on an ensemble, and reranking is a different experiment

The direct translation and SMT reranking results answer different questions. Direct decoding asks the neural system to produce its own translations. Reranking asks it to choose among 1,000 candidates already produced by an SMT system, using an equal combination of the neural and original scores.

| WMT'14 English–French setting | BLEU |
| --- | ---: |
| Phrase-based SMT baseline | 33.30 |
| Single forward LSTM, beam 12 | 26.17 |
| Single reversed LSTM, beam 12 | 30.59 |
| Five reversed LSTMs, beam 1 | 33.00 |
| Five reversed LSTMs, beam 2 | 34.50 |
| Five reversed LSTMs, beam 12 | 34.81 |
| Five reversed LSTMs reranking SMT's 1,000-best list | 36.5 |

Most of the ensemble's decoding gain from search arrives when the beam grows from one to two; increasing it to 12 adds only 0.31 BLEU. The paper notes that five models with beam two are cheaper than one model with beam 12. Spending compute on model diversity and spending it on a wider search are therefore distinct choices.

The 34.81 result should not be attributed to a single model. Nor should 36.5 be described as pure neural generation. The latter inherits the SMT system's candidate set; its much higher oracle rescoring score, around 45 BLEU, shows that better selection remains possible within those candidates.

### The learned state distinguishes who did what

The source's representation plot is a small but useful test of what survives compression. Swapping John and Mary preserves the word set while changing the relationship. In the left panel, those sentences occupy different groups, with similar relationships among the verbs within each group.

![Seq2seq source Figure 2 showing representations sensitive to word order and relatively stable across paraphrases](/assets/images/seq2seq-source-figure-2-word-order.png)
*Fig 2: Subject–object reversals separate in the left projection. The right groups selected active/passive paraphrases by meaning. These examples probe what the encoder preserves; they are not a comprehensive semantic evaluation. | source: [Seq2seq, Figure 2](https://arxiv.org/abs/1409.3215)*

The right panel provides the complementary observation: some changes in surface form, including active/passive phrasing, leave representations relatively close. Together the panels suggest sensitivity to relational meaning rather than only vocabulary. They are two-dimensional PCA projections of selected examples, so distances in this display cannot establish a general semantic guarantee.

### Vocabulary and execution choices remain part of the method

Training uses 12M parallel sentences, with 160,000 source words and 80,000 target words; out-of-vocabulary items become an unknown token. Names in the long translation examples expose this limitation directly. The source's length analysis reports little degradation over much of the tested range, so the fixed-state interface should not be presented as an observed universal failure on long sentences.

Each model has 384M parameters. Four GPUs hold the recurrent layers and four divide the large output softmax; training takes about ten days. Grouping sentences of similar lengths into batches yields a reported twofold speedup by reducing wasted computation. These details explain why an apparently simple encoder–decoder still required careful execution design.

## High-Level Takeaways

- Separate encoding and decoding support variable lengths without a predefined positional alignment.
- Reversing the source creates short early dependencies while leaving average alignment distance unchanged.
- The 34.81 BLEU headline uses five models; the stronger 36.5 result additionally uses SMT candidates.
- Narrow beam search captures most of the ensemble's search benefit in this experiment.
- The state retains useful relational information, with vocabulary limits and selected-example evaluation still constraining the claim.
