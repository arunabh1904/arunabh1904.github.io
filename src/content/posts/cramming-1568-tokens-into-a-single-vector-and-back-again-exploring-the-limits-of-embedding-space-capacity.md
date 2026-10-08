---
title: 'Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity'
date: '2025-02-18T17:08:45.000Z'
section: paper-shorts
postSlug: cramming-1568-tokens-into-a-single-vector-and-back-again-exploring-the-limits-of-embedding-space-capacity
legacyPath: /paper shorts/2025/02/18/cramming-1568-tokens-into-a-single-vector-and-back-again-exploring-the-limits-of-embedding-space-capacity.html
tags: [Language Models, Context Compression]
field: Language Models
summary: '2025 – Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity'
---

## 2025 – Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity

**Paper:** [arXiv:2502.13063](https://arxiv.org/abs/2502.13063) · [Full text, v3](https://arxiv.org/html/2502.13063v3) · [ACL 2025](https://aclanthology.org/2025.acl-long.948/) · [Official code](https://github.com/yurakuratov/hidden_capacity)

This note covers v3, revised 22 June 2025. The paper first appeared on arXiv on 18 February 2025.

## Summary

> A frozen Llama-3.1-8B can use one optimized, 4,096-dimensional input vector to recover a surprising amount of text: the paper reports a capacity of 1,568 tokens on natural-language passages. That number uses a 99% teacher-forced token-accuracy threshold, averaged over sampled texts. Each passage requires its own optimization, with up to 5,000 gradient steps. The result demonstrates substantial usable input-vector capacity; it does not establish a fast general encoder, exact recovery of arbitrary sequences, or lossless compression of images and spatial features.

## Core Insights

### Optimize the input vector for one known text

The experiment removes the encoder from the usual compression pipeline. Given a target passage, it asks whether a small set of input vectors can make an existing language model predict that passage. The language model stays frozen. Only the memory vectors change, and a new passage starts a new optimization problem.

Let the target tokens be $t_1,\ldots,t_N$ and the memory be $M\in\mathbb{R}^{K\times d}$. The vectors have the same width $d$ as the model's input embeddings. They are prepended to the text, and next-token cross-entropy trains them through the frozen model. Conceptually, the objective is

$$
M^*=\arg\min_M\; -\sum_i \log p_\theta(t_i\mid M,t_{<i}),
\qquad \theta\text{ remains fixed}.
$$

The correct preceding tokens are supplied during this optimization. For example, when the target contains an unusual name, the memory can push probability toward that name at the relevant position. It need not encode everything the language model already predicts from the surrounding words. This shared contribution from the vector and the pretrained decoder becomes central to interpreting capacity.

What receives the gradient? Source Figure 2 makes the distinction visible. The memory vector is trainable, while the language model and ordinary token embeddings remain fixed. Each token prediction can attend to the memory and the preceding text.

![Cramming source Figure 2: a trainable memory vector precedes teacher-forced text tokens in a frozen language model](/assets/images/cramming-1568-source-figure-2.png)
*Fig 1: A separate memory vector is optimized for each passage. The frozen language model predicts target tokens from that vector and the correct preceding tokens during training. | source: [Kuratov et al., Figure 2](https://aclanthology.org/2025.acl-long.948/)*

Figure extracted from the ACL proceedings under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); page text and the original caption are omitted. [Open the full-size figure](/assets/images/cramming-1568-source-figure-2.png).

There is no learned projection from an arbitrary small encoder into this space. The optimized variable already has the decoder's required width. A general encoder would have to learn how to produce useful vectors for unseen inputs in one forward pass; this experiment instead searches directly for a vector for each known target.

The appendix uses random initialization and AdamW with learning rate 0.01, weight decay 0.01, and both beta parameters set to 0.9. Optimization stops at perfect token accuracy or after at most 5,000 steps. The released runner also has a patience-based stopping condition. These are per-passage fitting steps, not training epochs for an encoder shared across a corpus.

### Read 1,568 as a measured threshold, not a universal limit

The main natural-text experiment uses 50 passages at each tested length. The model is scored with teacher forcing: every next-token prediction receives the correct prefix, even if the model's preceding prediction was wrong. The analysis selects the largest tested length whose mean token accuracy reaches 0.99. The length grid includes 1,280, 1,568, and 2,048 tokens, so 1,568 is also a grid point rather than an exact mathematical boundary.

A 99% token score does not establish exact sequence recovery. In free-running generation, an early error changes the prefix for subsequent predictions. Perfect teacher-forced predictions can support exact greedy reconstruction from a matching start condition, but the headline threshold permits errors. The reported statistic therefore needs to remain distinct from a corpus-wide exact-match rate measured through complete generation.

Table 1 shows how much the result depends on the decoder and text source:

| Model | Input width | PG-19 capacity | Recent fanfiction capacity | Random-word capacity |
| --- | --- | --- | --- | --- |
| Pythia-160M | 768 | 80 tokens | 80 tokens | 65 tokens |
| Llama-3.2-1B | 2,048 | 512 tokens | 512 tokens | 316 tokens |
| Llama-3.2-3B | 3,072 | 1,024 tokens | 1,024 tokens | 460 tokens |
| Llama-3.1-8B | 4,096 | 1,568 tokens | 1,568 tokens | 792 tokens |

PG-19 contains books that may have appeared in pretraining; it is part of the Pile used by Pythia. To test beyond that exposure, the authors collect 21 fanfiction works published after October 2024, each longer than 20,000 words. They extract the main text and sample passages beginning at sentence boundaries. Similar capacities on PG-19 and this newer writing weaken a simple explanation based on memorizing the evaluated books. They do not remove the decoder's learned knowledge of language.

The random condition samples words from the top 100,000 entries in the GloVe vocabulary. These are not uniformly random tokenizer IDs. A word may split into predictable subword pieces, which the authors acknowledge can overestimate capacity relative to a direct random-token test. The 792-token result must retain that qualification.

### Measure what the vector adds to the decoder

A familiar phrase may be predictable without any memory vector. Counting every correctly recovered token as newly stored information would credit the vector for knowledge already in the model. The paper therefore introduces token gain: the number of correct predictions with memory minus the number correct without it. For Llama-3.1-8B on PG-19, the reported token gain is $1094.1\pm127.6$, below the 1,568-token capacity figure. These statistics describe different quantities and should not be substituted for one another.

The paper also measures the reduction in total sequence cross-entropy:

$$
\Delta H=H_\theta(t)-H_\theta(t\mid M).
$$

Use the same log base and tokenization for both terms. This measures how much uncertainty the fitted vector removes under a particular decoder. Beyond the region of accurate reconstruction, the plots show approximately constant cross-entropy reduction for a given model. A passage with more surprising words uses that capacity faster, so its recoverable token count can be lower even when its length matches an easier passage.

That pattern is an empirical regularity across the tested sources, not a universal information-theoretic law. The paper explicitly warns against directly comparing information-gain values across different vocabularies. There is also a units detail for reproduction: the paper describes information gain in bits, while its [released analysis](https://github.com/yurakuratov/hidden_capacity/blob/c371c3f411abc0adff31b40ef1e0140804cd0827/notebooks/ablations_analyze_results.ipynb) multiplies PyTorch cross-entropy losses by sequence length without an explicit conversion from natural logs. Divide those implementation values by $\ln 2$ before interpreting them as bits.

The decoder contributes more than embedding width. Among models around one billion parameters, the reported natural-text capacities range from 128 to 512 tokens. Pythia-2.8B reaches only 128, below Pythia-1.4B's 160. Parameter count alone does not explain the ordering. The v3 experiments also find the effect in Mamba; Mamba-1.4B reaches 512 tokens on PG-19, so attention is not required for this form of input-vector control.

### More memory vectors help, but their geometry is not a semantic map

Increasing the memory count gives the optimization more degrees of freedom. Llama-3.2-1B reaches a reported 7,168-token capacity with 16 vectors, versus 512 with one. Pythia-160M reaches 2,016 tokens with 32 vectors, approaching its 2,048-position context limit once memory positions are included. Scaling is roughly linear over the tested range, with deviations for Llama. The paper's suggestion that an entire novel could fit into a small memory set is an extrapolation, not a demonstrated novel-length reconstruction.

Recoverability does not imply a simple geometry for those vectors. Appendix E fits multiple memories for the same 64-token GovReport sequence using Sheared-LLaMA-1.3B. Different initializations produce quite different vectors. Their cosine similarities overlap with similarities between memories of different texts, so similarity is not an established semantic retrieval metric here.

Would averaging two successful memories preserve the text? Source Figure 8 tests the straight line between pairs of fitted vectors for the same sequence. Accuracy drops between the endpoints. The interpolation parameter changes only the memory vector, which isolates a failure of this simple operation on the learned representation.

![Cramming source Figure 8: reconstruction accuracy falls when interpolating between different memory vectors for the same sequence](/assets/images/cramming-1568-source-figure-8.png)
*Fig 2: Both endpoint memories reconstruct the same sequence, but linear interpolation between them introduces errors. These solutions do not form one continuously correct basin along the tested paths. | source: [Kuratov et al., Figure 8](https://aclanthology.org/2025.acl-long.948/)*

Figure extracted from the ACL proceedings under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); page text and the original caption are omitted. [Open the full-size figure](/assets/images/cramming-1568-source-figure-8.png).

This experiment does not show that a trained encoder cannot learn a useful organization. It shows that per-example reconstruction optimization does not automatically supply one. Smooth interpolation, nearest-neighbor retrieval, and compositional use as context require their own evidence.

### Separate embedding count, storage cost, and adapter usefulness

The headline compares 1,568 ordinary input embeddings with one memory embedding. Under Appendix F's storage assumption, a 4,096-dimensional bfloat16 vector occupies 8,192 bytes. The reported raw-text-size divided by vector-size ratio is about 0.8, so the vector is larger than the source text in bytes. This result is compatible with a large reduction in input positions because an ordinary token ID occupies far fewer bytes than its floating-point embedding.

Encoding is also expensive. Each fitting experiment runs on one A100 80GB. The appendix reports costs from roughly a dozen seconds for small models and short passages to 10–20 minutes for larger models and longer passages. Reconstruction still generates the output tokens. A one-position initial representation therefore does not establish a 1,568-fold end-to-end speedup, nor does it show that downstream QA can consume the vector as effectively as the original text.

The distinction is useful for [spatial-grid and object adapters](/blog/2026/10/06/adding-spatial-grids-and-object-tokens-to-vision-language-models.html). Concatenating several small feature vectors and projecting them to the reader's width solves a tensor-shape problem. Cramming shows that an input vector can carry substantial sequence-specific control when optimized through a particular decoder. It does not test a learned grid projector, object boxes, positional encodings, camera changes, or geometric reasoning.

[Perceiver IO](/paper%20shorts/2021/07/30/perceiver-io-a-general-architecture-for-structured-inputs-and-outputs.html) offers a direct architectural contrast: a trained encoder reads inputs into latents, and queries specify what the decoder should recover. Here, an optimization procedure creates each memory and a frozen language model reconstructs text. The shared question is bottleneck capacity; the encoding mechanism and evidence for generalization differ.

My synthesis is to use this experiment as a diagnostic for adapter design. Compare a reusable projector with per-example optimized vectors at the same memory count and width. Then measure held-out reconstruction, object and position queries, and the actual downstream task. A large gap would identify unused capacity accessible to optimization; closing that gap without preserving the required spatial facts would still fail the adapter's job.

## High-Level Takeaways

- One optimized Llama-3.1-8B input vector reaches the paper's 1,568-token capacity threshold. The threshold uses mean teacher-forced token accuracy of 99%, so it must not be reported as guaranteed exact recovery of arbitrary inputs.
- The experiment fits each memory separately while freezing the decoder. It establishes usable representations for tested examples, while leaving the cost and generalization of a reusable encoder unresolved.
- Token gain and cross-entropy reduction separate the memory's contribution from the decoder's prior knowledge. Random words remain an imperfect substitute for random tokenizer IDs.
- A reduction in embedding count can coexist with larger byte storage and expensive encoding. Appendix F reports about 0.8× byte compression, and fitting can take 10–20 minutes per passage.
- My synthesis: use optimized memories to probe an adapter's capacity gap, then test the information its task needs. Text reconstruction and embedding width alone do not establish spatial alignment, semantic usefulness, or lossless multimodal compression.
