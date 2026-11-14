---
title: 'DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining'
date: '2023-05-17T00:00:00.000Z'
section: paper-shorts
postSlug: doremi-optimizing-data-mixtures-speeds-up-language-model-pretraining
legacyPath: /paper shorts/2023/05/17/doremi-optimizing-data-mixtures-speeds-up-language-model-pretraining.html
tags: [Language Models, Data Mixtures, Scaling Laws]
field: 'Multimodal Scaling & Data Mixtures'
summary: '2023 – DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining'
---

## 2023 – DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining

**arXiv:** [2305.10429](https://arxiv.org/abs/2305.10429)

## Summary

> DoReMi uses a small proxy model to choose domain weights for a larger language model without tuning against downstream tasks. A reference model defines per-domain excess loss; Group DRO then upweights domains where the proxy has the most learnable headroom. On The Pile, domain weights learned with a 280M proxy improve an 8B model's average one-shot accuracy by 6.5 points and reach the baseline at 2.6× fewer steps.

## Core Insights

### Optimize a mixture through learnable headroom

The three-step procedure is deliberately asymmetric. First train a small reference model on baseline domain weights. Then train a small proxy with Group DRO over domains, using excess loss

$$
\ell_\theta(x)-\ell_{\mathrm{ref}}(x)
$$

as a measure of learnable headroom. Domains with high positive excess loss receive more weight. Finally, average the weight trajectory and resample the full training data for a larger model. The proxy is not the robust model deployed; it is an inexpensive instrument for deciding what the large model should see.

![Source Figure 1: DoReMi's reference, proxy, and full-model pipeline](/assets/images/doremi-optimizing-data-mixtures-source-figure-1.png)
*Fig 1: The pipeline uses two small 280M models to select domain weights for an 8B model. | source: [DoReMi, Figure 1](https://arxiv.org/abs/2305.10429)*

### Downweighting a domain need not hurt it

On The Pile, DoReMi increases Pile-CC from 0.112 to 0.606 and downweights several low- or high-entropy domains, including ArXiv, PubMed Central, Books3, and StackExchange. The resulting 8B model nevertheless lowers held-out log-perplexity on all 22 domains. The authors' explanation is that very easy domains need few samples, very high-entropy domains are close to a uniform prior, and medium-entropy domains supply useful shared structure.

The downstream result is the stronger operational test: DoReMi improves average exact-match accuracy on TriviaQA, NaturalQuestions, WebQuestions, SQuADv2, and LAMBADA by 6.5 percentage points and reaches the baseline at 75k rather than 200k steps. On GLaM, iterated DoReMi is comparable to domain weights tuned on the downstream tasks themselves. The result supports a universal objective, but the evaluation remains five one-shot tasks and two dataset families.

### The proxy is a model of the target, not an oracle

Two 280M runs cost about 8% of the 8B training FLOPs in the main experiment. The domain weights transfer across this 30× scale gap, but Appendix B shows that proxy size changes the mixture: the 280M proxy concentrates on Pile-CC, while a 1B proxy puts more weight on OpenWebText2. Domain weights also move quickly at the beginning of a run and stabilize after roughly 50k steps, so the best mixture depends on training horizon and proxy design.

The limitations are substantive. “Domain” means provenance, so fine-grained quality differences inside a source are invisible; very fine groups can make DRO overly pessimistic; and the proxy-to-main transfer boundary is not established. [Data mixing laws](/paper%20shorts/2024/03/25/data-mixing-laws-optimizing-data-mixtures-by-predicting-language-modeling-performance.html) later make the target validation distribution explicit instead of optimizing all domains symmetrically.

## High-Level Takeaways

- Use excess loss to select domains with learnable headroom, then train the full model on the learned mixture.
- A domain can be downweighted while its perplexity improves through shared structure and better allocation elsewhere.
- Proxy-to-main transfer is powerful but depends on proxy size, horizon, domain definition, and the target model.
- The compute savings are credible in the reported 280M→8B setup; extrapolating them to another scale needs a transfer check.
- [Data-constrained scaling](/paper%20shorts/2023/05/25/scaling-data-constrained-language-models.html) answers a different bottleneck: repeated tokens, not domain weights.
