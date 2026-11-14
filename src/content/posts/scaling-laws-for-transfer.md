---
title: 'Scaling Laws for Transfer'
date: '2021-02-02T00:00:00.000Z'
section: paper-shorts
postSlug: scaling-laws-for-transfer
legacyPath: /paper shorts/2021/02/02/scaling-laws-for-transfer.html
tags: [Language Models, Scaling Laws, Transfer Learning]
field: 'Language Models'
summary: '2021 – Scaling Laws for Transfer'
---

## 2021 – Scaling Laws for Transfer

**arXiv:** [2102.01293](https://arxiv.org/abs/2102.01293)

## Summary

> Hernandez et al. turn the benefit of pretraining into an effective-data quantity: how many target-domain characters does a pretrained model save relative to training from scratch? In text-to-Python transfer, that amount grows with fine-tuning data and model size but with diminishing returns. Transfer is therefore a resource trade: it is most valuable in the low-data regime, and it can become a liability when a small model is asked to absorb too much target data.

## Core Insights

### Measure transfer in target-domain data units

The paper compares three curricula for Python: training from scratch, text pretraining followed by Python fine-tuning, and pretraining on a 50/50 mixture of text and non-Python code followed by Python fine-tuning. Models vary across roughly four orders of magnitude in size and fine-tuning data. The authors define $D_T$ as the additional Python data a same-size scratch model would need to reach the fine-tuned model's loss, then define effective data as $D_E=D_T+D_F$.

In the low-data regime, the fitted law is

$$
D_T=k(D_F)^\alpha N^\beta.
$$

For text-to-Python, the reported coefficients are approximately $k=1.9\times10^4$, $\alpha=0.18$, and $\beta=0.38$. For the text-plus-non-Python-code curriculum, $\alpha$ falls to about $0.096$ while $\beta$ remains about $0.38$. The smaller $\alpha$ is consistent with the authors' interpretation that the pretraining distribution is closer to Python, but that interpretation is a hypothesis about this transfer pair, not a universal distance metric.

![Source Figure 2 showing transfer fits across model sizes and fine-tuning data](/assets/images/scaling-laws-for-transfer-source-figure-2.png)
*Fig 1: Effective transferred data is inferred by mapping fine-tuned loss back to the scratch curve. The image is the source transfer-fit panel. | source: [Scaling Laws for Transfer, Figure 2](https://arxiv.org/abs/2102.01293)*

### The multiplier is large only where target data is scarce

The effective-data multiplier is roughly $kN^\beta/D_F^{1-\alpha}$. It falls as fine-tuning data grows: a pretrained model can replace a large amount of target data early, but the same pretraining contributes less once the target set is already large. Because $\beta\approx2\alpha$ in the text-to-Python fit, a tenfold model increase is comparable to roughly a hundredfold increase in fine-tuning data in the low-data regime.

The scratch curves expose the boundary. From-scratch models hit a data wall; pretraining changes the slope rather than merely shifting every curve by a fixed amount. In the high-data regime, however, fine-tuning can underperform scratch training for small models. Even ten or one hundred times the characteristic data scale did not reliably remove this “ossification” in the tested runs, and the authors did not establish whether more careful tuning or larger models would fix it.

### Pretraining is a front-loaded investment

Ignoring the cost of producing the pretrained checkpoint, fine-tuning usually reaches a target loss with less compute than scratch training. That makes transfer attractive when the same representation can serve many low-data tasks. It also makes the accounting asymmetric: the result says how to amortize a pretraining run, not that pretraining is free.

Appendix controls matter. Injecting only about 0.3% known Python into pretraining raised effective Python data by roughly 4.5× rather than the 2× expected from raw token counts, supporting genuine transfer rather than accidental overlap. The authors fit the low-data regime separately because the medium- and high-data regions are poorly conditioned. Use the law to choose whether to pretrain for a target, not to assume that transfer survives every data regime.

## High-Level Takeaways

- Transfer can be measured as equivalent target-domain data, making pretraining benefits comparable to a concrete resource.
- The benefit is strongest when fine-tuning data is scarce and shrinks as target data grows.
- A closer pretraining distribution changes the fitted transfer exponent, but the exponent is not a universal semantic distance.
- Small models can ossify under high-data fine-tuning; the low-data law should not be extrapolated across that boundary.
- [Kaplan's scaling note](/paper%20shorts/2020/01/23/scaling-laws-for-neural-language-models.html) supplies the base model/data scaling picture; this paper adds a transfer axis to it.
