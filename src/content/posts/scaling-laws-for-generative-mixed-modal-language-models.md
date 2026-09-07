---
title: 'Scaling Laws for Generative Mixed-Modal Language Models'
date: '2023-01-10T00:00:00.000Z'
section: paper-shorts
postSlug: scaling-laws-for-generative-mixed-modal-language-models
legacyPath: /paper shorts/2023/01/10/scaling-laws-for-generative-mixed-modal-language-models.html
tags: [Scaling Laws]
field: 'Multimodal Scaling & Data Mixtures'
summary: "2023 – Scaling Laws for Generative Mixed-Modal Language Models"
---

**arXiv:** [2301.03728](https://arxiv.org/abs/2301.03728)  
**Conference:** Technical report

## Summary

> This paper treats text, images, speech, code, and molecules as token sequences and asks how their losses interact inside one decoder-only model. Across more than 250 experiments, seven modalities, 8M–6.7B fitting models, and a 30B validation run, it adds a learned interaction term to Chinchilla-style unimodal scaling. The result is a useful distinction between competition at small scale and synergy after a mixture has enough capacity and data.

## Core Insights

### The interaction term turns “mixed” into a measurable regime

The authors define a modality empirically: with σ=3, one dataset is treated as distinct when its perplexity under a model trained on another is more than three times the primary dataset's mean perplexity. The selected seven modalities are Text, Image, Image-Text, Speech, Speech-Text, Code, and Molecules. Images are compressed to discrete VQGAN tokens with an 8× spatial reduction and an 8,192-entry codebook; speech becomes 50 Hz HuBERT units clustered into 2,000 tokens. A shared BPE vocabulary then lets one causal-masked decoder process arbitrary modality orders.

For a pair of modalities, the proposed law is

$$
L(N,D_i,D_j)=\frac{L(N,D_i)+L(N,D_j)}{2}-C_{i,j}
+\frac{A_{i,j}}{N^{\alpha_{i,j}}}
+\frac{B_{i,j}}{(|D_i|+|D_j|)^{\beta_{i,j}}}.
$$

The first term is what two independent models would achieve after averaging their losses. The interaction constant $C_{i,j}$ captures asymptotic synergy, while the two positive terms capture finite-capacity and finite-data competition. Synergy appears when $C_{i,j}$ is larger than the remaining competition terms. This makes the barrier a prediction from fitted laws, rather than a claim that any mixture is automatically beneficial.

### The Speech|Text run is the decisive extrapolation

The paper fits seven model sizes from 8M to 6.7B on 5B, 10B, and 100B token budgets, using 50/50 token mixtures for each selected pair. It then estimates the compute point where Speech|Text should cross its competition barrier. The predicted optimum is a 28.35B model with 45.12B tokens, so the authors train a nearby 30B model on 50B tokens, alongside 350M and 2.7B comparisons.

The source Figure 5 plots $0.5(L_{\text{Text}}+L_{\text{Speech}})/L_{\text{Speech|Text}}$ against updates: the 350M and 2.7B curves stay above the dashed barrier, while the 30B curve crosses it late. The paper calls that crossing the onset of synergy. There is a direction-of-inequality caveat in the printed source: because perplexity is lower-is-better, the displayed unimodal-over-mixed fraction would mathematically make a mixed model that beats the unimodal average produce a value above one, even though the caption labels below one as barrier crossing. I report the observed crossing and the paper's convention rather than silently reversing either one.

![Source Figure 5: Speech-text competition ratio across model sizes and training updates](/assets/images/scaling-laws-for-generative-mixed-modal-language-models-source-figure-5.png)
*Fig 1: The source plot makes the scale transition visible: 350M and 2.7B remain above the dashed competition barrier, while 30B crosses it late in training. | source: [Scaling Laws for Generative Mixed-Modal Language Models, Figure 5](https://arxiv.org/abs/2301.03728)*

### Competition appears in the training dynamics

The average perplexity can decrease smoothly while one submodality stops improving. In the source Figure 6, speech perplexity in a 2.7B Speech|Text run flattens for roughly 15,000 updates before improving again. Across 6.7B runs, the fraction of non-text loss in a flat regime correlates with the fitted interaction parameter $\alpha_{i,j}$, while the authors find no analogous correlation with $\beta_{i,j}$. Their interpretation is coordinate-ascent-like optimization: shared capacity temporarily serves one modality more effectively, then the balance shifts.

The other correlations are similarly operational. The mixed-modality optimal batch size tracks $\beta_{i,j}$, and the number of gradient-norm spikes tracks $\log(N)/\alpha_{i,j}$, which links stronger finite-model competition to less stable training. These are correlations in the studied runs, not a proof that $\alpha$ or $\beta$ causes the optimization behavior. The training recipe also matters: one epoch, 1M-token batches, bf16, gradient clipping at 1.0, Adam with β1=0.9 and β2=0.98, and restarts when perplexity failed to fall after 500M tokens.

![Speech-token perplexity flattening during a mixed-modal run](/assets/images/scaling-laws-for-generative-mixed-modal-language-models-source-figure-6.webp)
*Fig 2: Speech-token perplexity in the 2.7B Speech|Text run flattens for the shaded region, roughly 15,000 updates, even though the mixed training process continues. | source: [Scaling Laws for Generative Mixed-Modal Language Models, Figure 6](https://arxiv.org/abs/2301.03728)*

## High-Level Takeaways

- Mixed-modal scaling needs an interaction term because independent modality laws do not predict competition or synergy.
- The 30B Speech|Text run is valuable because it tests a predicted barrier beyond the smaller fitting models.
- Submodality plateaus and gradient spikes are diagnostics for a mixture that is capacity-limited or poorly tuned.
- The law is measured on specific tokenizers, datasets, and a causal-masked decoder; transfer requires new proxy fits.
