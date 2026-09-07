---
title: 'DINOv3'
date: '2025-08-13T00:00:00.000Z'
section: paper-shorts
postSlug: dinov3
legacyPath: /paper shorts/2025/08/13/dinov3.html
tags:
  - Self-Supervised Learning
  - Dense Vision
field: 'Vision Foundations'
topics:
  - learning
summary: '2025 – DINOv3'
---

## 2025 – DINOv3

**arXiv:** [2508.10104](https://arxiv.org/abs/2508.10104)

**Code and models:** [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)

## Summary

> DINOv3 scales self-supervised vision to a 7B-parameter ViT on a curated 1,689-million-image set, but its central result is about what scale breaks. During long DINO/iBOT training, ImageNet classification keeps improving while patch features lose locality and dense prediction degrades. Gram anchoring repairs that failure by matching pairwise patch similarities to an earlier, spatially healthier teacher. The paper reports 88.2→88.0 ImageNet linear accuracy alongside 50.3→55.7 ADE20K mIoU and 0.307→0.281 NYUv2 RMSE in its Gram ablation.

## Core Insights

### Scale improves the global token while eroding patch locality

The training recipe keeps DINO's global objective and iBOT's patch objective, but the two signals do not guarantee the same representation quality at long horizons. On both ViT-g and ViT-7B, linear-probe classification rises monotonically while segmentation starts to fall after roughly 200k iterations. By 600k and later, a patch chosen in one region becomes highly similar to increasingly irrelevant patches. The class token is becoming a better global descriptor while the patch grid is becoming a worse map of local evidence.

![Figure 8: Gram anchoring evolution on dense and global benchmarks](/assets/images/dinov3-gram-anchoring-paper-figure.png)
*Fig 1: This source Figure 8 tracks VOC, ADE20K, and ObjectNet during Gram refinement; dense curves recover quickly when patch relationships are anchored, while global performance changes only mildly. | source: [DINOv3, Figure 8](https://arxiv.org/abs/2508.10104)*

That diagnosis changes what a training dashboard should monitor. ImageNet linear accuracy is a useful global probe, but it cannot tell whether a patch selected on a dog's ear still retrieves nearby fur or instead retrieves unrelated regions with the same image-level semantics. Dense probes and patch-similarity maps are not optional diagnostics once the intended use includes segmentation, depth, or correspondence.

### Gram anchoring preserves relationships rather than coordinates

The authors choose an early iteration of the EMA teacher whose dense features are still strong. For P L2-normalized patch features X_S from the student and X_G from the Gram teacher, the added objective is

\[
\mathcal{L}_{\mathrm{Gram}} = \left\|X_S X_S^\top - X_G X_G^\top\right\|_F^2.
\]

The paper writes the Frobenius loss without an explicit $1/P^2$ average; an implementation can absorb that optional averaging into the Gram-loss weight. The model is therefore free to rotate or otherwise change the feature basis as long as the pairwise patch similarities remain close. This is a better fit for self-supervised training than copying the earlier feature vectors directly: the teacher supplies a geometry of local relationships, not a frozen coordinate system. The paper starts the refinement late, updates the Gram teacher every 10k iterations, and also uses a high-resolution variant whose teacher sees twice the normal input resolution before its feature map is downsampled 2×2 with bicubic interpolation.

The teacher choice has a sharp boundary. In the ablation, a 200k teacher is strong, a 100k teacher is similarly useful, and a 1M teacher is worse because it has already inherited the locality problem.

### The repair trades a little global score for a large dense recovery

| Gram teacher and resolution | ImageNet linear | ADE20K mIoU | NYUv2 RMSE ↓ |
| --- | ---: | ---: | ---: |
| Baseline | 88.2 | 50.3 | 0.307 |
| 200k, ×1 | 88.0 | 53.6 | 0.285 |
| 200k, ×2 | 88.0 | 55.7 | 0.281 |
| 100k, ×2 | 87.9 | 55.7 | 0.284 |
| 1M, ×2 | 88.1 | 54.9 | 0.290 |

The ×2 row gives the clearest intuition: the high-resolution teacher supplies a smoother local geometry, and the student distills that geometry without paying a large classification penalty. The result is measured on the paper's probes, so it does not establish that every dense task benefits equally or that a teacher checkpoint can be selected without validation. DINOv3's later high-resolution adaptation, multi-student distillation, and text alignment extend the release beyond this one regularizer.

![Figure 3: High-resolution DINOv3 patch similarity maps](/assets/images/dinov3-source-figure-3.webp)
*Fig 2: This source Figure 3 visualizes cosine similarity from a red-marked patch to all patches at 4096×4096 input resolution; the map shows the dense feature interface the Gram objective is designed to keep coherent. | source: [DINOv3, Figure 3](https://arxiv.org/abs/2508.10104)*

The scale itself is part of the engineering boundary. The paper starts from an approximately 17-billion-image public-post pool and curates 1,689 million images into LVD-1689M; the flagship has 7B parameters (6.7B in the model table). Replication therefore requires both substantial data curation and a reliable checkpoint-selection strategy. Gram anchoring makes the trade-off explicit: more training is useful only if the objective protects the spatial information downstream tasks need.

## High-Level Takeaways

- DINOv3 turns a hidden scaling failure into a measurable one: global classification can improve while patch locality and dense quality decline.
- Gram anchoring matches the patch-similarity matrix, allowing feature coordinates to change while preserving the geometry that dense probes use.
- In the reported ablation, a 200k high-resolution teacher lifts ADE20K from 50.3 to 55.7 mIoU and lowers NYUv2 RMSE from 0.307 to 0.281, with ImageNet linear accuracy moving from 88.2 to 88.0.
- The method depends on finding an earlier, spatially healthy teacher; anchoring to a 1M teacher is less effective, and the full 7B/data-scale recipe remains expensive to reproduce.
