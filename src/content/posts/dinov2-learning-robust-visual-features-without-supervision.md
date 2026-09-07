---
title: 'DINOv2: Learning Robust Visual Features without Supervision'
date: '2023-04-14T00:00:00.000Z'
section: paper-shorts
postSlug: dinov2-learning-robust-visual-features-without-supervision
legacyPath: /paper shorts/2023/04/14/dinov2-learning-robust-visual-features-without-supervision.html
tags:
  - Self-Supervised Learning
  - Foundation Models
field: 'Vision Foundations'
topics:
  - learning
summary: '2023 – DINOv2: Learning Robust Visual Features without Supervision'
---

## 2023 – DINOv2: Learning Robust Visual Features without Supervision

**arXiv:** [2304.07193](https://arxiv.org/abs/2304.07193)

**Code and models:** [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

**Journal:** TMLR 2024

## Summary

> DINOv2 is a scale-and-data study built around two complementary self-supervised targets. DINO aligns global class-token outputs across crops; iBOT predicts teacher patch outputs for masked student patches. The authors curate LVD-142M by deduplicating a 1.2-billion-image web pool and retrieving images near diverse curated datasets, then train a 1.1-billion-parameter ViT-g and distill it into smaller encoders. ViT-g reaches 86.5% ImageNet top-1 with a linear probe, while the same frozen features transfer to depth, segmentation, retrieval, and video.

## Core Insights

### The data pipeline changes what “unlabeled” means

DINOv2 does not treat a web crawl as a ready-made training set. It starts with curated sources including ImageNet-22k, the ImageNet-1k training set, Google Landmarks, fine-grained recognition data, and training splits from several dense-task datasets. A public web crawl is filtered for unsafe URLs, NSFW content, and PCA-hash duplicates; identifiable faces are blurred, leaving about 1.2 billion unique images.

The authors then use image similarity to control redundancy and coverage. The appendix describes the raw uncurated source as 1.3B images; a self-deduplication graph keeps one representative from each connected component among 64 nearest neighbors with cosine similarity above 0.6, reducing it to 1.1B images. A second pass removes components too similar to the train or test splits of evaluation benchmarks, using a stricter similarity threshold and leaving 744M images. A self-supervised ViT-H/16 trained on ImageNet-22k embeds the images; Faiss retrieval then finds visually related web images for the curated queries. Large query sets use nearest-neighbor retrieval, typically four neighbors per query, while smaller sets use 100,000 clusters with a one-million-image cap per retrieved dataset. The result is LVD-142M with 142,109,386 images.

![DINOv2 data curation and retrieval pipeline](/assets/images/dinov2-learning-robust-visual-features-without-supervision-source-figure-3.webp)
*Fig 1: Curated and uncurated images are embedded, the uncurated pool is deduplicated, and visually related images are retrieved to augment the curated sources. The figure explains why data selection is part of the representation, even without labels or text. | source: [DINOv2, Figure 3](https://arxiv.org/abs/2304.07193)*

The processing itself is a real systems contribution: the deduplication and retrieval run in under two days on 20 nodes, each equipped with eight V100-32GB GPUs. “Self-supervised” does not mean “distribution-free.” The retrieval queries define which concepts are overrepresented, and Table 15 includes training data from ADE20K, Cityscapes, Pascal VOC, KITTI, NYU Depth V2, and SUN RGB-D, along with designated train or base data from retrieval datasets. Near-duplicate validation and test images are removed, but many reported transfer tasks still have related training data in the pretraining mixture.

### Global invariance and patch detail use different targets

DINOv2 combines an image-level DINO loss with an iBOT-style patch loss. The class token from a student crop passes through a DINO head and is matched by cross-entropy to the teacher class token from another crop. Separately, the student randomly masks input patches and predicts the teacher's output for the corresponding visible teacher patches. The first target encourages global semantic invariance; the second forces patch tokens to retain local information that a classification token could ignore.

The two losses use separate projection heads. Teacher outputs use Sinkhorn-Knopp centering for three iterations, and the student uses softmax normalization. A KoLeo regularizer spreads the normalized class tokens by penalizing small nearest-neighbor distances within a GPU batch. The authors also run a short high-resolution phase at $518\times518$ rather than paying the cost of high-resolution training throughout.

The training loop is engineered for scale. FlashAttention-style kernels reduce attention memory, sequence packing forwards $224$ and $98$-pixel crops in one block-diagonal sequence, and efficient stochastic depth skips dropped residual computations. Fully Sharded Data Parallel shards the student, teacher, and AdamW state across GPUs. Against the iBOT implementation on the same hardware, the authors report about twice the speed with one-third the memory. ViT-g uses width 1536, 24 heads, 40 blocks, SwiGLU feed-forward layers, and about 1.1B parameters; the 64-dimensional head size is chosen to make the attention kernels efficient.

### The ablations separate retrieval, dense prediction, and stability

The ViT-L/14 ablation starts from iBOT at 72.9% ImageNet k-NN and 82.3% linear accuracy. Adding the paper's components reaches 82.0% and 84.5%. KoLeo is the most diagnostic for instance-level structure: it moves Oxford-M retrieval from 55.6 to 63.9 mAP while leaving ADE20K near 47 mIoU. The masked-image objective has the complementary effect: ADE20K rises from 44.2 to 47.1 mIoU, while retrieval changes from 64.3 to 63.9. The objectives are not interchangeable regularizers; one spreads global instances and the other preserves patch-level evidence.

The data comparison is similarly useful. With the same ViT-g/14 and the same number of iterations, LVD-142M reaches 85.8 ImageNet-1k, 73.9 ImageNet-A, 47.7 ADE20K, 64.6 Oxford-M, 82.3 iNaturalist-2018, 86.4 iNaturalist-2021, and 67.6 Places205. A random 142M-image sample from the uncurated pool reaches 83.3, 59.4, 48.5, 54.3, 68.0, 76.4, and 67.2. LVD is not uniformly better on every number—uncurated data is slightly higher on ADE20K in this controlled table—but it is much stronger on robustness, retrieval, and fine-grained domains. Compared with ImageNet-22k, LVD improves every listed task except ImageNet-1k. As model size grows, the gap between LVD and ImageNet-22k widens.

### Distillation turns the largest model into a usable family

Smaller DINOv2 models are distilled from a frozen ViT-g rather than trained from scratch. The distillation loop removes student masking and stochastic depth, applies the iBOT loss to the two global crops, and retains an exponential-moving-average student as the final model. This changes the deployment question: the 1.1B teacher supplies the target quality, while ViT-S/B/L checkpoints carry a smaller frozen feature grid.

![DINOv2 distillation across image and pixel-level task groups](/assets/images/dinov2-source-figure-5-distillation.png)
*Fig 2: The left radar compares individual benchmarks; the right tables average metrics over eight task groups. Distilled ViT-L improves over scratch ViT-L, with lower values preferred on the depth axes and higher values on the accuracy axes. | source: [DINOv2, Figure 5](https://arxiv.org/abs/2304.07193)*

In the table of eight task-group averages, scratch ViT-L/14 scores 84.5 ImageNet, 72.2 segmentation, 1.10 depth RMSE, 90.2 classification, 75.8 fine-grained classification, 71.3 retrieval, 69.5 ImageNet-A/R/Sketch, and 67.3 video. Distillation changes those to 86.3, 73.3, 1.08, 91.2, 77.6, 76.3, 74.5, and 67.5. The distilled model beats scratch training on all 12 underlying benchmarks and sometimes approaches or exceeds the teacher. That is evidence for transferring a representation, not merely compressing logits for one classifier.

### Frozen features are strong, but the probe protocol matters

On ImageNet-1k, the frozen ViT-g/14 reaches 83.5% k-NN and 86.5% linear top-1, versus 82.3% linear for the earlier iBOT ViT-L/16. The result is 0.3 points above OpenCLIP ViT-G/14 and 0.1 above EVA-CLIP ViT-g/14 under the paper's linear evaluation. The DINOv2 model is also more robust on the alternate sets: ImageNet-A 75.9, ImageNet-R 78.8, ImageNet-C mCE 28.2 (lower is better), and ImageNet-Sketch 62.5 for ViT-g/14.

Patch tokens transfer without updating the backbone. With a simple linear segmentation probe, ViT-g/14 reaches 49.0 mIoU on ADE20K, 81.0 on Cityscapes, and 83.0 on Pascal VOC; the boosted multi-scale setup reaches 53.0, 81.0, and 86.2. A frozen-backbone ViT-Adapter plus Mask2Former reaches 60.2 ADE20K mIoU while keeping the backbone frozen. For depth, a DPT decoder on frozen ViT-g features reports RMSE 0.279 on NYU Depth V2, 2.11 on KITTI, and 0.338 when trained on NYU and transferred to SUN RGB-D. These are learned probes on frozen features, not zero-shot predictions.

![DINOv2 patch features establish semantic correspondences across images](/assets/images/dinov2-source-figure-1-pca.png)
*Fig 3: Color-coded PCA components expose recurring parts in the patch features across changes in pose, category, and style. Background is removed before the displayed three-component projection. | source: [DINOv2, Figure 1](https://arxiv.org/abs/2304.07193)*

The paper's qualitative PCA explains why the patch tokens are useful. It thresholds the first component to remove background, then computes three color-coded components across related images. Wings, limbs, and heads align across pose, style, and even category changes. The same representation can match a plane wing with a bird wing, but this is a feature-space correspondence probe, not a guarantee that every domain has the same visual parts.

## High-Level Takeaways

- DINOv2's generality comes from the combination of curated data, global and patch-level targets, scale, and distillation; the loss alone is not the paper's explanation.
- KoLeo mainly improves instance retrieval, while masked patch prediction mainly improves dense prediction. Their ablations support keeping both objectives.
- LVD-142M beats a same-size uncurated sample across most robustness and transfer measures, but its curated mixture includes many downstream training domains and therefore changes the evaluation boundary.
- Distillation lets ViT-S/B/L inherit the 1.1B ViT-g representation; frozen linear probes are strong enough that fine-tuning adds only about two ImageNet points.
- The model still carries distribution and resource costs: training ViT-g is estimated at 22,016 GPU-hours and 3.7 tCO2eq, and Table 12 reports 74.0 accuracy in Africa versus 89.7 in Europe on the geographical fairness probe.
