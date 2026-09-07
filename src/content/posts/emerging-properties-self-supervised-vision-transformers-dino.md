---
title: 'DINO: Emerging Properties in Self-Supervised Vision Transformers'
date: '2021-04-29T00:00:00.000Z'
section: paper-shorts
postSlug: emerging-properties-self-supervised-vision-transformers-dino
legacyPath: /paper shorts/2021/04/29/emerging-properties-self-supervised-vision-transformers-dino.html
tags:
  - Self-Supervised Learning
  - Vision Transformers
field: 'Vision Foundations'
topics:
  - learning
summary: '2021 – DINO: Emerging Properties in Self-Supervised Vision Transformers'
---

## 2021 – DINO: Emerging Properties in Self-Supervised Vision Transformers

**arXiv:** [2104.14294](https://arxiv.org/abs/2104.14294)

**Code:** [facebookresearch/dino](https://github.com/facebookresearch/dino)

**Conference:** ICCV 2021

## Summary

> DINO turns two augmented views of an image into a moving-target classification problem. A student predicts a momentum-updated teacher, while centering and sharpening the teacher output keep the predictions informative instead of collapsed. On ImageNet, ViT-S/8 reaches 78.3% with a frozen k-NN classifier and ViT-B/8 reaches 80.1% with a linear probe. The striking object-aligned attention maps are an emergent property of the learned features, not a segmentation objective.

## Core Insights

### The target is another view of the model

For two views $x$ and $x'$, DINO minimizes cross-entropy between the student distribution $P_s(x)$ and a stopped-gradient teacher distribution $P_t(x')$. The networks have the same backbone and projection head, but the teacher is not updated by backpropagation. Its parameters follow the student with an exponential moving average,

$$
\theta_t \leftarrow \lambda\theta_t+(1-\lambda)\theta_s,
$$

where $\lambda$ follows a cosine schedule from 0.996 toward 1. The teacher is therefore a smoothed ensemble of recent students. In the paper's default multi-crop setup, the teacher sees two $224\times224$ global crops, while the student sees those crops plus several $96\times96$ local crops. Every local prediction must agree with a target formed from a wider view of the same image. That local-to-global pressure is the part that a plain two-view description hides.

The visible source diagram below shows one pair of views for clarity. In the full loss, the two teacher global crops supervise every other student crop. The projection head is a three-layer MLP with hidden dimension 2048, $\ell_2$ normalization, and a weight-normalized output layer. DINO does not add a BYOL-style predictor, and the ViT version stays batch-normalization-free.

![DINO student and momentum-teacher self-distillation](/assets/images/emerging-properties-self-supervised-vision-transformers-dino-source-figure-2.webp)
*Fig 1: One view pair passes through student and teacher networks; the teacher is centered, stop-gradient is applied to its target, and an exponential moving average updates its weights. The paper expands this pair into its multi-crop loss. | source: [DINO, Figure 2](https://arxiv.org/abs/2104.14294)*

The teacher logits are centered by a batch statistic updated with its own exponential moving average, then divided by a low temperature to sharpen the distribution. The student uses temperature 0.1; the teacher temperature is warmed from 0.04 to 0.07 during the first 30 epochs. These are target-generation choices, not cosmetic normalization. The center removes a dominant output dimension, while sharpening prevents the target from becoming uniform.

### Collapse is a balance between two failure modes

The ablation makes the stability story concrete. On ViT-S/16 after 300 epochs, the default DINO recipe reaches 72.8% ImageNet k-NN and 76.1% linear accuracy. Removing the momentum teacher collapses both scores to 0.1%. Keeping momentum but removing multi-crop drops the pair to 67.9% and 72.5%; replacing cross-entropy with MSE drops it to 52.6% and 62.4%. Sinkhorn-Knopp normalization gives 72.2% and 76.0% in the same setting, so it is not the source of the result once the momentum target is present. Adding a predictor also changes little (71.8% and 75.6%).

The reason is visible in the paper's collapse decomposition. Centering alone prevents one prototype from dominating but drives all outputs toward the uniform distribution. Sharpening alone produces the opposite degeneracy. With both, the teacher supplies a nontrivial, slowly changing target. Momentum is still essential for the strongest features: a copied student target fails to converge, a previous-epoch teacher reaches 66.6% k-NN, and the momentum teacher reaches 72.8% in the teacher comparison.

### Smaller patches buy semantic resolution with real throughput cost

At $224\times224$, a ViT-S/16 has 197 tokens including the class token; ViT-S/8 has 785. The parameter count stays near 21M, but the patch grid grows from $14\times14$ to $28\times28$. Doubling each spatial dimension quadruples the token count and makes the pairwise attention matrix roughly sixteen times larger. Table 1 measures 1007 images/s for ViT-S/16 and 180 for ViT-S/8 on a V100 with 128 samples per forward. The smaller patches improve the frozen representation: ViT-S/8 reaches 79.7% linear and 78.3% k-NN accuracy, while ViT-B/8 reaches 80.1% and 77.4%. The base model has 85M parameters, so the small ViT-S/8 is the more interesting efficiency point.

![DINO k-NN accuracy versus throughput for different patch sizes](/assets/images/dino-source-figure-5-throughput.png)
*Fig 2: In this 300-epoch sweep, smaller input patches improve k-NN accuracy without adding model parameters, but the curve moves left because the longer token sequence lowers throughput. The plotted ViT-B/8 and ViT-S/8 points expose this accuracy–systems trade-off. | source: [DINO, Figure 5](https://arxiv.org/abs/2104.14294)*

Multi-crop has a similar budget trade-off. On two eight-GPU machines, two global crops alone reach 72.5% after 300 epochs in 45.9 hours and use 9.3 GB per GPU. Adding ten local crops reaches 76.1% in 72.6 hours and uses 15.4 GB; at 100 epochs, the ten-crop run reaches 74.6% in 24.2 hours while the global-only run reaches 67.8% in 15.3 hours. The authors' useful comparison is that the ten-crop recipe reaches about two points more than the 300-epoch global-only result after only 100 epochs, while using more memory per GPU. Its 24.2-hour training run is shorter than the 45.9-hour global-only baseline. More views eventually saturate: six local crops reach 75.9%, only 0.2 points below ten.

### Attention exposes structure the loss never names

The last-layer class-token attention of DINO ViT-S/8 often isolates an object or one of its parts. The paper turns that observation into a probe by retaining 60% of the attention mass and comparing the resulting masks with PASCAL VOC ground truth. The Jaccard score is 45.9 for ViT-S/16 and 44.7 for ViT-S/8, versus 27.3 and 23.7 for supervised ViTs; random masks score 22.0 and 21.8.

![Attention-derived masks from supervised and DINO ViTs](/assets/images/emerging-properties-self-supervised-vision-transformers-dino-source-figure-4.webp)
*Fig 3: Thresholding a single best class-token attention head at 60% mass produces more object-aligned masks for DINO than for supervised ViTs on these examples; the bottom table reports PASCAL VOC Jaccard scores. | source: [DINO, Figure 4](https://arxiv.org/abs/2104.14294)*

This is a representation probe, not a segmentation system. The attention maps are smooth and were never optimized against masks; choosing a head and a threshold is part of the evaluation. The dense transfer result is also deliberately lightweight: on DAVIS-2017, the authors match features between consecutive frames without training a task head or fine-tuning the backbone. ViT-B/8 reaches 71.4 on mean $\mathcal{J}\&\mathcal{F}$, compared with 62.3 for ViT-B/16. In retrieval, frozen DINO ViT-S/16 features trained on Google Landmarks v2 reach 51.5/24.3 mAP on the Medium/Hard Oxford splits and 75.3/51.6 on the corresponding Paris splits. These results suggest that spatial and instance information survives the image-level pretext task, but they do not establish universal dense-task performance.

The paper also shows transfer after fine-tuning: DINO reaches 81.5% ImageNet top-1 for ViT-S/16 and 82.8% for ViT-B/16, compared with 79.9% and 81.8% for the supervised counterparts under the cited protocol. DINO works with ResNet-50 as well, reaching 75.3% linear and 67.5% k-NN accuracy, so the study does not prove that the objective is exclusive to transformers. ViTs make the learned scene layout unusually easy to inspect through their attention maps.

## High-Level Takeaways

- DINO's core target is a momentum ensemble of the student evaluated on another crop; centering and sharpening make that target informative without labels or negative pairs.
- Multi-crop is a representation choice and a systems choice: local views improve the local-to-global signal, while extra crops increase memory use and computation.
- Smaller patches improve the frozen k-NN representation at nearly unchanged parameter count, but the token sequence makes inference substantially slower.
- Object-aligned attention is measured against PASCAL VOC and DAVIS, yet the maps remain smooth probes that need thresholding and do not replace a trained segmentation head.
- The paper's strongest evidence is ImageNet-centered. DINO works on ResNet-50 too, so the experiments support a powerful objective–ViT combination rather than a claim that transformers alone create the effect.
