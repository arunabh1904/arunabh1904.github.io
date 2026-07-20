---
title: Sigmoid Loss for Language-Image Pre-Training (SigLIP)
date: '2023-10-01T04:00:00.000Z'
section: paper-shorts
postSlug: sigmoid-loss-for-language-image-pre-training-siglip
legacyPath: >-
  /paper
  shorts/2023/10/01/sigmoid-loss-for-language-image-pre-training-siglip.html
tags:
  - Other
field: 'Vision-Language Models'
summary: "2023 – Sigmoid Loss for Language-Image Pre-Training (SigLIP)"
---
## 2023 – Sigmoid Loss for Language-Image Pre-Training (SigLIP)

**arXiv:** [2303.15343](https://arxiv.org/abs/2303.15343)

**GitHub:** [google-research/big_vision](https://github.com/google-research/big_vision/tree/main/big_vision/models/proj/siglip)

**Hugging Face ports:** [transformers model doc](https://huggingface.co/docs/transformers/model_doc/siglip)

**Conference:** ICCV 2023 (oral)

## Paper Insights

SigLIP replaces CLIP's softmax contrastive loss with independent pairwise sigmoid losses over image-text pairs. CLIP's softmax needs a global view of pairwise similarities across the batch, which pushes toward large batches and cross-device communication. Sigmoid loss treats each pair as a binary classification problem, making training more memory efficient and easier to scale on fewer accelerators. The evidence compares zero-shot ImageNet and related transfer results, showing strong performance with smaller hardware budgets and practical batch sizes. The caveat is that loss simplicity does not remove the need for high-quality image-text data or careful temperature/bias handling. The paper matters because it made language-image pretraining less tied to enormous synchronized batches.

**Summary:** SigLIP changes one important piece of CLIP: the loss. Instead of a softmax contrastive objective over the batch, it uses per-pair sigmoid binary cross-entropy. That decouples training quality from very large batch sizes and makes strong language-image pre-training possible with fewer devices.

The result is surprisingly practical. A ViT-L/256 trained for two days on four TPUs reaches 84.5% ImageNet zero-shot accuracy. The paper also shows that returns diminish beyond about 32k images per batch, and that the objective works well with Locked-image Tuning, where the vision encoder is frozen and the text side adapts.

**Evals / Benchmarks**

| Model | Params | Train setup | ImageNet zero-shot top-1 | Notes |
| ----- | ------ | ----------- | ----------------------- | ----- |
| SigLIP-B/16 | 86 M | 4 TPUv4, 2 days | 79.7 % | 4 k batch |
| SigLIP-L/256 + LiT | 400 M | 4 TPUv4, 2 days | 84.5 % | 20 k batch |
| CLIP-B/16 | 86 M | 32 TPUv3, 12 days | 76.2 % | 32 k batch |

**Tiny SigLIP pairwise loss (PyTorch)**
```python
def siglip_loss(img_emb, txt_emb, temperature=0.07):
    """img_emb, txt_emb: L2-normalised feature tensors [B, D]"""
    logits = (img_emb @ txt_emb.t()) / temperature
    labels = torch.eye(logits.size(0), device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, labels)
```

**Critiques:** SigLIP is compelling because the intervention is so small: swap the loss, get better scaling behavior. It is easier to reproduce at smaller batch sizes than CLIP-style training. The main caveat is the data. The strongest results use the private WebLI dataset, public alternatives lag slightly, and web-scale bias remains part of the model.

## Decision Lens

SigLIP informs whether image-text pre-training needs a globally normalized softmax over the batch. Its atomic unit is one image-text pair with an independent sigmoid classification target, which removes the requirement that every accelerator participate in one shared denominator.

The paper establishes that simpler pairwise normalization can match or improve contrastive transfer while easing large-batch communication. The missing systems ablation is a wall-clock and accuracy comparison across batch sizes and cluster topologies with identical encoders and data. At 10× scale, negative-pair imbalance and web-data noise may replace all-gather as the bottleneck. The claim would fail if a carefully tuned softmax objective reached the same transfer at equal end-to-end throughput or if sigmoid training degraded calibration on hard negatives.

**Takeaway:** SigLIP shows that language-image contrastive learning does not require a softmax over huge batches. A sigmoid BCE loss can deliver better accuracy with much less training infrastructure.
