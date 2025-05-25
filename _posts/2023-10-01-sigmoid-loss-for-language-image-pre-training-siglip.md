---
layout: content
title: "Sigmoid Loss for Language-Image Pre-Training (SigLIP)"
date: 2023-10-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Computer Vision
---

## 2023 – Sigmoid Loss for Language-Image Pre-Training (SigLIP)

**arXiv:** [2303.15343](https://arxiv.org/abs/2303.15343)

**GitHub:** [google-research/big_vision](https://github.com/google-research/big_vision/tree/main/big_vision/models/proj/siglip)

**Hugging Face ports:** [transformers model doc](https://huggingface.co/docs/transformers/model_doc/siglip)

**Conference:** ICCV 2023 (oral)

**Plain-language abstract**
SigLIP replaces CLIP's softmax contrastive loss with a per-pair sigmoid BCE.
The new objective decouples training from batch size and fits into memory with
fewer devices. A ViT-L/256 trained for two days on four TPUs reaches 84.5 %
ImageNet zero-shot accuracy.

**Novel insights**
- Pairwise sigmoid loss scales from small to huge batches.
- Works well with Locked-image Tuning: freeze the vision encoder and train only
  the text head.
- Shows diminishing returns beyond 32 k images per batch.
- 400 M parameters beat prior open-vocabulary baselines.

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

**Critiques**
- **What I liked:** One-line loss swap yields large gains and is easy to
  reproduce. Scales down to small batches.
- **Limitations:** Trained on the private WebLI dataset; public alternatives
  lag slightly and web bias remains.

**Take-home message**
SigLIP demonstrates that contrastive learning does not require a softmax.
A simple sigmoid-BCE loss achieves better accuracy with drastically less
compute, enabling lighter open-vocabulary models.

