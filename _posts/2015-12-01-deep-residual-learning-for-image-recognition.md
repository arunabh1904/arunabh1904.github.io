---
layout: content
title: "Deep Residual Learning for Image Recognition"
date: 2015-12-01 00:00:00 -0500
categories: ["Paper Shorts"]
field: Computer Vision
---

## 2015 – Deep Residual Learning for Image Recognition

**arXiv:** [1512.03385](https://arxiv.org/abs/1512.03385)

**GitHub:** [KaimingHe/deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks)

**Project PDF:** [CVPR 2016 paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

**Conference:** CVPR 2016 (1st place ILSVRC 2015 classifier)

**Summary (abstract in plain English):** ResNet reframes a layer’s objective from learning an outright mapping \(H(x)\) to learning a residual \(F(x)=H(x)-x\), then adding back the identity shortcut: \(H(x)=F(x)+x\). These identity “highways” let gradients flow through 152-layer nets without vanishing, so extremely deep models become easier to optimise and more accurate. Despite being eight times deeper than VGG-19, a ResNet-152 is 38 % cheaper in FLOPs thanks to the 1×1–3×3–1×1 bottleneck design. An ensemble of ResNets (50–152 layers) achieved 3.57 % top‑5 error on ImageNet, winning ILSVRC 2015.

**Novel insights:**
- **Residual vs. direct mapping:** if the identity mapping is optimal, a residual block can simply drive \(F(x) \rightarrow 0\), dodging the degradation that afflicts plain deep nets.
- **Cost-free identity shortcuts** give gradient highways without adding parameters or inference cost.
- **Depth with efficiency:** bottleneck blocks cut computation so a 152-layer ResNet needs 11.3 GFLOPs vs. VGG‑19’s 19.6 GFLOPs while being far deeper.

**Evals / Benchmarks:**

| Model | Params | FLOPs | ImageNet‑1k Top‑5 (val) | Notes |
| ----- | ------ | ----- | ----------------------- | ----- |
| ResNet‑50 | 26 M | 3.8 GF | 6.7 % | single-crop |
| ResNet‑152 | 60 M | 11.3 GF | 4.49 % | single-crop |
| Ensemble (6 nets) | — | — | 3.57 % | ILSVRC 2015 winner |

Transfer: ResNet‑101 backbones set SOTA on ImageNet detection, localization and MS‑COCO detection/segmentation in 2015.

**Critiques & limitations:**
- **What I liked:** Elegant, minimal change with giant practical impact; the residual idea is now standard in CV and NLP. Thorough ablations showed depth degradation and its cure convincingly. Demonstrated excellent transfer across many vision tasks.
- **What I didn’t like / open issues:** Block variants (identity vs. projection, BN/ReLU order) needed later clarification (ResNet v2, Pre‑Act). The paper focused on accuracy without real-time latency or energy numbers. Training 100–150‑layer nets is still heavy for small datasets and lighter regimes lack guidance.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)
```

**Take-home message:** ResNet’s single idea—“just add the identity back”—re‑shaped deep-learning practice: today virtually every high-performance network, from CNNs to Transformers, inherits its skip-connection DNA.


