---
title: Deep Residual Learning for Image Recognition
date: '2015-12-01T05:00:00.000Z'
section: paper-shorts
postSlug: deep-residual-learning-for-image-recognition
legacyPath: /paper shorts/2015/12/01/deep-residual-learning-for-image-recognition.html
tags:
  - Other
field: 'Vision Foundations'
summary: ResNet made very deep CNNs practical by learning residual updates and carrying gradients through identity shortcuts.
---
## 2015 – Deep Residual Learning for Image Recognition

**arXiv:** [1512.03385](https://arxiv.org/abs/1512.03385)

**GitHub:** [KaimingHe/deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks)

**Project PDF:** [CVPR 2016 paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

**Conference:** CVPR 2016 (1st place ILSVRC 2015 classifier)

## Paper Insights

ResNet addresses the degradation problem: deeper plain networks can have higher training error even though, in principle, extra layers could learn identity mappings. The residual block changes the target from learning H(x) directly to learning F(x) = H(x) - x, then adds the shortcut x back. Identity shortcuts add almost no parameters or compute but make very deep optimization tractable. The evidence covers ImageNet and CIFAR, including 50-, 101-, and 152-layer networks that outperform shallower baselines and win major recognition/detection tasks. The limitation is not conceptual but architectural: later work still had to refine normalization, bottlenecks, width, and training recipes. The lasting idea is that skip connections make depth usable.

![Residual block schematic](/assets/images/resnet.png)
_Residual-block diagram from the ResNet paper/project materials._

**Summary:** ResNet made depth easier to optimize by changing what each block has to learn. Instead of learning a direct mapping $H(x)$, a residual block learns $F(x)=H(x)-x$ and adds the input back through an identity shortcut: $H(x)=F(x)+x$. If the best transformation is close to identity, the block can push $F(x)$ toward zero rather than forcing a stack of layers to relearn the input.

Those shortcuts act like gradient highways without adding parameters or inference cost. They let the authors train 152-layer networks that were both deeper and more accurate than plain CNNs. Bottleneck blocks, built from 1x1, 3x3, and 1x1 convolutions, kept the compute manageable: ResNet-152 was far deeper than VGG-19 while using fewer FLOPs. An ensemble of ResNets achieved 3.57% top-5 error on ImageNet and won ILSVRC 2015.

## Decision Lens

ResNet informs the decision to add depth through residual updates rather than ask each block to relearn a complete transformation. The operative unit is a residual block applied across a mini-batch of images, with identity shortcuts carrying both activations and gradients across depth.

The ImageNet-to-detection transfer results show that the optimization benefit survives beyond classification. They do not isolate identity shortcuts from batch normalization, initialization, or the bottleneck block design as cleanly as a modern controlled study could.

At 10× depth, activation memory, normalization behavior, communication, and diminishing returns dominate the original degradation problem. The residual-learning claim would weaken if a plain network with matched depth, normalization, initialization, FLOPs, and training time reached the same accuracy and optimization stability.

**Context:** The residual connection is a tiny architectural change with huge practical reach. It fixed the degradation problem in very deep CNNs, transferred well to detection and segmentation, and later became part of the default design vocabulary for modern deep networks, including Transformers.

**Evals / Benchmarks:**

| Model | Params | FLOPs | ImageNet‑1k Top‑5 (val) | Notes |
| ----- | ------ | ----- | ----------------------- | ----- |
| ResNet‑50 | 26 M | 3.8 GF | 6.7 % | single-crop |
| ResNet‑152 | 60 M | 11.3 GF | 4.49 % | single-crop |
| Ensemble (6 nets) | — | — | 3.57 % | ILSVRC 2015 winner |

Transfer mattered as much as classification. ResNet-101 backbones set state of the art on ImageNet detection, localization, and MS-COCO detection and segmentation in 2015.

**Critiques & limitations:** ResNet is compelling because the intervention is so small: add the identity back, then let depth do useful work. The ablations make the degradation problem and its cure clear. The paper is less helpful on deployment tradeoffs, such as latency and energy, and later work had to clarify block variants such as projection shortcuts, BN/ReLU order, and pre-activation. Training 100-layer models also remains heavy when the dataset or compute budget is small.

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

**Takeaway:** ResNet's core idea is almost comically simple: add the identity back. That shortcut reshaped deep-learning practice, and today almost every high-performance architecture inherits some version of its skip-connection logic.
