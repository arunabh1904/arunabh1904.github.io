---
title: EfficientNet — Rethinking Model Scaling for ConvNets
date: '2019-05-28T00:00:00.000Z'
section: paper-shorts
postSlug: efficientnet-rethinking-model-scaling-for-convnets
legacyPath: >-
  /paper
  shorts/2019/05/01/efficientnet-rethinking-model-scaling-for-convnets.html
tags:
  - Other
field: 'Vision Foundations'
summary: "2019 – EfficientNet — Rethinking Model Scaling for ConvNets"
---
## 2019 – EfficientNet — Rethinking Model Scaling for ConvNets

**arXiv:** [1905.11946](https://arxiv.org/abs/1905.11946)

**GitHub:** [tensorflow/tpu](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) (TPU implementation)

**Project page / Google AI Blog:** [EfficientNet: Improving Accuracy and Efficiency Through Scaling](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)

**Conference:** ICML 2019

## Summary

> EfficientNet asks how to scale a ConvNet once a good baseline exists. Instead of independently increasing depth, width, or input resolution, compound scaling uses one coefficient to grow all three dimensions in a balanced way. The authors first search for an efficient baseline, EfficientNet-B0, then scale it to B1-B7 under resource constraints. The empirical case is ImageNet accuracy versus parameters/FLOPs, with transfer results showing that the same family works beyond ImageNet. Higher resolution needs enough depth and width to use the extra pixels, while width or depth alone can waste compute. The caveat is that the baseline architecture and search space still matter; compound scaling cannot rescue a weak base model.

## Core Insights

![Figure 2 from EfficientNet: compound scaling balances network width, depth, and resolution](/assets/images/efficientnet-paper-figure-2-model-scaling.png)
*Figure 2 from the [EfficientNet paper](https://arxiv.org/abs/1905.11946), via ar5iv. source: [EfficientNet paper](https://arxiv.org/abs/1905.11946)*

![Figure 1 from EfficientNet — Rethinking Model Scaling for ConvNets](/assets/images/efficientnet-rethinking-model-scaling-for-convnets-source-figure-1.webp)
*Figure 1 Model Size vs. ImageNet Accuracy. All numbers are for single-crop, single-model. Our EfficientNets significantly outperform other ConvNets. In particular, EfficientNet-B7 achieves new state-of-the-art 84.3% top-1 accuracy but being 8.4x smaller and 6.1x faster than GPipe. EfficientNet-B1 is 7.6x smaller and 5.7x faster than ResNet-152. Details are in Table 2 and 4. source: [EfficientNet — Rethinking Model Scaling for ConvNets](https://arxiv.org/abs/1905.11946)*

![Figure 3 from EfficientNet — Rethinking Model Scaling for ConvNets](/assets/images/efficientnet-rethinking-model-scaling-for-convnets-source-figure-3.webp)
*Figure 3 Scaling Up a Baseline Model with Different Network Width ( ), Depth ( ), and Resolution ( ) Coefficients. Bigger networks with larger width, depth, or resolution tend to achieve higher accuracy, but the accuracy gain quickly saturate after reaching 80%, demonstrating the limitation of single dimension scaling. Baseline network is described in Table 1. source: [EfficientNet — Rethinking Model Scaling for ConvNets](https://arxiv.org/abs/1905.11946)*


### Method and reported result

EfficientNet argues that model scaling should be balanced, not improvised one axis at a time. Standard CNNs often grow by becoming deeper, wider, or by consuming higher-resolution images. Tan and Le show that scaling only one dimension leaves accuracy and efficiency on the table.

Their recipe starts with a mobile-sized architecture found by neural architecture search, EfficientNet-B0. From there, a single compound factor $\phi$ scales depth $\alpha^\phi$, width $\beta^\phi$, and resolution $\gamma^\phi$ together. That rule turns one searched micro-architecture into the B1-B7 family while keeping the accuracy-to-compute tradeoff unusually strong.

### Reported evidence and cost

| Model | Params | FLOPs | ImageNet Top-1 | Notes |
| ----- | ------ | ----- | -------------- | ----- |
| EfficientNet-B0 | 5.3 M | 0.39 B | 77.3 % | Mobile-class baseline |
| EfficientNet-B1 | 7.8 M | 0.70 B | 79.2 % | ≈ ResNet-152 accuracy, 27× cheaper |
| EfficientNet-B4 | 19 M | 4.2 B | 83.0 % | Beats NASNet-A (331×48) with 7× fewer FLOPs |
| EfficientNet-B7 | 66 M | 37 B | 84.3 % | 8.4× smaller & 6.1× faster than GPipe-NASNet |

The reported training recipe uses 600 epochs on ImageNet with AutoAugment and dropout. On TPU-v3 and mobile-oriented hardware, the smaller B0-B3 models run in real time, which is part of why the family became popular outside leaderboard settings.

### Where the evidence stops

The compound scaling formula is easy to reuse, but the clean story depends on a strong searched baseline and a heavy training recipe. The paper is mainly optimized for classification; detection and segmentation need extra tuning. EfficientNet also makes clear that "efficient" can mean fewer FLOPs at inference while still requiring expensive architecture search and long training runs.

EfficientNet's lasting lesson is that how you scale can matter more than how much you scale. Balanced depth, width, and resolution gave CNNs a better accuracy-efficiency frontier.

### MBConv layers in PyTorch

```python
class MBConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand: int = 6):
        super().__init__()
        hid = in_ch * expand
        layers = [
            nn.Conv2d(in_ch, hid, 1, bias=False),
            nn.BatchNorm2d(hid),
            nn.SiLU(),
            nn.Conv2d(hid, hid, 3, stride, 1, groups=hid, bias=False),
            nn.BatchNorm2d(hid),
            nn.SiLU(),
            nn.Conv2d(hid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)
        self.use_res = stride == 1 and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_res:
            out += x
        return out
```

## High-Level Takeaways

- EfficientNet informs how to spend additional CNN compute across depth, width, and input resolution instead of scaling one axis by habit. The training unit remains an image, but compound scaling changes the capacity and spatial detail available to every block under a common FLOP multiplier.
- The measured family shows that balanced scaling gives a better ImageNet accuracy-efficiency frontier around the searched B0 baseline; it does not prove universal coefficients across tasks or hardware. The missing experiment re-optimizes the coefficients for detection, segmentation, and memory-bound accelerators under measured latency rather than FLOPs. At 10× resolution, activation memory and data movement dominate. Compound scaling would fail as a general rule if hardware-aware single-axis or neural-architecture scaling consistently won at matched latency and energy.
- EfficientNet made scaling feel like a design problem rather than a brute-force contest. B1 roughly matched ResNet-152 with 27x fewer FLOPs, while B7 topped ImageNet with a fraction of the parameters used by earlier NAS-heavy models.
