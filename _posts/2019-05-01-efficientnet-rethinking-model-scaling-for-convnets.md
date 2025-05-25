---
layout: content
title: "EfficientNet — Rethinking Model Scaling for ConvNets"
date: 2019-05-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Computer Vision
---

## 2019 – EfficientNet — Rethinking Model Scaling for ConvNets

**arXiv:** [1905.11946](https://arxiv.org/abs/1905.11946)

**GitHub:** [tensorflow/tpu](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) (TPU implementation)

**Project page / Google AI Blog:** [EfficientNet: Improving Accuracy and Efficiency Through Scaling](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)

**Conference:** ICML 2019

**Summary (abstract in plain English):**
Standard CNNs tend to grow along a single axis: make them deeper, wider or feed higher-resolution images.
Tan and Le show that such unbalanced scaling leaves accuracy on the table.
They first search for a mobile-sized baseline (B0) then scale depth, width and resolution together using a single factor \(\phi\).
The compound rule keeps FLOPs roughly constant while delivering much better accuracy–efficiency trade-offs.

**Novel insights:**
- Compound scaling jointly tunes depth \(\alpha^\phi\), width \(\beta^\phi\) and resolution \(\gamma^\phi\) for better performance.
- One NAS-derived micro-architecture (B0) can be stretched analytically into a full B1–B7 family.
- B1 matches ResNet-152 with 27× fewer FLOPs, while B7 tops ImageNet using a fraction of prior NAS parameters.

**Evals / Latency benchmarks:**

| Model | Params | FLOPs | ImageNet Top-1 | Notes |
| ----- | ------ | ----- | -------------- | ----- |
| EfficientNet-B0 | 5.3 M | 0.39 B | 77.3 % | Mobile-class baseline |
| EfficientNet-B1 | 7.8 M | 0.70 B | 79.2 % | ≈ ResNet-152 accuracy, 27× cheaper |
| EfficientNet-B4 | 19 M | 4.2 B | 83.0 % | Beats NASNet-A (331×48) with 7× fewer FLOPs |
| EfficientNet-B7 | 66 M | 37 B | 84.3 % | 8.4× smaller & 6.1× faster than GPipe-NASNet |

Training uses 600 epochs on ImageNet with AutoAugment and dropout.
Inference on TPU-v3 shows B0–B3 running in real time on modern mobile GPUs.

**Critiques & limitations:**
- **What I liked:** Simple scaling formula practitioners can reuse.
- **Limitations:** Optimised mainly for classification; detection or segmentation need extra tuning.
  Training B0 requires expensive NAS and heavy data augmentation.

**Take-home message:**
EfficientNet's compound scaling still guides modern network design—how you scale often matters more than how big you scale.

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
