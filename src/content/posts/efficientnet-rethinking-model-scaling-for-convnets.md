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

> EfficientNet asks how to spend a larger ConvNet budget once the basic architecture is fixed. Its compound rule grows depth, width, and image resolution together; around 1.8 billion FLOPs, the controlled B0 experiment reaches 81.1% ImageNet top-1 accuracy versus roughly 79% when scaling one axis alone. The B0–B7 family combines that rule with a searched mobile architecture and a tuned training recipe. Its efficiency is measured in parameters, operations, and a particular CPU benchmark—not a universal latency guarantee.

## Core Insights

### More pixels need capacity that can use them

A wider network has more feature channels at each location. A deeper network performs more successive transformations. A higher-resolution input gives the network finer spatial evidence. Those changes can complement each other, but one cannot indefinitely compensate for neglecting the others. Extra pixels are less useful if the model lacks capacity to represent their detail; extra channels are less useful if the input has already discarded that detail.

The source diagram makes the three budgets visible. Width changes the horizontal size of a layer, depth adds layers, and resolution changes the spatial input size. Compound scaling changes all three. The authors motivate this balance through receptive field and representational capacity, then test it empirically. The intuition guides the experiment; it is not a theorem prescribing an optimal network for every dataset.

![Depth, width, resolution, and compound scaling of a ConvNet](/assets/images/efficientnet-source-figure-2-scaling.png)
*Fig 1: The five diagrams hold the basic layer pattern fixed while changing width, depth, resolution, or all three, separating architecture choice from how a network grows. | source: [EfficientNet, Figure 2](https://arxiv.org/abs/1905.11946)*

This extends the question raised by [ResNet](/paper%20shorts/2015/12/01/deep-residual-learning-for-image-recognition.html). Residual connections make additional depth easier to optimize. EfficientNet asks whether depth is the best place to spend the next unit of compute once that optimization is possible.

### The exponents follow the cost of a convolution

A regular convolution costs approximately $HWk^2C_{in}C_{out}$ multiply-adds. Scaling both channel dimensions by $w$ multiplies this cost by $w^2$; scaling both spatial dimensions by $r$ multiplies it by $r^2$; repeating more layers by a factor $d$ adds a linear factor. This motivates

$$
\text{compute multiplier}\approx d\,w^2r^2.
$$

EfficientNet parameterizes those choices using one resource coefficient $\phi$:

$$
d=\alpha^\phi,\qquad w=\beta^\phi,\qquad r=\gamma^\phi,
\qquad \alpha\beta^2\gamma^2\approx2.
$$

The constants allocate the budget; $\phi$ determines its scale. Increasing $\phi$ by one approximately doubles operations under this model. It does not double depth, width, and resolution separately, which would cost about $2\times2^2\times2^2=32$ times as much for regular convolutions.

The authors search at $\phi=1$ around the small B0 model and choose $\alpha=1.2$, $\beta=1.1$, and $\gamma=1.15$. Their product under the cost rule is about 1.92, close to the intended factor of two. They then reuse the constants when constructing larger models. Channel and layer rounding, and operators such as depthwise convolutions with different cost dependence, make this an approximate budget rule. Searching around a small model reduces search expense; it does not establish that the same constants remain optimal at every larger scale.

### The strongest evidence keeps B0 fixed

The family’s leaderboard advantage mixes several ingredients, so Figure 8 asks a cleaner question: how do different scaling strategies behave when they start from the **same** EfficientNet-B0 architecture? Follow each line as compute increases. The three single-axis curves improve rapidly and then flatten around 80%; the compound curve continues upward over the plotted range.

![Accuracy versus FLOPs for different ways of scaling EfficientNet-B0](/assets/images/efficientnet-source-figure-8-controlled.png)
*Fig 2: Scaling depth, width, or resolution alone gives diminishing returns from the same B0 baseline; compound scaling reaches higher accuracy at comparable operation counts in this experiment. | source: [EfficientNet, Figure 8](https://arxiv.org/abs/1905.11946)*

Table 7 provides a concrete comparison near 1.8–1.9 billion FLOPs:

| Scaling from B0 | FLOPs | ImageNet top-1 |
| --- | ---: | ---: |
| Depth only, $d=4$ | 1.8B | 79.0% |
| Width only, $w=2$ | 1.8B | 78.9% |
| Resolution only, $r=2$ | 1.9B | 79.1% |
| Compound, $d=1.4,w=1.2,r=1.3$ | 1.8B | 81.1% |

The result supports coordinated scaling within this architecture and recipe. Table 3 also tests the idea beyond EfficientNet: a compound-scaled ResNet-50 reaches 78.8% at 16.7B FLOPs, compared with 78.1% for depth-only scaling at 16.2B. The comparison is close in compute, not exactly equal. MobileNetV1 and V2 show the same direction, giving the rule a broader empirical basis than the searched B0 family alone.

### The baseline was searched for operations, not measured latency

B0 comes from a neural architecture search over a mobile inverted-bottleneck space. Its objective combines accuracy with a FLOP penalty, targeting roughly 400 million operations. The authors explicitly choose FLOPs over latency because they are not targeting one hardware device.

The resulting network uses MBConv blocks with squeeze-and-excitation. Its inverted bottleneck first expands channels, performs a depthwise spatial convolution, then projects back to a narrow output; squeeze-and-excitation reweights channels using image-dependent context. B0 mixes $3\times3$ and $5\times5$ spatial kernels and different stage repetition counts. Compound scaling preserves that basic operator pattern while changing the network’s dimensions.

The original ImageNet Table 2 reports the following family points. These are the table’s rounded values; nearby ablation and latency tables contain slightly different accuracy values and should retain their own context.

| Model | Parameters | FLOPs | Top-1 validation accuracy |
| --- | ---: | ---: | ---: |
| B0 | 5.3M | 0.39B | 77.1% |
| B1 | 7.8M | 0.70B | 79.1% |
| B4 | 19M | 4.2B | 82.9% |
| B7 | 66M | 37B | 84.3% |

B1 uses roughly 16 times fewer FLOPs than the 11B-FLOP ResNet-152 comparison, while exceeding its listed 77.8% accuracy. B7 has about 8.4 times fewer parameters than GPipe’s 557M, with both rounded to 84.3% in this table. These are whole-model comparisons. The paper itself attributes the gains to architecture, scaling, and training settings together.

The training recipe includes RMSProp, SiLU, AutoAugment, stochastic depth, and dropout that increases from 0.2 for B0 to 0.5 for B7. The authors reserve 25,000 training images for early stopping, then report accuracy on the original validation set. Growing the model also changes regularization; the published family is more than a dimension multiplier applied to an otherwise untouched experiment.

### Smaller operation counts need a hardware check

Table 4 measures inference with batch size one on a single Intel Xeon E5-2690 CPU core, averaged over 20 runs. B1 takes 0.098 seconds against ResNet-152’s 0.554 seconds, a 5.7× speedup. B7 takes 3.1 seconds against GPipe’s 19.0 seconds, a 6.1× speedup. The latter is much faster than its comparator but still takes seconds per image. These measurements support efficiency on that CPU setup; they do not establish real-time performance on a phone or GPU.

Transfer uses ImageNet-pretrained checkpoints fine-tuned on eight downstream datasets. For example, B7 reaches 91.7% on CIFAR-100 versus GPipe’s 91.3%, but 98.9% on CIFAR-10 versus 99.0%. The paper reports leading results on five of the eight datasets, including ties in its table. Its useful conclusion is strong transfer with far fewer parameters, while individual tasks still differ.

The minimal block below preserves the earlier note’s channel-expansion, depthwise-convolution, projection, and same-shape residual path. It omits squeeze-and-excitation, stochastic depth, and B0’s variable kernel sizes, so it illustrates the inverted-bottleneck core rather than reproducing an EfficientNet block in full.

```python
import torch
import torch.nn as nn

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

- Compound scaling allocates a compute budget across depth, width, and image resolution; the approximate $dw^2r^2$ cost explains why those dimensions cannot all double freely.
- The same-B0 ablation is the cleanest scaling result: about 81.1% top-1 versus 79% at comparable operations.
- B0’s searched MBConv architecture and the family’s regularization recipe contribute separately to its accuracy–efficiency advantage.
- The 6.1× latency claim is a batch-one, single-core CPU comparison; B7 still takes 3.1 seconds per image there.
- Reusing coefficients found around B0 makes larger-model design cheaper, while optimal scaling remains dependent on the architecture, task, and resource being measured.
