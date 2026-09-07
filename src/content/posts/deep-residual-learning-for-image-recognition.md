---
title: Deep Residual Learning for Image Recognition
date: '2015-12-10T00:00:00.000Z'
section: paper-shorts
postSlug: deep-residual-learning-for-image-recognition
legacyPath: /paper shorts/2015/12/01/deep-residual-learning-for-image-recognition.html
tags:
  - Other
field: 'Vision Foundations'
summary: "2015 – Deep Residual Learning for Image Recognition"
---
## 2015 – Deep Residual Learning for Image Recognition

**arXiv:** [1512.03385](https://arxiv.org/abs/1512.03385)

**GitHub:** [KaimingHe/deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks)

**Project PDF:** [CVPR 2016 paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

**Conference:** CVPR 2016 (1st place ILSVRC 2015 classifier)

## Summary

> ResNet makes extra depth easier to optimize by asking a block to learn a correction to its input. Its decisive comparison is a 34-layer network with and without parameter-free shortcuts: ImageNet top-1 error falls from 28.54% to 25.03%, and the deeper residual network also fits the training data better. Bottleneck blocks make 50–152 layers affordable. The 1,202-layer CIFAR experiment supplies the limit: residual learning can remove an optimization obstacle without making extra capacity generalize better.

## Core Insights

### The deeper network is failing on the training set

The surprising result is not that a bigger network overfits. The plain 34-layer ImageNet network has higher **training** error than the plain 18-layer network. In principle, the additional layers could copy their inputs and reproduce the shallower solution. In practice, the optimizer does not find that solution within the training budget. The paper calls this the degradation problem.

The left panel below shows the deeper plain network staying above the shallower one. On the right, adding shortcuts reverses that ordering: the 34-layer residual model reaches lower training and validation error. Thin curves show training error; thick curves show center-crop validation error. The matched numerical comparison uses ten-crop testing, so the plotted validation curve and the table are different views of the experiment.

![ImageNet training and validation curves for plain and residual networks](/assets/images/resnet-source-figure-4-training.png)
*Fig 1: Increasing depth hurts the plain network's training fit but helps its residual counterpart; these shortcuts add no trainable parameters in the matched comparison. | source: [Deep Residual Learning, Figure 4](https://arxiv.org/abs/1512.03385)*

| ImageNet model | Plain top-1 error | Residual top-1 error | Evaluation |
| --- | ---: | ---: | --- |
| 18 layers | 27.94% | 27.88% | Validation, ten crops |
| 34 layers | 28.54% | 25.03% | Validation, ten crops |

This is stronger evidence than a deeper leaderboard entry alone. Both sides use batch normalization, the same initialization and training recipe, and the same depth and width within each row. The shortcuts in this experiment use identity mappings and zero-padding where dimensions increase. The authors also report healthy gradient norms in the plain networks, and three times as many iterations did not remove the degradation. “It fixes vanishing gradients” is therefore too narrow an account of the evidence: the intervention changes how an already trainable network represents the solution.

### Learn a correction while preserving an available signal

For a desired mapping $H(x)$, the residual branch learns $F(x)=H(x)-x$. The block adds the input back:

$$
y = F(x;W)+x.
$$

If little needs to change, the residual branch can approach zero. A plain stack must instead learn the identity through its weighted layers. This is a change in parameterization, not a claim that the residual model represents an entirely different family of functions. The paper interprets its smaller residual responses as evidence that learning perturbations around an existing signal is useful.

The original block applies ReLU **after** this addition. Its output is $\operatorname{ReLU}(F(x)+x)$, rather than the later pre-activation formulation. The elementwise sum also requires matching spatial sizes and channel counts. When those change, the paper tests either a subsampled, zero-padded shortcut or a learned $1\times1$ projection. The projection costs parameters; an identity shortcut does not. Even the identity case still performs an addition, so “no inference cost” is an approximation.

The shortcut ablation makes that distinction concrete. ResNet-34 with zero-padding reaches 25.03% top-1 error. Projections only at dimension changes reach 24.52%; projecting every shortcut reaches 24.19%. All three improve substantially over the plain model's 28.54%. Learned projections can help, but they are not what makes the main degradation result disappear.

### The bottleneck spends spatial compute on fewer channels

The figure compares a two-layer residual branch with the three-layer bottleneck used in ResNet-50/101/152. Follow the right branch: $256$ channels become $64$ through a $1\times1$ convolution, the $3\times3$ convolution works at width $64$, and another $1\times1$ restores width $256$. The shortcut carries the original wide representation around that narrower computation.

![Basic and bottleneck residual blocks](/assets/images/resnet-source-figure-5-bottleneck.png)
*Fig 2: The bottleneck narrows channels before the spatial convolution and restores them before addition, allowing more weighted layers at manageable compute. | source: [Deep Residual Learning, Figure 5](https://arxiv.org/abs/1512.03385)*

For this illustrated block, ignoring biases and normalization, the three convolutions use $256\times64 + 9\times64^2 + 64\times256 = 69{,}632$ weights. A wide $256\rightarrow256$ shortcut projection alone would add $65{,}536$. That is why keeping ordinary shortcuts as identities matters especially in bottleneck networks: projecting the wide bypass can cost nearly as much as the entire residual branch.

ResNet-50 uses 3, 4, 6, and 3 bottleneck blocks across its four stages. With three convolutions per block, the initial convolution, and the final classifier, this gives 50 weighted layers. ResNet-152 increases the stage counts to 3, 8, 36, and 3. It uses 11.3 billion multiply-adds in the paper's accounting, versus VGG-19's 19.6 billion; layer count alone is a poor estimate of computational cost.

The ImageNet results also depend on how predictions are collected. ResNet-152 has 5.71% top-5 validation error with ten crops, and 4.49% with the paper's dense, multiscale single-model evaluation. The famous 3.57% is the **test-set result of a six-model ensemble**. Those numbers should not be labeled single-crop measurements or placed in one column without their evaluation protocols.

### A thousand layers separate optimization from generalization

The CIFAR-10 experiment pushes the argument beyond a convenient depth range. Across the left and middle panels, adding layers hurts plain networks but helps residual networks. The right panel asks a different question: once training works, does far more depth still improve test performance?

![CIFAR-10 depth comparison including the 1202-layer residual network](/assets/images/resnet-source-figure-6-cifar.png)
*Fig 3: Residual networks improve with depth through 110 layers, but the 1,202-layer model fits training data while performing worse on the test set. | source: [Deep Residual Learning, Figure 6](https://arxiv.org/abs/1512.03385)*

The 1,202-layer network reaches below 0.1% training error, yet its test error is 7.93%. The 110-layer network's best test error is 6.43%, with 6.61% ± 0.16 across five runs. The authors attribute the deeper model's worse test result to overfitting: it has 19.4M parameters against 1.7M, on a 50,000-image training set. These experiments use basic crop/flip augmentation, weight decay, and no dropout. Residual learning made the large model optimizable; it did not supply the regularization needed to make that capacity worthwhile.

### Better features transfer without explaining every competition gain

Replacing VGG-16 with ResNet-101 in the baseline Faster R-CNN system raises COCO validation AP averaged over IoU thresholds from 21.2 to 27.2. That is 6.0 absolute points, approximately 28% relative. The appendix specifies a stride-16 shared feature map, RoI pooling before the final residual stage, and fixed batch-normalization statistics during detection fine-tuning. The transfer result is a concrete backbone comparison, not a classification score reused as evidence of localization.

The stronger competition result adds further changes: box refinement, global context, multiscale testing, additional training images, and ensembling. The improved single model reaches 34.9 COCO test-dev AP; a three-network ensemble reaches 37.4. Those scores belong to the full detection system. The backbone comparison establishes that the learned representation transfers; it does not attribute all subsequent gains to the identity shortcut.

The existing minimal PyTorch block below captures the original same-shape, post-addition-ReLU case. Both convolutions preserve `[batch, channels, height, width]`; a stage that changes that shape needs a matching shortcut.

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

## High-Level Takeaways

- The central failure is higher training error in a deeper plain network; parameter-free shortcuts reverse it under a matched ImageNet comparison.
- Residual blocks learn changes relative to an available input, while dimension-changing projections remain a separate architectural choice.
- Bottlenecks make depth economical by moving the expensive spatial convolution into a narrower channel space.
- Ten-crop, multiscale single-model, and ensemble results measure different inference arrangements; the 3.57% ImageNet result uses six models.
- The 1,202-layer CIFAR result exposes the boundary: easier optimization does not guarantee that additional capacity improves generalization.
