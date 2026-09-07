---
title: 'An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale'
date: '2020-10-22T00:00:00.000Z'
section: paper-shorts
postSlug: an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale
legacyPath: >-
  /paper
  shorts/2020/10/01/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale.html
tags:
  - Other
field: 'Vision Foundations'
summary: "2020 – An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale"
---
## 2020 – An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale

**arXiv:** [2010.11929](https://arxiv.org/abs/2010.11929)

**GitHub:** [google-research/vision_transformer](https://github.com/google-research/vision_transformer)

**Project page / Google AI Blog:** [Transformers for Image Recognition at Scale](https://research.google/blog/transformers-for-image-recognition-at-scale/)

**Conference:** ICLR 2021

## Summary

> ViT shows that a standard Transformer can become a strong image classifier when patches replace words and pretraining supplies enough visual experience. The decisive result is a data-regime crossover: larger ViTs underperform smaller ones on ImageNet-only training, but benefit much more from JFT-300M. The 88.55% ImageNet headline belongs to ViT-H/14 after large supervised pretraining and high-resolution fine-tuning. The paper also finds that learned spatial structure, a retuned pooling head, and patch size matter more than a simple claim that attention eliminates visual priors.

## Core Insights

### Patch size controls the sequence the Transformer sees

For an RGB image of size $224\times224$, a $16\times16$ patch contains $16\times16\times3=768$ scalar values. There are $14\times14=196$ such patches. A shared linear projection maps each flattened patch into the model width; for ViT-B/16, that width is also 768. Adding a learned class token gives a sequence of shape `[batch, 197, 768]`. The coincidentally equal input-patch and embedding widths are not a requirement: other patch sizes and model widths change them independently.

Follow the overview from the image tiles through the projection. Each tile becomes one token, learned position embeddings give it an address, and the Transformer updates all tokens through global self-attention. The class token is trained to gather information useful for classification. Inside the encoder, layer normalization precedes attention and the MLP, each followed by a residual addition; the MLP uses GELU.

![Vision Transformer patch embedding and encoder architecture](/assets/images/vit-source-figure-1-architecture.png)
*Fig 1: A shared projection turns image patches into tokens, learned positions preserve their arrangement, and a pre-normalized Transformer builds the class-token representation. | source: [Vision Transformer, Figure 1](https://arxiv.org/abs/2010.11929)*

The patch operation already makes a spatial assumption: nearby pixels are grouped and processed together. The architecture uses fewer image-specific priors than a CNN, especially in global attention, but it is not devoid of spatial structure. Nor does the paper establish exact translation invariance; it shows that useful spatial relationships can be learned from data.

The sequence length is $N=HW/P^2$. At fixed image size, halving patch width doubles the number of patches along each axis, making four times as many tokens and roughly sixteen times as many patch-to-patch attention scores. Parameters need not grow with that token count. This is why a model can become substantially more expensive without becoming substantially larger in parameter count.

### The data crossover is more revealing than the largest score

The plot compares transfer to ImageNet after pretraining on ImageNet, ImageNet-21k, or JFT-300M. The shaded region shows the BiT ResNet range. On the smallest dataset, CNN priors help and the larger ViT is harder to fit usefully. As pretraining data increases, the larger ViT gains more and the ordering changes.

![ImageNet transfer accuracy as pretraining dataset size increases](/assets/images/vit-source-figure-3-data-scale.png)
*Fig 2: ViT’s advantage depends on pretraining scale; larger variants gain disproportionately on JFT, while BiT ResNets are stronger in the smaller-data regime of this study. | source: [Vision Transformer, Figure 3](https://arxiv.org/abs/2010.11929)*

Appendix Table 5 gives the controlled transfer numbers below. All use 384-pixel fine-tuning without the extra techniques used for the headline ImageNet result.

| Pretraining data | ViT-B/16 | ViT-L/16 | ImageNet top-1 after fine-tuning |
| --- | ---: | ---: | --- |
| ImageNet, about 1.3M images | 77.91% | 76.53% | Larger model is worse |
| ImageNet-21k, about 14M images | 83.97% | 85.15% | Larger model begins to help |
| JFT-300M, about 303M images | 84.15% | 87.12% | Larger model gains substantially |

This is a result for the paper’s architecture and regularization choices. It does not prove that every Transformer needs 300 million images. The authors tune basic regularization for the smaller datasets, and the public ImageNet-21k model already transfers well. A second experiment keeps JFT-subset hyperparameters fixed and measures linear few-shot accuracy on frozen features; that supports the crossover too, but it is a different evaluation from full fine-tuning.

The main pretraining is supervised. JFT labels and ImageNet labels teach the visual representations; the fact that the network resembles a language Transformer does not make its training objective next-token prediction. The paper’s masked-patch experiment is a separate preliminary study.

### High-resolution transfer changes the token budget

Fine-tuning replaces the pretrained two-layer classification head with a zero-initialized linear head for the target classes. The paper normally pretrains at 224-pixel resolution and fine-tunes at 384. With $16\times16$ patches, that changes the grid from $14\times14$ to $24\times24$: 196 patch tokens become 576. The patch-to-patch score matrix grows by about $(576/196)^2\approx8.6$ times, even though the learned attention projections retain the same shapes.

The old positional table has only one embedding per training-grid location. ViT reshapes its patch positions into their original 2D grid and interpolates them to the new grid, keeping the class position separate. This is another explicit use of image geometry. Merely accepting a longer sequence does not make the original position table fit the new resolution.

The headline results go beyond the default 384 setting. Table 2 fine-tunes ViT-L/16 at 512 and ViT-H/14 at 518, and uses Polyak averaging. With JFT pretraining, they reach 87.76% and 88.55% ImageNet top-1 respectively. ViT-H/14 has 32 layers, width 1,280, and about 632M parameters. Those scores do not belong to the 86M-parameter ViT-B/16.

### The compute advantage has a controlled comparison

The large-model table reports about 680 TPUv3-core-days for JFT-pretrained ViT-L/16 and 2,500 for ViT-H/14, versus 9,900 for BiT-L and 12,300 for Noisy Student. Core-days multiply accelerator count by elapsed days; they are not per-image inference FLOPs. The paper explicitly notes that schedules, optimizers, and regularization contribute to these whole-run differences.

Its separate scaling study trains ViTs, BiT-style ResNets, and hybrids on JFT under a shared experimental framework. Across the average of five transfer tasks, ViT reaches comparable performance with roughly 2–4 times less pretraining compute. Small-budget hybrids benefit from a convolutional feature extractor before attention; at larger scales, their advantage fades. That is the more useful evidence for the architectural trade-off than dividing two unrelated headline training costs.

The TPU timing appendix also cautions against treating asymptotic attention cost as measured end-to-end latency. It benchmarks peak throughput across batch sizes, and the steep high-resolution scaling becomes prominent only for the largest models and resolutions tested. Attention-score growth is real, while its share of total runtime depends on the rest of the model and the implementation.

### The model learns spatial structure, but the class token is optional

Figure 7 inspects three parts of the representation. On the left, patch-projection filters resemble local visual basis functions. In the middle, nearby patches acquire similar positional embeddings, with row and column structure emerging from learned vectors. On the right, different attention heads span different distances: some stay local early, while others already gather information across much of the image. Later layers attend more broadly.

![Learned patch filters, positional similarities, and attention distances in ViT](/assets/images/vit-source-figure-7-representations.png)
*Fig 3: Patch filters capture local structure, position embeddings recover spatial relationships, and attention heads mix local and global evidence across depth. | source: [Vision Transformer, Figure 7](https://arxiv.org/abs/2010.11929)*

The positional ablation supports a modest conclusion. ImageNet five-shot linear accuracy is 61.38% without positions and 64.21% with the default learned 1D positions; learned 2D and relative alternatives are about 64.0%. Position matters here, while the tested encoding variants are close. These five-shot values are not the paper’s fully fine-tuned classification scores.

The class token has an equally instructive ablation. Global average pooling initially looked much worse, but that gap disappeared after retuning the learning rate. The special token is a workable interface, not a demonstrated necessity. The paper’s qualitative object-attention visualizations likewise use attention rollout averaged across heads and composed through layers; they should not be described as individual heads or treated as ground-truth segmentation masks.

The exploratory self-supervised model corrupts half the patch embeddings and predicts a quantized mean color for each corrupted patch. It reaches 79.9% ImageNet accuracy, roughly two points over training from scratch but about four below its supervised-pretraining comparison. This establishes a promising starting point for later visual self-supervision, with substantial room left by this particular target.

The existing patch-embedding example below implements the fixed-resolution token interface with a strided convolution equivalent to the shared patch projection. It outputs tokens only; it does not include the Transformer, classifier, or positional interpolation needed for a resolution change.

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16,
                 in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim))

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x + self.pos_embed
```

## High-Level Takeaways

- ViT turns patches into a standard Transformer sequence; patch size controls attention cost even when parameter count barely changes.
- The paper’s central evidence is a data-regime crossover, with large supervised pretraining making bigger ViTs substantially more useful.
- The 88.55% ImageNet result is ViT-H/14 with high-resolution fine-tuning and averaging, not a small-model or default-resolution score.
- Learned positions recover useful image structure, and average pooling can replace the class token after learning-rate adjustment.
- Controlled scaling supports a pretraining-compute advantage, while the masked-patch experiment remains a separate, weaker self-supervised result.
