---
layout: content
title: "An Image Is Worth 16\u00d716 Words: Transformers for Image Recognition at Scale"
date: 2020-10-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Computer Vision
---

## 2020 – An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale

**arXiv:** [2010.11929](https://arxiv.org/abs/2010.11929)

**GitHub:** [google-research/vision_transformer](https://github.com/google-research/vision_transformer)

**Project page / Google AI Blog:** [Transformers for Image Recognition at Scale](https://research.google/blog/transformers-for-image-recognition-at-scale/)

**Conference:** ICLR 2021

![Vision Transformer](/assets/images/Vision Transformer VIT.png)

**Summary (abstract in plain English):** ViT slices an image into 16×16 patches, flattens them, adds a learnable positional embedding and a [CLS] token, then processes the sequence with a Transformer encoder. When pre-trained on very large datasets and fine-tuned, ViT matches or exceeds leading CNNs while using fewer training FLOPs.

**Novel insights:**
- Images become token sequences by patchifying into 16×16 windows and projecting each to an embedding.
- A learnable [CLS] token aggregates global information for classification.
- With over 100 M images, ViT learns locality and translation invariance without convolutions.
- Smaller models trained on massive data outperform larger networks trained on limited data.

**Evals / latency benchmarks:**

| Model | Pre-train data | Params | ImageNet-1k top-1 | Train FLOPs† | Inference FLOPs‡ |
| ----- | -------------- | ------ | ---------------- | ----------- | ---------------- |
| ViT-B/16 | ImageNet-21k | 86 M | 84.0 % | 55 B | 17.6 B |
| ViT-B/16 | JFT-300M | 86 M | 88.5 % | 184 B | 17.6 B |
| ViT-L/16 | JFT-300M | 307 M | 88.6 % | 604 B | 64.8 B |

*EfficientNet-L2 + NoisyStudent scores 87.1 % with roughly 480 B training FLOPs and 35 B inference FLOPs.*

†Single-crop training FLOPs. ‡Forward-pass FLOPs at 224² resolution.

**Tiny PyTorch snippet – patchifying + [CLS] token**

```python
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

**Critiques & limitations:**
- **What I liked:** Simple architecture unifies vision and NLP research. Patch-based tokens allow standard Transformers to handle images.
- **Limitations:** Vanilla ViT needs huge datasets like JFT-300M. Quadratic attention cost hampers very high resolutions and dense prediction tasks.

**Take-home message:** With sufficient data, a plain Transformer rivaled and sometimes beat convolutional backbones on image classification, inspiring numerous efficient and scaled-up vision models.
