---
title: 'An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale'
date: '2020-10-01T04:00:00.000Z'
section: paper-shorts
postSlug: an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale
legacyPath: >-
  /paper
  shorts/2020/10/01/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale.html
tags:
  - Other
field: Computer Vision
summary: ViT showed that patchified images and standard Transformer encoders can rival CNNs when pre-training data is large enough.
---
## 2020 – An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale

**arXiv:** [2010.11929](https://arxiv.org/abs/2010.11929)

**GitHub:** [google-research/vision_transformer](https://github.com/google-research/vision_transformer)

**Project page / Google AI Blog:** [Transformers for Image Recognition at Scale](https://research.google/blog/transformers-for-image-recognition-at-scale/)

**Conference:** ICLR 2021

![Vision Transformer](/assets/images/Vision Transformer VIT.png)

**Summary:** ViT treats an image like a sequence. It slices the image into 16x16 patches, projects each patch into an embedding, adds positional embeddings and a `[CLS]` token, then feeds the sequence into a standard Transformer encoder. With enough pre-training data, that plain architecture matches or exceeds leading CNNs while using fewer training FLOPs.

The surprising part is not the patch trick by itself. It is that locality and translation invariance can emerge from data rather than being hard-coded through convolutions. ViT performs poorly when trained from scratch on smaller datasets, but with more than 100M images it becomes a strong visual backbone.

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

**Critiques & limitations:** ViT's appeal is its plainness: patchify the image and reuse the Transformer stack. That simplicity helped unify vision and NLP research. The cost is data hunger. Vanilla ViT needs huge pre-training datasets such as JFT-300M, and quadratic attention makes very high resolutions and dense prediction tasks expensive.

**Take-home message:** With enough data, a plain Transformer can rival convolutional backbones for image classification. ViT did not make convolutions obsolete overnight, but it made attention-first vision models credible.
