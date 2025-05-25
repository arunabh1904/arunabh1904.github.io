---
layout: content
title: "Learning Transferable Visual Models From Natural Language Supervision"
date: 2021-03-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Computer Vision
---

## 2021 – Learning Transferable Visual Models From Natural Language Supervision (CLIP)

**arXiv:** [2103.00020](https://arxiv.org/abs/2103.00020)

**GitHub:** [openai/CLIP](https://github.com/openai/CLIP) | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)

**Project / blog:** [OpenAI CLIP announcement](https://openai.com/index/clip/)

**Conference:** Released as a tech report; widely cited (ICML 2021 oral-style spotlight)

**Plain-language summary**
CLIP pairs an image encoder (ResNet-50/101 or ViT-B/32) with a text Transformer and trains them contrastively on 400&nbsp;M image–caption pairs scraped from the web. The model maximises cosine similarity for matching pairs and minimises it for mismatches. After pre-training, vision tasks become prompt engineering exercises: supply category names like "a photo of a tiger" and CLIP performs zero-shot classification by picking the text whose embedding is closest to the image embedding.

**Novel insights**
- Contrastive language–image pre-training scales GPT-style supervision to vision.
- Zero-shot prompting replaces hard-coded softmax heads, enabling one model for many datasets.
- Prompt engineering matters: templated phrases boost accuracy by roughly two points.
- Multimodal embeddings enable text-guided search and generation.

**Evals / Benchmarks**

| Dataset / task | Zero-shot top-1 | Supervised baseline | Encoder |
| -------------- | --------------- | ------------------- | ------- |
| ImageNet-1k | 76.2 % | 76.2 % (ResNet-50 with labels) | ViT-B/32 |
| Oxford-Pets | 88.3 % | 93.5 % (finetuned) | RN50 |
| UCF-101 (action) | 77.5 % | 84.2 % (I3D) | RN101 |
| Avg. over 27 tasks | 70.1 % | 75.3 % (task-specific models) | RN50 |

Training cost is roughly 400&nbsp;M pairs on 256&nbsp;A100 GPUs for around two weeks. Inference requires a single forward pass through both encoders.

**Tiny PyTorch snippet — zero-shot classifier with CLIP**
```python
import clip, torch
model, preprocess = clip.load("ViT-B/32", device="cuda")

labels = ["cat", "dog", "airplane"]
prompts = [f"a photo of a {c}" for c in labels]
text_feat = clip.tokenize(prompts).cuda()
with torch.no_grad():
    text_emb = model.encode_text(text_feat)
text_emb /= text_emb.norm(dim=-1, keepdim=True)

from PIL import Image
img = preprocess(Image.open("mystery.jpg")).unsqueeze(0).cuda()
with torch.no_grad():
    img_emb = model.encode_image(img)
img_emb /= img_emb.norm()

probs = (100. * img_emb @ text_emb.T).softmax(dim=-1)
pred = labels[probs.argmax()]
print("Predicted class:", pred)
```

**Critiques**
- **What I liked:** Elegant bridge between vision and language that spawned today's multimodal ecosystem. One model handles many tasks out of the box.
- **What I didn't like / open issues:** Heavy compute and proprietary dataset; sensitive to prompt wording; fairness concerns mirror web-scale data.

CLIP showed that natural-language supervision alone can endow vision models with rich, transferable semantics.

