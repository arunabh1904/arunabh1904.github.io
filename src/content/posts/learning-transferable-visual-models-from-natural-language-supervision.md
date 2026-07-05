---
title: Learning Transferable Visual Models From Natural Language Supervision
date: '2021-03-01T04:00:00.000Z'
section: paper-shorts
postSlug: learning-transferable-visual-models-from-natural-language-supervision
legacyPath: >-
  /paper
  shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html
tags:
  - Other
field: Computer Vision
summary: >-
  2021 – Learning Transferable Visual Models From Natural Language Supervision
  (CLIP)
---
## 2021 – Learning Transferable Visual Models From Natural Language Supervision (CLIP)

**arXiv:** [2103.00020](https://arxiv.org/abs/2103.00020)

**GitHub:** [openai/CLIP](https://github.com/openai/CLIP) | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)

**Project / blog:** [OpenAI CLIP announcement](https://openai.com/index/clip/)

**Conference:** Released as a tech report; widely cited (ICML 2021 oral-style spotlight)

**Plain-language summary:** CLIP trains vision through language. It pairs an image encoder with a text Transformer and learns from 400M web image-caption pairs by pulling matching image/text embeddings together and pushing mismatches apart. After pre-training, classification becomes a prompt comparison problem: embed the image, embed candidate labels such as "a photo of a tiger", and choose the closest text embedding.

That framing replaced fixed softmax heads with natural-language supervision. One model could perform zero-shot classification across many datasets, and prompt wording became part of the evaluation surface. The same shared embedding space also made text-guided image search and generation feel natural.

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

**Critiques:** CLIP is one of the cleanest bridges between vision and language: one embedding space, many tasks, no dataset-specific classifier head. The weaknesses follow from the same design. Training depends on heavy compute and a proprietary web-scale dataset, zero-shot accuracy can be sensitive to prompt wording, and fairness problems mirror the data scraped from the web.

**Take-home message:** CLIP showed that natural-language supervision can give vision models rich, transferable semantics. It helped turn multimodal learning from a niche setup into a default way to build open-vocabulary systems.
