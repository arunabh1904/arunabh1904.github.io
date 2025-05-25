---
layout: content
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
date: 2019-06-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Natural Language Processing
---
## 2019 – BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

**arXiv:** [1810.04805](https://arxiv.org/abs/1810.04805)  
**GitHub:** [google-research/bert](https://github.com/google-research/bert)  
**Project page / blog:** [Google AI Blog – "Open Sourcing BERT"](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)  
**Conference:** NAACL 2019

![BERT Model](/assets/images/bert.png)

### tiny_pretrain.py – core BERT objectives in ~40 LOC
```python
import random, torch
from typing import Iterable, Tuple, Generator
from transformers import BertTokenizer

def mask_tokens(ids: torch.Tensor | list[int], tok: BertTokenizer,
                p: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor(ids)
    is_special = torch.tensor(tok.get_special_tokens_mask(ids.tolist(), True)).bool()
    mask = ~is_special & (torch.rand_like(ids.float()) < p)

    labels = torch.where(mask, ids, torch.full_like(ids, -100))
    rand = torch.randint(len(tok), ids.shape, dtype=torch.long)

    ids[mask] = torch.where(torch.rand_like(ids.float()) < 0.8,
                            tok.mask_token_id,
                            ids[mask])
    ids[(ids == tok.mask_token_id) & (torch.rand_like(ids.float()) < 0.125)] = \
        rand[(ids == tok.mask_token_id) & (torch.rand_like(ids.float()) < 0.125)]
    return ids, labels


def make_sentence_pairs(docs: Iterable[str], ratio: float = 0.5
                        ) -> Generator[Tuple[str, str, int], None, None]:
    docs = [d.split('.') for d in docs]
    for sents in docs:
        sents = [s.strip() for s in sents if s.strip()]
        for i in range(len(sents)-1):
            a, b_true = sents[i], sents[i+1]
            if random.random() < ratio:
                b = random.choice(random.choice(docs)).strip()
                yield a, b, 1
            else:
                yield a, b_true, 0
```

### BERT, Six Years On: What Still Matters

TL;DR A 12-layer (base) or 24-layer (large) Transformer encoder, pre-trained on 3.3 B tokens with the two code snippets above, still underpins half the NLP landscape.

---

## How the Pre-training Works

### Masked-Language Modelling (MLM)

The `mask_tokens` function corrupts 15 % of input pieces at random, forcing the network to infer each blank from both left and right context. This made BERT genuinely bidirectional, unlike GPT-1’s left-to-right guessing game or ELMo’s shallow concatenation of two one-directional RNNs.

### Next-Sentence Prediction (NSP)

`make_sentence_pairs` creates 50 % real A→B pairs and 50 % mismatches. The classifier sitting on `[CLS]` learns discourse-level coherence. Later work (RoBERTa) showed you can ditch NSP without hurting accuracy, but the idea seeded a huge body of inter-sentence objectives (e.g. sentence embedding models, dense retrieval).

---

## Key Numbers

| Benchmark | Metric | Prev SOTA | BERT-large | Gain |
| --------- | ------ | --------- | ---------- | ---- |
| GLUE | Avg. score | 72.8 | 80.5 | +7.7 |
| SQuAD 1.1 | F1 | 88.5 | 93.2 | +4.7 |
| SQuAD 2.0 | F1 | 78.0 | 83.1 | +5.1 |
| MNLI | Acc. | 82.1 | 86.7 | +4.6 |

Training cost: ~4 days on 16 TPU v3 chips; a single downstream fine-tune (e.g. QA) ~30 min on one TPU.

---

## Why BERT Hit Hard
1. True bidirectionality — MLM beat GPT-style causal masks by reading the whole sentence at once.
2. One backbone, many heads — QA, NLI, NER, sentiment: swap a task head, fine-tune end-to-end, done.
3. `[CLS]` + segment embeddings — became the default recipe for pairwise classification and retrieval.
4. Open weights & code — bootstrapped an ecosystem of tweaked replicas (RoBERTa, ALBERT, DistilBERT, etc.).

---

## Where It Falls Short
- Compute-hungry — four-day TPU bills were eye-watering in 2018; still non-trivial today.
- NSP flop — later ablations showed it adds little; many modern variants drop it.
- 512-token ceiling and quadratic attention — hopeless on long documents without hacks.
- Static knowledge — BERT knows nothing after late 2018; domain adaptation or continual learning is required.

---

## Looking Forward

BERT’s recipe is a floor, not a ceiling. Better corruption schemes (SpanBERT, DeBERTa’s disentangled masks), cheaper compressors (DistilBERT, TinyBERT), and smarter attention (Longformer, FlashAttention) all stand on its shoulders. Yet the two short snippets at the top remain the intellectual core: mask some tokens, test sentence order, back-propagate, repeat. That elegance is why BERT is still worth teaching in 2025.

