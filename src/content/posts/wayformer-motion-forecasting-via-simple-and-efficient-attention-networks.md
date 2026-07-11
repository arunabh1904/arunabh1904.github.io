---
title: 'Wayformer: Motion Forecasting via Simple and Efficient Attention Networks'
date: '2022-07-12T04:00:00.000Z'
section: paper-shorts
postSlug: wayformer-motion-forecasting-via-simple-and-efficient-attention-networks
legacyPath: /paper shorts/2022/07/12/wayformer-motion-forecasting-via-simple-and-efficient-attention-networks.html
tags:
  - Other
field: BEV
summary: Wayformer shows that a simpler attention stack can fuse heterogeneous road, agent, and traffic-light inputs for motion forecasting when the fusion strategy is chosen carefully.
---
## 2022 - Wayformer

**arXiv:** [2207.05844](https://arxiv.org/abs/2207.05844)

**Summary:** Wayformer asks whether motion forecasting really needs many modality-specific modules. Its answer is mostly no: a homogeneous attention architecture can work well if it fuses static and dynamic scene tokens in the right place.

The paper is useful because it turns forecasting architecture design into a set of fusion and efficiency choices: early fusion, late fusion, hierarchical fusion, factorized attention, and latent-query attention.

## Paper Insights

Wayformer encodes heterogeneous driving inputs such as road geometry, lane connectivity, traffic light state, agent histories, and agent interactions with Transformer-style attention. The model studies several fusion patterns and efficient attention variants, then shows that early fusion is especially strong on Waymo Open Motion Dataset and Argoverse.

The design philosophy is close to "make the representation uniform, then spend effort on scaling attention." The caveat is that simple attention can hide useful structure: the model may learn relations that methods like LaneGCN encode explicitly.

![Figure 1 from Wayformer showing an encoder-decoder Transformer for multimodal scene inputs and trajectory distributions](/assets/images/wayformer-motion-forecasting-via-simple-and-efficient-attention-networks-paper-figure.png)
_Figure 1 shows Wayformer as an encoder-decoder attention network over heterogeneous scene tokens, with multimodal trajectory prediction at the output. From the [Wayformer paper](https://arxiv.org/abs/2207.05844), via ar5iv._

**What to look at:**
- Heterogeneous inputs become a shared token set.
- Early fusion lets agents, roads, and signals interact before heavy abstraction.
- Factorized and latent-query attention trade accuracy for speed and memory.

**Evals / Benchmarks / Artifacts:**

| Design choice | Detail | Why it matters |
| ------------- | ------ | -------------- |
| Fusion | Early, late, and hierarchical variants | Tests where heterogeneous scene information should meet. |
| Efficiency | Factorized and latent-query attention | Makes large scene attention more practical. |
| Inputs | Road geometry, traffic lights, agent history | Covers the messy inputs forecasting systems actually use. |
| Benchmarks | Waymo Open Motion Dataset and Argoverse | Compares across major public motion forecasting settings. |

**Context:** Wayformer helped normalize unified attention over heterogeneous driving scenes while keeping the architecture relatively simple.

**Takeaway:** A strong general attention backbone can compete with highly specialized forecasting stacks when the fusion strategy is right.
