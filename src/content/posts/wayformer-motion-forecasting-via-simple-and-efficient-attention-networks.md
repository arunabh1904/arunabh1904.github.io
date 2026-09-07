---
title: 'Wayformer: Motion Forecasting via Simple and Efficient Attention Networks'
date: '2022-07-12T00:00:00.000Z'
section: paper-shorts
postSlug: wayformer-motion-forecasting-via-simple-and-efficient-attention-networks
legacyPath: /paper shorts/2022/07/12/wayformer-motion-forecasting-via-simple-and-efficient-attention-networks.html
tags:
  - Other
field: 'Motion Forecasting & Planning'
summary: "2022 – Wayformer: Motion Forecasting via Simple and Efficient Attention Networks"
---
## 2022 – Wayformer

**arXiv:** [2207.05844](https://arxiv.org/abs/2207.05844)

### Method and reported result

Wayformer asks whether motion forecasting really needs many modality-specific modules. Its answer is mostly no: a homogeneous attention architecture can work well if it fuses static and dynamic scene tokens in the right place.

## Summary

> The paper is useful because it turns forecasting architecture design into a set of fusion and efficiency choices: early fusion, late fusion, hierarchical fusion, factorized attention, and latent-query attention.

## Core Insights

Wayformer encodes heterogeneous driving inputs such as road geometry, lane connectivity, traffic light state, agent histories, and agent interactions with Transformer-style attention. The model studies several fusion patterns and efficient attention variants, then shows that early fusion is especially strong on Waymo Open Motion Dataset and Argoverse.

The design philosophy is close to "make the representation uniform, then spend effort on scaling attention." The caveat is that simple attention can hide useful structure: the model may learn relations that methods like LaneGCN encode explicitly.

### Fusion placement is a latency decision

The latency sweep gives the architectural choice a useful boundary. With short budgets up to about 16 ms, late fusion is the strongest option in the paper's study; between roughly 16 and 32 ms, hierarchical fusion is competitive; once the budget exceeds 32 ms, early fusion catches up as the extra cross-modal interactions become affordable. The result is not a universal ranking of fusion schemes. It says that a fusion point is coupled to token count, attention depth, and the deployment budget.

There are also limits hidden by the uniform token interface. The experiments are ego-centric and repeatedly compute a scene representation for each target agent. Sparse abstract tokens can miss cues such as a pedestrian's visual pose or a vehicle's wheel angle, and independent per-agent decoding does not enumerate the combinatorial joint futures that an interactive planner may need. Wayformer's simplicity is strongest when its inputs already contain the relevant structure and the evaluation budget resembles the tested regime.

![Figure 1 from Wayformer showing an encoder-decoder Transformer for multimodal scene inputs and trajectory distributions](/assets/images/wayformer-motion-forecasting-via-simple-and-efficient-attention-networks-paper-figure.png)
*Fig 1: Shows Wayformer as an encoder-decoder attention network over heterogeneous scene tokens, with multimodal trajectory prediction at the output. | source: [Wayformer paper](https://arxiv.org/abs/2207.05844)*

![Figure 5 from Wayformer: Motion Forecasting via Simple and Efficient Attention Networks](/assets/images/wayformer-motion-forecasting-via-simple-and-efficient-attention-networks-source-figure-5.webp)
*Fig 2: Across attention layouts, lower minADE generally requires higher latency; interleaved, sequential, and multi-axis variants trace similar accuracy-efficiency frontiers around 16–48 ms. | source: [Wayformer: Motion Forecasting via Simple and Efficient Attention Networks](https://arxiv.org/abs/2207.05844)*

![Figure 6 from Wayformer: Motion Forecasting via Simple and Efficient Attention Networks](/assets/images/wayformer-motion-forecasting-via-simple-and-efficient-attention-networks-source-figure-6.webp)
*Fig 3: Increasing latent-token reduction lowers latency but degrades minADE, exposing the tradeoff between input compression and motion-forecasting accuracy. | source: [Wayformer: Motion Forecasting via Simple and Efficient Attention Networks](https://arxiv.org/abs/2207.05844)*






## High-Level Takeaways

- Wayformer informs how much forecasting quality comes from a specialized interaction graph versus a carefully chosen attention fusion strategy. Its atomic tokens represent agents, road elements, and traffic lights over time; early, late, or hierarchical fusion determines when those modalities can interact.
- The results favor early fusion in the tested regimes, but fusion choice is entangled with token count and attention approximation. The missing factorial study matches FLOPs while varying fusion point, latent bottleneck, and full versus factorized attention across scene densities. At 10× actors and map tokens, early fusion's quadratic interaction cost becomes decisive. The simplicity claim would fail if hierarchical fusion matched accuracy and calibration with a materially better latency curve.
- Wayformer helped normalize unified attention over heterogeneous driving scenes while keeping the architecture relatively simple.
- A strong general attention backbone can compete with highly specialized forecasting stacks when the fusion strategy is right.
