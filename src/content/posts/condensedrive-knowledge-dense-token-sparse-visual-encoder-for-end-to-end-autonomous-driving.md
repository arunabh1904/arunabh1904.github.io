---
title: >-
  ConDenseDrive: Knowledge-Dense, Token-Sparse Visual Encoder for End-to-End
  Autonomous Driving
date: '2026-10-12T09:00:00.000Z'
section: paper-shorts
postSlug: >-
  condensedrive-knowledge-dense-token-sparse-visual-encoder-for-end-to-end-autonomous-driving
legacyPath: >-
  /paper
  shorts/2026/10/12/condensedrive-knowledge-dense-token-sparse-visual-encoder-for-end-to-end-autonomous-driving.html
tags:
  - Autonomous Driving
  - Planning
field: 'Autonomous Driving: VLA & Planning'
summary: >-
  2026 – ConDenseDrive: Knowledge-Dense, Token-Sparse Visual Encoder for
  End-to-End Autonomous Driving
---
## 2026 – ConDenseDrive: Knowledge-Dense, Token-Sparse Visual Encoder for End-to-End Autonomous Driving

**Paper:** [arXiv:2610.16908](https://arxiv.org/abs/2610.16908) · [Full text, v1](https://arxiv.org/html/2610.16908v1)

## Summary

> ConDenseDrive improves a small driving encoder before compressing its tokens. A DeiT-S student learns visual, semantic, and geometric features from three teachers, adapts register tokens to planning, and then uses those registers to retain useful image patches. On NAVSIM v1, the two-frame model reaches 94.5 PDMS at 26.6 FPS on an A100, versus 94.3 PDMS at 13.6 FPS without compression. The result favors improving and routing a compact encoder before scaling it, within the paper's camera setup and simulator-based evaluations.

## Core Insights

### Distill the features before deciding which to discard

ConDenseDrive keeps a [DrivoR](</paper shorts/2026/01/08/driving-on-registers.html>)-style trajectory generator and scorer while changing the visual encoder. Its first stage trains a 22M-parameter DeiT-S student from three frozen teachers: DINOv3 ViT-L/16, SigLIP2-SO400M NaFlex, and Depth Anything 3-Large. They supply complementary patch-level visual, semantic, and geometric targets from NAVSIM front-camera images.

Teacher features have different widths and spatial resolutions. A separate convolutional adapter maps student features into each teacher's space, where cosine loss aligns feature direction and Smooth L1 matches values. The losses are weighted 0.9 and 0.1; teacher branches contribute equally. Only the student and adapters train. These adapters are distillation projectors, not vision-to-language connectors, and disappear after this stage.

The sequence matters. Compressing an encoder before it has learned driving-relevant representations could discard information based on the wrong notion of importance. The paper first learns richer patches, then learns which patches support planning.

![ConDenseDrive source Figure 2 showing teacher distillation, temporal register adaptation, and register-guided token compression](/assets/images/condensedrive-source-figure-2.webp)
*Fig 1: Teacher features train the small encoder first. Driving supervision then shapes register tokens, whose attention controls patch retention during the final compression stage. | source: [ConDenseDrive, Figure 2](https://arxiv.org/html/2610.16908v1#S4.F2)*

[View the figure at full resolution](/assets/images/condensedrive-source-figure-2.webp).

### Registers connect camera patches to planning

The second stage uses front, rear, left, and right images resized to 1136 × 672. Each camera and frame has 16 register tokens, which attend to that image's patches. Temporal attention then combines registers within each camera stream, adding learned time embeddings and relative ego-motion embeddings. Registers from all views and observed frames reach the trajectory generator and scorer, along with ego state and navigation information. There is no language model in this path.

The distilled backbone remains frozen and receives rank-32 LoRA updates in its query and value projections. Registers, temporal attention, and planning modules learn jointly. The objective adds a minimum L1 trajectory loss across candidates to binary cross-entropy for driving-quality scores. This is where the registers become useful to planning rather than generic image pooling.

Stage three starts from the complete dense driving checkpoint. At Transformer blocks 4, 7, and 10, attention from the register tokens scores the current patch tokens. The top fraction survives; the rest are combined into one weighted token. The weights are deliberately not renormalized, retaining their original attention mass. Register and CLS tokens always survive, and a fused token can participate in later compression. No separate routing network or routing loss is introduced.

The main two-frame model retains 40% at each compression block. That does not mean retaining 40% of the original patches throughout the encoder: compression acts repeatedly on the remaining sequence. The planner interface stays intact, so the main savings occur inside visual encoding.

### Training cost buys back inference cost

The three AdamW stages run for 50, 25, and 10 epochs, with batch sizes 128, 64, and 64. Learning rates are $5\times10^{-4}$ for distillation and $2\times10^{-4}$ thereafter, with weight decay 0.01, cosine decay, and 10% warmup. Stage three reinitializes optimizer state. Training uses four A100 GPUs; the end-to-end stages sample frames at 2 Hz, so two frames span a 0.5-second interval.

The paper trains on 103,288 NAVSIM navtrain scenes and tests on 12,146 navtest scenes. Its ablation separates representation quality from temporal context and compression:

| Configuration | Frames | PDMS | FPS |
| --- | ---: | ---: | ---: |
| Original DeiT-S | 1 | 91.8 | 25.4 |
| Distilled encoder | 1 | 94.1 | 25.4 |
| Distillation plus temporal attention | 2 | 94.3 | 13.6 |
| Temporal model plus token compression | 2 | 94.5 | 26.6 |

Most of the quality gain comes from distillation. Temporal context adds a smaller gain and nearly doubles latency; compression recovers that cost. Four dense frames reach only 93.9 PDMS at 7.3 FPS, so more history is not automatically useful. The dense-checkpoint initialization also matters: under the same one-frame compression setting, it scores 94.3 versus 93.6 when the driving components start from scratch.

### Match the evaluation contract before comparing scores

The headline NAVSIM v2 result is 90.8 EPDMS under the corrected evaluator; the original evaluator produces 87.1. Those numbers should never share an unlabeled leaderboard column. NAVSIM measures planning on recorded data and pseudo-simulation, so the comparison does not establish the same behavior in fully reactive traffic.

The paper also tests zero-shot transfer on all 436 HUGSIM scenarios, without training on HUGSIM's four source datasets. Here it uses a different configuration: one frame and a 0.7 keep rate. HD-Score increases from DrivoR's 32.5 to 37.4, with route completion rising from 47.2 to 49.9. The extreme-difficulty score nevertheless declines from 14.1 to 13.3. The average improvement leaves difficult failure modes unresolved.

Runtime uses batch one on a single A100, with 100 warmups and 1,000 synchronized timed iterations. It includes model inference but excludes loading and host preprocessing. The measured two-fold gain is therefore relative to the dense two-frame model under that protocol, not a promise about an entire vehicle stack.

The next decision test is whether the same retained patches remain sufficient under distribution shift. A small unusual obstacle may receive little attention precisely because the planner has not learned its importance. Matching the dense and compressed systems on rare hazards, repeated closed-loop rollouts, and target hardware would test whether average-score stability conceals a new failure tail. The paper's results justify compression as a candidate; they do not settle that boundary.

## High-Level Takeaways

- ConDenseDrive transfers three kinds of teacher features into a small encoder, then discards the teacher adapters before deployment.
- Planning-trained registers provide both the compact planner interface and the attention signal for repeated patch compression.
- The controlled ablation attributes most quality improvement to distillation and most recovered throughput to compression, while extra frame history alone offers little benefit.
- Corrected NAVSIM metrics, HUGSIM's different model configuration, and inference-only timing define the evidence boundary. Rare-hazard performance is the test that could reverse the compression decision.
