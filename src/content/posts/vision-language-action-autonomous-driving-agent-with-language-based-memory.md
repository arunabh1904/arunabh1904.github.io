---
title: Vision-Language-Action Autonomous Driving Agent with Language-based Memory
date: '2026-09-29T09:00:00.000Z'
section: paper-shorts
postSlug: vision-language-action-autonomous-driving-agent-with-language-based-memory
legacyPath: /paper shorts/2026/09/29/vision-language-action-autonomous-driving-agent-with-language-based-memory.html
tags: ["Autonomous Driving", "Research"]
field: 'Autonomous Driving: VLA & Planning'
summary: 2026 – Vision-Language-Action Autonomous Driving Agent with Language-based Memory
---

## 2026 – Vision-Language-Action Autonomous Driving Agent with Language-based Memory

**Paper:** [arXiv:2609.38641](https://arxiv.org/abs/2609.38641) · [Full text and appendices](https://arxiv.org/html/2609.38641v1)

## Summary

> AD-Memo stores short descriptions of driving-relevant objects and events, then feeds those descriptions into later predictions. It trains a 2B Alpamayo 2 backbone with supervised learning followed by Da Capo, reinforcement learning that replays recorded driving observations while rolling out the model's own memory. This reduces the mismatch between reference and generated memory without simulating a responsive road environment. The evidence supports better open-loop prediction and scene recall; it does not demonstrate closed-loop vehicle control or a new visual projector.

## Core Insights

### Memory should preserve what the next frame cannot reveal

At an all-way stop, two stationary cars can look identical whether the other car arrived before or after ego. AD-Memo records those events in language. It targets roughly ten-second dependencies while retaining a short visual window: four images separated by 0.1 s, from a single-camera-focused backbone, plus tokenized past ego waypoints and language context. Every 0.4 s keyframe produces reasoning, a memory entry, and a predicted trajectory. Earlier entries become future input; intermediate frames can reuse the last keyframe's memory.

The output trajectory has **64 waypoints at 10 Hz over 6.4 seconds**. A terminal VQA task asks about the clip after the driving steps. The question is withheld while memory is being written, forcing the policy to preserve generally relevant evidence rather than just the eventual answer. A third task teaches standalone memory writing from a single frame and earlier memory.

![Source Figure 2a: AD-Memo supervised driving and language memory example](/assets/images/october-2609.38641-s3-f2-sf1.webp)
*Fig 1: The driving target includes a memory description that can be reused at later keyframes. Relevant object state is made explicit in language alongside reasoning and trajectory output. | source: [AD-Memo, Figure 2a](https://arxiv.org/html/2609.38641v1#S3.F2)*

[Open figure at full resolution](/assets/images/october-2609.38641-s3-f2-sf1.webp)

This is an **interface and training change to the VLA backbone**. The main experiments use a 2B Alpamayo 2 checkpoint without its action expert. Trajectories are generated through the backbone's token output, not a newly trained diffusion action head. The paper does not specify a new visual-language projector, its layer dimensions, or a complete image-resolution contract. The memory itself enters as language tokens. An experiment with the finalized Alpamayo action expert is explicitly left for future work because retraining that expert is expensive.

### The two datasets obtain their memory labels differently

The all-way-stop set filters more than one million Physical AI AV clips to **12,394 clips**. Ego must stop at at most 0.3 m/s for at least 0.2 s; another car must also stop within one second of ego's stopping interval; at most one vehicle can already be stopped when the clip begins. Ego trajectories and other vehicles' boxes identify stop/go events. Rules map those events into six states, from approaching the crossing through stopping to passing it, and concatenate per-car descriptions into memory.

Its non-overlapping clip split is **7,436 SFT / 2,479 RL / 2,479 test**. Questions ask arrival order, avoiding departure-order questions because ego departs last in 66.5% of pairwise cases. This is curated, memory-dependent supervision rather than an unbiased sample of all driving. A clip-level split is reported; a stronger geographic or recording-session separation is not established.

General driving uses **45,446 challenging clips**, split 27,246 / 9,088 / 9,112. Qwen3-VL-30B-A3B-Instruct selects complex interactions. Specialized perception grounds objects and map elements; a VLM adds semantic attributes; GPT-5.6 Luna selects evidence affecting ego's action, calibrates the action description against the recorded future, and links evidence to action in a typed decision graph. Rule checks reject malformed or disconnected graphs. Graphs are generated every 1.2 s, and language annotation fills the intervening keyframes.

Memory descriptions follow action-connected graph nodes and are merged across the clip to maintain object identity and avoid repeated, misleading statements. Questions are generated independently of memory. Training uses easier object questions, while testing uses multi-hop temporal questions. The labels therefore combine recorded geometry, perception outputs, model judgments, and scripted validation; they are not all independent human annotations. Average memory entries contain about 15 tokens in the stop set and 29 in general driving. Some data-generation prompts are disclosed only as excerpts.

### Da Capo closes the memory loop while leaving vehicle motion open-loop

SFT teaches driving-with-memory, terminal VQA, and—in general driving—standalone memory writing. It uses reference memories, creating exposure bias when the deployed model reads its own imperfect descriptions. Da Capo instead samples multiple full memory histories. At a given time, every rollout sees the same recorded images and past ego trajectory, but its earlier generated memory differs.

![Source Figure 3: AD-Memo semi-closed-loop reinforcement learning](/assets/images/october-2609.38641-s3-f3.webp)
*Fig 2: Parallel rollouts share recorded environmental observations but maintain separate generated memories. Driving tokens receive local credit, while memory tokens receive credit for later driving and terminal question answering. | source: [AD-Memo, Figure 3](https://arxiv.org/html/2609.38641v1#S3.F3)*

[Open figure at full resolution](/assets/images/october-2609.38641-s3-f3.webp)

The driving reward is $\max(-0.2\,\mathrm{ADE},-1)$. Because predicted driving does not alter later replayed observations, trajectory tokens receive a step-local return. Memory tokens receive the remaining driving rewards plus a terminal VQA reward weighted by 0.5. Group-centered, standard-deviation-normalized advantages then train the respective token blocks with a clipped policy objective. The causal credit assignment is the contribution: a past driving prediction should not receive credit for a future outcome it cannot affect in this replay setup.

The appendix proves expected-gradient equivalence to trajectory-level centering **without** standard-deviation normalization. It explicitly notes that the implemented normalization breaks that exact equivalence; its justification is empirical balancing of heterogeneous reward scales. The theorem should not be presented as a proof for the complete implemented algorithm.

### The recipe updates both vision and language, then samples memory histories

| Stage | Reported recipe |
| --- | --- |
| SFT | Two epochs; global batch 128; fused AdamW; vision rate $10^{-5}$, language/LM-head rate $10^{-4}$; weight decay 0.1; cosine decay with 100-step warm-up; clipping 1; bfloat16; 8,192-token limit. VQA weight 10; standalone-memory weight one for general driving and zero for stops. |
| Da Capo | Two stop-set epochs or one general-driving epoch; 12 clips × 12 rollouts per batch; AdamW at $3\times10^{-6}$; weight decay 0.01; KL coefficient 0.001; asymmetric clipping 0.20/0.28; temperature 0.6, top-p 0.98; 512 response / 4,096 context tokens. |
| Compute | SFT: eight H100 80GB GPUs, about 16 hours for 8,900 general-driving steps. RL: four eight-GPU nodes, divided between rollout and training, typically two to three days. |

The evaluation configuration lists a memory buffer of 50 keyframes, even though the motivation focuses on approximately ten-second dependencies. At 0.4 s spacing that buffer can cover a longer interval. Buffer capacity, typical clip length, and the measured dependency horizon are different quantities.

### Average prediction improves, but diversity and oracle metrics can worsen

Stop evaluation averages over **50,533 pre-stop keyframes from 2,479 clips**, using six sampled trajectories. AD-Memo reaches 1.866 m average ADE, 89.26% stop success, 45.01% go success, and 51.66% arrival-order QA accuracy. The no-memory model with the same RL recipe obtains 1.952 m, 88.92%, 43.76%, and 35.32%. Reference-memory SFT attains 89.53% QA accuracy versus 45.99% with its own memory, exposing substantial remaining memory-generation error.

Stop *evaluation* defines stopping below 0.5 m/s for 0.2 s and applies explicit timing tolerances; it differs from the 0.3 m/s *dataset selection* threshold. Most-likely ADE selects by sequence probability, while minADE is an oracle over six samples. Neither is a closed-loop rollout metric.

On general driving's **145,462 keyframes**, RL lowers average ADE from 2.011 to 1.864 m and most-likely ADE from 2.049 to 1.869 m relative to SFT. However, minADE worsens from 1.005 to 1.091 m, corner distance worsens, and QA remains roughly unchanged at 65.5%. A more concentrated policy can improve typical samples while losing alternate intentions. The paper's broad “best” language should therefore be read metric by metric.

Memory portability is evaluated by giving generated descriptions and the final frame to a separate model on LingoQA and WaymoQA. That supports a reusable language interface, but there is no matched vector-memory baseline establishing universal superiority, no completed action-expert experiment, and no reactive closed-loop safety test.

## High-Level Takeaways

- AD-Memo's reusable component is event-focused language memory, not a new visual projector or action expert.
- Rule-derived stop labels and model-assisted decision graphs have different provenance and failure modes.
- Semi-closed-loop RL corrects generated-memory exposure while replaying recorded vehicle motion.
- The recipe improves average open-loop prediction and temporal QA, but some oracle/diversity-sensitive metrics regress.
- Memory fidelity, token cost, long-term regulations, and transfer to a trained action expert remain open engineering questions.
