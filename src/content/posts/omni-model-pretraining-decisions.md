---
title: 'Omni-Model Pretraining: Zero to Hero'
date: '2026-07-15T09:00:00.000Z'
section: blog
postSlug: omni-model-pretraining-decisions
legacyPath: /blog/2026/07/15/omni-model-pretraining-decisions.html
tags:
  - Multimodal AI
  - Pretraining
  - Research Leadership
summary: A paper-linked guide to representation, modality sharing, objectives, data mixtures, proxy scaling, video and action modeling, and reliable large-scale omni-model training.
---

# Omni-Model Pretraining: Zero to Hero

An omni-model is easy to describe and expensive to specify. Put text, images, video, audio, and actions into one model; train at scale; expect transfer. Every clause hides a decision that can waste a major run.

Should an image be a sequence of discrete codes or a grid of continuous latents? Should understanding and generation share a tokenizer? Should video frames compete with text tokens in the same context window? Should actions use the language vocabulary, a frequency tokenizer, or a continuous expert? If a mixture works at 300M parameters, will it still work at 30B? If one modality's loss falls faster, is that positive transfer or gradient domination?

Pretraining leadership is the ability to reduce those questions to evidence before the full bill arrives.

![Omni-model pretraining decision stack](/assets/images/omni-pretraining-decision-stack.svg)
_The stack is coupled: representation changes sequence length, which changes systems cost, mixture economics, and the capabilities that can be measured._

This guide follows the decisions in the order they should be made. Its companion, [Post-Training Vision-Language-Action Models: Zero to Hero](/blog/2026/07/16/post-training-vision-language-action-models-zero-to-hero.html), starts where this one ends: a pretrained model enters deployment, produces failures, and must improve without losing its general capabilities.

The scope is pretraining design, not a catalog of multimodal models. I use papers in two ways: reported experiments establish what happened inside a particular recipe; the decision tests, kill criteria, and preferred architecture are my synthesis. Keeping that boundary visible matters because multimodal papers often change the tokenizer, data, parameter count, and training budget together. A strong result can justify adopting a recipe without identifying which ingredient caused the gain.

## 1. Define the capability contract before the architecture

“Multimodal” is not a capability. A contrastive encoder, visual assistant, image generator, video predictor, and robot policy can all consume images and text while solving different problems.

Write the contract as observable behavior:

- retrieve and classify open-vocabulary concepts;
- answer questions that require fine visual grounding;
- generate images with controllable structure;
- understand and generate temporally coherent video;
- preserve state and action-conditioned consequences;
- emit actions at a deployment control rate;
- add a new modality without retraining every component.

The contract determines what must be shared. [CLIP](/paper%20shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html) only needs image and text representations to meet in an embedding space. [LLaVA](/paper%20shorts/2023/04/01/visual-instruction-tuning-llava.html) needs visual tokens to enter a generative language model. [Chameleon](/paper%20shorts/2024/05/16/chameleon-mixed-modal-early-fusion-foundation-models.html) asks one autoregressive transformer to understand and generate mixed-modal sequences. [Transfusion](/paper%20shorts/2024/08/20/transfusion-predict-the-next-token-and-diffuse-images-with-one-multimodal-model.html) shares a transformer while giving text and images different objectives. [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html) shares semantic context but gives continuous control a flow-based action expert.

No architecture dominates those contracts for free.

## 2. Representation is the first scaling law

Before choosing transformer depth, decide what counts as one training unit.

Text already arrives as compressed symbolic tokens. Images can be patches, contrastive features, discrete codes, or continuous latents. Video adds a temporal sampling policy on top of spatial compression. Actions can be per-dimension bins, chunks, frequency coefficients, diffusion trajectories, or flow samples.

The unit determines sequence length:

$$
T_{total}
=T_{text}+T_{image}+T_{video}+T_{audio}+T_{action}.
$$

Attention cost, context competition, batch composition, and loss weighting all inherit that choice.

[ViT](/paper%20shorts/2020/10/01/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale.html) made image patches look like tokens, but a 16×16 patch is not semantically equivalent to a wordpiece. [Qwen2-VL](/paper%20shorts/2024/09/01/qwen2-vl-enhancing-vision-language-model-perception-of-the-world-at-any-resolution.html) lets visual-token count follow input resolution, preserving small text and document detail at variable cost. [DeepSeek-VL2](/paper%20shorts/2024/12/01/deepseek-vl2-mixture-of-experts-vision-language-models.html) combines dynamic tiling with sparse language capacity and latent attention to manage that cost.

The first useful budget is therefore not “percentage of image data.” It is tokens or FLOPs per capability gain. A 10% video mixture can consume most of the compute if every clip expands into thousands of visual tokens.

### Understanding and generation want different information

Understanding rewards invariance. A classifier should ignore texture changes that preserve object identity. Generation punishes lost detail. A tokenizer optimized for semantic invariance can be a poor reconstruction code; a pixel-faithful code can waste sequence budget on information irrelevant to reasoning.

[Janus](/paper%20shorts/2024/10/17/janus-decoupling-visual-encoding-for-unified-multimodal-understanding-and-generation.html) responds by separating the visual encoders for understanding and generation while sharing the transformer. [TokenFlow](/paper%20shorts/2024/12/04/tokenflow-unified-image-tokenizer-for-multimodal-understanding-and-generation.html) uses dual codebooks linked by shared indices to preserve semantic and fine-grained information. [Transfusion](/paper%20shorts/2024/08/20/transfusion-predict-the-next-token-and-diffuse-images-with-one-multimodal-model.html) avoids discrete image generation and diffuses continuous patches through a shared trunk.

Those designs are three answers to one question: where should modality specialization live?

| Specialization boundary | Representative paper | What remains shared | Main risk |
| --- | --- | --- | --- |
| None: one discrete stream | Chameleon | Token space, transformer, next-token loss | Visual compression and modality interference |
| Separate visual encoders | Janus | Multimodal transformer | More interfaces and capacity accounting |
| Dual-purpose tokenizer | TokenFlow | Token indices and downstream model | Coupled codebooks may fail under severe compression |
| Modality-specific loss/I-O | Transfusion | Transformer blocks and cross-modal context | Loss balance and serving complexity |

The kill experiment is matched compute and matched sequence length. If separate routes win only because they add parameters or visual tokens, the result is capacity, not reduced interference.

## 3. Sharing is a spectrum, not a binary choice

Architectures often get labeled “early fusion” or “late fusion,” but the operational choices are more granular:

- input tokenizer and embedding;
- positional encoding;
- attention and MLP blocks;
- normalization parameters;
- mixture-of-experts routing;
- output head and loss;
- optimizer state and learning-rate schedule.

[Scaling Laws for Native Multimodal Models](/paper%20shorts/2025/04/10/scaling-laws-for-native-multimodal-models.html) finds no inherent late-fusion advantage in its studied regime. Early fusion is stronger at smaller scale and simpler to deploy, while modality-aware MoE parameters restore specialization inside a unified model. [Perceiver IO](/paper%20shorts/2021/07/30/perceiver-io-a-general-architecture-for-structured-inputs-and-outputs.html) offers a different compression principle: map large structured inputs into a fixed latent bottleneck, process there, and query structured outputs.

The right sharing boundary depends on whether gradients agree. Build an objective-interaction matrix rather than arguing from intuition:

| Add this objective ↓ / measure this capability → | Text | VLM | Image generation | Video | Action |
| --- | --- | --- | --- | --- | --- |
| Text next-token prediction | — | ? | ? | ? | ? |
| Image–text understanding | ? | — | ? | ? | ? |
| Image generation | ? | ? | — | ? | ? |
| Video prediction | ? | ? | ? | — | ? |
| Action prediction | ? | ? | ? | ? | — |

Every cell should report transfer under a matched compute budget, along with gradient cosine similarity, per-objective gradient norm, update norm by module, and throughput. A lower joint loss does not imply that any downstream capability improved.

This is where [MM1](/paper%20shorts/2024/03/14/mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training.html) is so useful. Its controlled studies suggest that image encoder quality, resolution, visual-token count, and data composition matter more than endlessly modifying the connector. That is an experiment-allocation result: sweep the variables with large causal leverage before polishing the bridge between them.

## 4. One model can use several losses

A unified transformer does not require a unified objective.

Text commonly uses next-token cross-entropy:

$$
\mathcal{L}_{text}
=-\sum_t\log p_\theta(x_t\mid x_{<t}).
$$

Contrastive understanding uses paired image–text scores. [SigLIP](/paper%20shorts/2023/10/01/sigmoid-loss-for-language-image-pre-training-siglip.html) replaces batch-softmax competition with independent sigmoid pair losses, reducing dependence on enormous synchronized negative sets. Diffusion or flow predicts noise, velocity, or a vector field over a sampled time. Robot imitation may use token CE, L1 action error, denoising, or flow matching.

A joint objective looks simple:

$$
\mathcal{L}=\sum_m \lambda_m\,\mathcal{L}_m,
$$

but $\lambda_m$ is not merely a hyperparameter. Losses average over different units. One image example may contribute hundreds of patches, one video contributes many frames, and one robot episode contributes many correlated action steps. The apparent balance changes with sequence packing, mask count, and gradient accumulation.

Normalize and log at three levels:

1. per predicted unit—token, patch, frame, denoising target, or action dimension;
2. per example or trajectory, so long examples do not silently dominate;
3. per consumed FLOP, so an expensive modality has to justify its share.

Then measure parameter-level conflict. If video gradients are ten times larger in shared attention blocks, a small sampling share can still control the update. If text loss falls while VLM transfer regresses, aggregate convergence is hiding interference.

## 5. Data mixture is a policy over scarce compute

An omni corpus is not a pile of datasets. It is a sampling policy over modalities, domains, quality levels, sequence lengths, and training stages.

[Scaling Laws for Generative Mixed-Modal Language Models](/paper%20shorts/2023/01/10/scaling-laws-for-generative-mixed-modal-language-models.html) adds an interaction term to unimodal scaling so proxy runs can estimate synergy or competition between modalities. [Scaling Laws for Optimal Data Mixtures](/paper%20shorts/2025/07/12/scaling-laws-for-optimal-data-mixtures.html) treats mixture weights as variables in a fitted loss surface conditioned on model size and total data. The useful form is not one universal law but a family of target-specific predictions:

$$
L_m(N,D,h)
=E_m+A_mN^{-\alpha_m}+B_mD_m^{-\beta_m}+I_m(h,N,D),
$$

where $h$ is the mixture and $I_m$ captures transfer or interference.

The word “target” matters. A mixture optimal for image generation may be wrong for OCR, video reasoning, or action grounding. There is no context-free optimal dataset.

The proxy study should vary:

- model size $N$ and total compute;
- total examples and effective unique examples;
- modality/domain weights $h$;
- image and video compression;
- frame rate, clip duration, and context length;
- shared versus expert capacity;
- curriculum order and learning-rate resets.

Report uncertainty on extrapolated rankings, not only fitted loss. The expensive decision is whether candidate A will still beat candidate B at target scale. If confidence intervals overlap, the proxy run did not justify a nine-figure allocation.

### Curriculum changes the interaction

Data mixtures are often nonstationary. A model may first learn vision-language alignment, then high-resolution grounding, then video, then action. [VideoLLaMA 3](/paper%20shorts/2025/01/01/videollama-3-frontier-multimodal-foundation-models.html) uses strong image-text alignment as the base for video and compresses redundant temporal tokens. [PaliGemma](/paper%20shorts/2024/07/10/paligemma-a-versatile-3b-vlm-for-transfer.html) upcycles resolution in stages. [Eagle 2](/paper%20shorts/2025/01/01/eagle-2-post-training-data-strategies-for-frontier-vision-language-models.html) shows how post-training ordering and curation can make a smaller VLM competitive.

Order can reduce optimization difficulty, but it can also cause forgetting. A staged curriculum needs retention evals after every transition and a controlled comparison with an interleaved mixture under equal compute.

## 6. Video is not automatically a world model

Video generation produces plausible futures. A world model preserves action-conditioned consequences. A policy chooses actions that achieve an outcome.

That distinction is easy to state and easy to ignore when generated video looks good.

[Wan](/paper%20shorts/2025/03/26/wan-open-and-advanced-large-scale-video-generative-models.html) is the systems reference for large-scale latent video generation: spatial/temporal VAE compression, caption and motion filtering, model-size tradeoffs, and a consumer-scale 1.3B variant. [Genie](/paper%20shorts/2024/02/23/genie-generative-interactive-environments.html) learns latent actions from unlabeled video and conditions a dynamics model on those actions.

The evaluation contract must change when control enters. FID and visual preference are not enough. Measure:

- action controllability and intervention consistency;
- object permanence and state persistence;
- geometry and contact plausibility;
- temporal order and causal response;
- rollout stability under repeated model predictions;
- whether the learned latent action means the same thing across scenes.

A model that produces a plausible door opening after any action is a video generator with weak conditioning, not a useful world model.

## 7. Actions expose the limits of modality symmetry

Actions are not just another token type. They change the data distribution, carry embodiment-specific units, and operate under a control deadline.

[RT-2](/paper%20shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html) shows the appeal of treating actions as language tokens: one decoder, tractable autoregression, and direct transfer from semantic pretraining. [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html) compresses action chunks in frequency space so high-frequency trajectories do not explode sequence length. [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html) uses a continuous flow expert instead of forcing actions through the text head.

The action-interface memo should compare:

| Choice | Likelihood | Multimodality | Latency | Main failure |
| --- | --- | --- | --- | --- |
| Direct regression | Explicit/simple | Weak | Low | Mode averaging |
| Discrete tokens | Exact autoregressive | Moderate | Sequential | Quantization and slow chunks |
| Frequency tokens | Exact autoregressive | Moderate | Shorter sequence | Loses abrupt corrections |
| Diffusion/flow | Implicit or path-based | Strong | Iterative/parallel | Complex post-training likelihood |
| Separate action expert | Interface-dependent | Strong | Extra serving path | Semantic/control coordination |

The pretraining question is which action prior should be learned across robots. The post-training question is whether that distribution exposes the likelihoods and responsiveness required by the improvement loop.

## 8. Scaling evidence needs a falsification boundary

A smooth curve is not the same as a causal law. Every scaling claim should name:

- the fitted range of model size, data, and compute;
- the architecture and tokenizer held fixed;
- the target loss or capability measured;
- the data quality assumptions;
- the residuals and confidence interval;
- the scale jump being extrapolated;
- the experiment that would make the recommendation reverse.

The most dangerous extrapolation assumes the bottleneck stays fixed. At small scale, parameters may dominate. At larger scale, unique high-quality video, long-context communication, tokenizer distortion, or evaluation contamination may take over. A modality interaction measured with one representation may disappear when compression changes.

For every candidate architecture, write a kill criterion before the proxy runs:

- **Discrete early fusion:** kill if visual-token cost grows faster than capability at matched FLOPs.
- **Hybrid objectives:** kill if loss balancing is unstable or serving complexity buys no capability.
- **Separate visual routes:** kill if gains disappear under parameter- and token-matched controls.
- **Modality experts:** kill if routing imbalance and communication erase active-parameter efficiency.
- **Action-conditioned world model:** kill if interventions do not produce consistent causal changes.

That habit converts paper reading into capital allocation.

## 9. A large run is a fault-tolerant system

Once the architecture and mixture are chosen, the research question becomes operational: can the training system preserve the intended experiment for months?

[TorchTitan](/paper%20shorts/2024/10/09/torchtitan-one-stop-pytorch-native-solution-for-production-ready-llm-pre-training.html) is useful because it treats tensor, pipeline, and data parallelism, checkpointing, compilation, and logging as composable parts of one PyTorch-native stack. The throughput number matters, but restart time and diagnostic quality matter more when a failed run costs days.

The runbook should cover:

| Failure | Detection | Automatic response | Root-cause evidence |
| --- | --- | --- | --- |
| Sudden loss spike | Per-modality loss and gradient outlier | Skip/rollback and quarantine batch | Data IDs, optimizer state, activation stats |
| Slow degradation | Eval residual versus scaling prediction | Pause curriculum transition | Mixture, LR, norm and update trends |
| NaN/FP8 range failure | Non-finite tensors and amax history | Reduce scale, reload safe checkpoint | First offending layer/rank |
| MoE imbalance | Expert load and dropped-token rate | Adjust routing/capacity | Modality-by-expert traffic |
| Modality domination | Gradient/update share by module | Reweight or resample | Per-objective norms and cosine similarity |
| Corrupt video shard | Decode and temporal-integrity checks | Quarantine shard | Source and preprocessing lineage |
| Dataloader/network stall | Step-time decomposition | Replace worker/rank | Host, shard, topology, retry logs |
| Eval regression | Capability dashboard | Block checkpoint promotion | Data/architecture changes since last good point |

Every checkpoint must bind model state to optimizer, scheduler, data cursor, mixture policy, tokenizer, code commit, and evaluation config. A weight file without that lineage is not a recoverable experiment.

## 10. The minimum convincing proxy program

Before a massive run, build a 100M–1B prototype program with at least text prediction, image-text understanding, visual generation, and a lightweight video or action objective.

Run three layers of experiments:

1. **Representation sweep:** image/video compression, visual tokens, action interface, shared versus separate encoders.
2. **Interaction sweep:** mixture weights, loss normalization, gradient conflict, curriculum order.
3. **Scaling sweep:** several model and data sizes with held-out capability evaluations and confidence intervals.

The deliverables should be decision artifacts:

- an architecture memo with matched-budget alternatives;
- an objective-interaction matrix;
- a fitted mixture model with uncertainty;
- a throughput and memory model by modality;
- a failure playbook and restart objective;
- a capability dashboard with kill criteria.

Do not present one mixture such as “45% text, 25% image-text, 10% image generation, 15% video, 5% action” as a universal answer. Present how that allocation was estimated, which target it optimizes, how uncertainty changes the ranking, and what evidence triggers a new allocation.

## 11. Reading path: broad map to literature depth

**Layer 1: representation foundations.** Read [ViT](/paper%20shorts/2020/10/01/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale.html), [CLIP](/paper%20shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html), [SigLIP](/paper%20shorts/2023/10/01/sigmoid-loss-for-language-image-pre-training-siglip.html), and [LLaVA](/paper%20shorts/2023/04/01/visual-instruction-tuning-llava.html). Track the training unit and the visual information each interface preserves.

**Layer 2: unified architecture families.** Read [Chameleon](/paper%20shorts/2024/05/16/chameleon-mixed-modal-early-fusion-foundation-models.html), [Transfusion](/paper%20shorts/2024/08/20/transfusion-predict-the-next-token-and-diffuse-images-with-one-multimodal-model.html), [Janus](/paper%20shorts/2024/10/17/janus-decoupling-visual-encoding-for-unified-multimodal-understanding-and-generation.html), [TokenFlow](/paper%20shorts/2024/12/04/tokenflow-unified-image-tokenizer-for-multimodal-understanding-and-generation.html), and [Native Multimodal Scaling Laws](/paper%20shorts/2025/04/10/scaling-laws-for-native-multimodal-models.html). For each, mark exactly where parameters and objectives split.

**Layer 3: experimental allocation.** Read [MM1](/paper%20shorts/2024/03/14/mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training.html), [Generative Mixed-Modal Scaling](/paper%20shorts/2023/01/10/scaling-laws-for-generative-mixed-modal-language-models.html), and [Optimal Data Mixtures](/paper%20shorts/2025/07/12/scaling-laws-for-optimal-data-mixtures.html). Reconstruct which curve is measured, which is extrapolated, and which control is missing.

**Layer 4: video, worlds, and actions.** Read [Wan](/paper%20shorts/2025/03/26/wan-open-and-advanced-large-scale-video-generative-models.html), [Genie](/paper%20shorts/2024/02/23/genie-generative-interactive-environments.html), [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html), and [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html). Ask whether the representation preserves causality or only appearance.

**Layer 5: systems.** Read [TorchTitan](/paper%20shorts/2024/10/09/torchtitan-one-stop-pytorch-native-solution-for-production-ready-llm-pre-training.html) with an operator's eye: what fails, what is checkpointed, what is measured, and which parallelism choice changes the research experiment?

Then move to the [post-training companion](/blog/2026/07/16/post-training-vision-language-action-models-zero-to-hero.html). Pretraining decides what the model can represent and how cheaply it can learn. Post-training decides whether those priors survive contact with deployment.

## The research thesis

The strongest omni model will probably not be the one that makes every modality identical. It will share the parameters that benefit from transfer and specialize the interfaces where invariance, fidelity, time, and control impose genuinely different requirements.

My preferred starting hypothesis is a shared transformer with modality-specific input/output experts, explicit per-objective accounting, and a data mixture chosen by proxy scaling rather than intuition. For physical intelligence, I would preserve metric, temporally persistent entity representations alongside open-vocabulary semantics. The generative model can imagine; the world model must preserve consequences; the policy must act; and the training system must reveal when one capability is improving at another's expense.

That is what “zero to hero” means in pretraining: not knowing every paper, but knowing how to turn a literature map into an experiment plan, a compute allocation, a kill decision, and a run that can survive long enough to answer the question.

## References

- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)
- [MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](https://arxiv.org/abs/2403.09611)
- [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818)
- [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/abs/2408.11039)
- [Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2410.13848)
- [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391)
- [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2501.09747)
- [Pi0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)
- [TorchTitan: One-stop PyTorch Native Solution for Production Ready LLM Pre-training](https://arxiv.org/abs/2410.06511)
