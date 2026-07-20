---
title: 'Pretraining Multimodal Models for Robotics: A Reading Guide'
date: '2026-07-15T09:00:00.000Z'
section: blog
postSlug: omni-model-pretraining-decisions
legacyPath: /blog/2026/07/15/omni-model-pretraining-decisions.html
tags:
  - Multimodal AI
  - Pretraining
  - Research Leadership
summary: A reading guide to the representations, robot data, action interfaces, mixtures, and scaling experiments behind multimodal robot policies.
---

# Pretraining Multimodal Models for Robotics: A Reading Guide

A robot can inherit the word “drawer” from the internet. The internet does not tell it how a sticky drawer feels, how far a particular arm can reach, or what to do after the gripper slips. Multimodal pretraining works because semantic and motor experience can transfer. It fails when “put everything in one model” becomes a substitute for deciding what should transfer, through which parameters, and under what evidence.

The hard choices arrive before the large run. An image can become patches, semantic features, discrete codes, or continuous latents. A robot trajectory can become per-axis bins, compressed action tokens, a diffusion target, or a flow field. A dataset mixture can be counted in examples, tokens, FLOPs, or gradient share; those allocations are not equivalent. A shared trunk can enable transfer while one high-volume modality quietly controls every update.

This guide is about making those choices legible. Its central claim is that a VLA is not created by adding an action head to a VLM. It is created by combining three priors—semantic, visual, and motor—without letting the cheapest one erase the others.

![Reading map from VLMs to deployed robot policies](/assets/images/multimodal-vla-reading-map.svg)
_Part II sits across the middle of the map. Robot pretraining must decide which experience deserves shared capacity and which action distribution the policy can execute on time._

![Omni-model pretraining decision stack](/assets/images/omni-pretraining-decision-stack.svg)
_The stack is coupled: representation changes sequence length, which changes systems cost, mixture economics, and the capabilities that can be measured._

This is Part II of the series. [Part I](/blog/2026/07/05/from-seeing-to-doing-the-evolution-of-vision-language-models.html) follows the visual interfaces that made language grounding possible. [Part III](/blog/2026/07/16/post-training-vision-language-action-models-zero-to-hero.html) starts when a pretrained policy enters deployment, creates its own state distribution, and has to learn from the result.

The scope is pretraining design, not a catalog of multimodal models. I use papers in two ways: reported experiments establish what happened inside a particular recipe; the decision tests, kill criteria, and preferred architecture are my synthesis. Keeping that boundary visible matters because multimodal papers often change the tokenizer, data, parameter count, and training budget together. A strong result can justify adopting a recipe without identifying which ingredient caused the gain.

## Begin with the capability contract

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

No architecture dominates those contracts for free. Before comparing models, write down which behavior must work zero-shot, which can be adapted with a small target dataset, which control rate is non-negotiable, and which regression would kill the program. A vague contract makes every later ablation look positive.

## Robot data is not web data

Internet data offers extraordinary breadth because images and text are cheap to copy. Robot trajectories are expensive, correlated, and attached to hardware. An hour of teleoperation contains the operator's habits, the controller's smoothing, the camera calibration, the reset procedure, and the parts of the state that happened to be logged. “More trajectories” can mean more task coverage, more embodiments, or simply more repetitions of one narrow behavior.

[Open X-Embodiment](/paper%20shorts/2023/10/13/open-x-embodiment-robotic-learning-datasets-and-rt-x-models.html) made cross-robot pretraining a concrete research program by standardizing data from 22 robot embodiments and reporting positive transfer in RT-X. [Octo](/paper%20shorts/2024/05/20/octo-an-open-source-generalist-robot-policy.html) trained a flexible policy on 800,000 Open X trajectories and treated new sensors, action spaces, and embodiments as adaptation problems. [OpenVLA](/paper%20shorts/2024/06/01/openvla-open-source-vision-language-action-model.html) combined internet-scale visual-language priors with 970,000 robot demonstrations in a 7B open model. [DROID](https://arxiv.org/abs/2403.12945) attacked a different axis: 76,000 trajectories across 564 scenes and 84 tasks, collected by 50 operators, to increase environmental diversity rather than only pooling existing benchmarks.

Those datasets teach different invariances:

| Variation in the corpus | Transfer the model might learn | Shortcut that can look like transfer |
| --- | --- | --- |
| More tasks on one robot | instruction and object semantics | memorized scene or controller conventions |
| More scenes with one setup | visual robustness and geometry | operator or reset regularity |
| More embodiments | shared task structure | averaging incompatible action units |
| More operators | recovery and style diversity | identity-specific timing artifacts |
| More failures and corrections | boundary states and recovery | learning the intervention device |

Cross-embodiment learning therefore needs an explicit interface contract. Which action dimensions are shared? Are commands expressed in end-effector deltas, joint positions, velocities, or gripper states? Which observation keys may be absent? How is control frequency represented? Padding a missing sensor with zeros is an implementation choice that the model can mistake for physical evidence.

The most useful dataset table is not trajectory count. It reports task entropy, scene entropy, embodiment coverage, operator coverage, success and recovery rates, control frequencies, action normalization, missing modalities, and effective unique windows after temporal overlap. Those quantities tell you what generalization the corpus can plausibly support.

## Representation is the first scaling law

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

## Decide where transfer is allowed

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

## One model can use several losses

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

![How a multimodal data mixture becomes a gradient budget](/assets/images/multimodal-pretraining-gradient-budget.svg)
_Example share is only the first accounting layer. Sequence expansion, loss reduction, compute, and gradient geometry determine which modality actually moves shared parameters._

This distinction becomes acute in robotics. Overlapping windows from one trajectory can produce thousands of training examples without thousands of independent decisions. A long action chunk can contribute many supervised dimensions while representing one correlated maneuver. The cleanest dashboard keeps four ledgers side by side: sampled examples, predicted units, consumed FLOPs, and update norm by module. A mixture is balanced only relative to a capability objective, not because its percentages sum to 100.

## A data mixture is a policy over scarce compute

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

## Video is not automatically a world model

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

## The action distribution is an architectural choice

Actions change the data distribution, carry embodiment-specific units, and operate under a control deadline. Treating them as ordinary tokens hides those constraints behind a convenient interface.

[RT-2](/paper%20shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html) shows the appeal of treating actions as language tokens: one decoder, tractable autoregression, and direct transfer from semantic pretraining. [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html) shows why naive binning breaks on high-frequency dexterity and compresses action chunks with a discrete cosine transform. [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html) uses a continuous flow expert instead of forcing actions through the text head. [Pi0.5](/paper%20shorts/2025/04/22/pi0-5-vision-language-action-model-with-open-world-generalization.html) adds high-level semantic subtask prediction and heterogeneous co-training, making the boundary between “what next” and “how” explicit.

The action-interface memo should compare:

| Choice | Likelihood | Multimodality | Latency | Main failure |
| --- | --- | --- | --- | --- |
| Direct regression | Explicit/simple | Weak | Low | Mode averaging |
| Discrete tokens | Exact autoregressive | Moderate | Sequential | Quantization and slow chunks |
| Frequency tokens | Exact autoregressive | Moderate | Shorter sequence | Loses abrupt corrections |
| Diffusion/flow | Implicit or path-based | Strong | Iterative/parallel | Complex post-training likelihood |
| Separate action expert | Interface-dependent | Strong | Extra serving path | Semantic/control coordination |

The pretraining question is which action prior should be learned across robots. The answer cannot be read from imitation loss alone. Reconstruct a short trajectory with each representation, compare spectral and contact-heavy errors, measure effective horizon and wall-clock control rate, and test whether a small target dataset can change the embodiment without erasing semantic transfer. The post-training question comes later: does the chosen distribution expose the likelihoods and responsiveness required by the improvement loop?

## Scaling evidence needs a falsification boundary

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

## A large run is a fault-tolerant experiment

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

## The minimum convincing proxy program

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

## A reading course with three routes

There is no single best order after the foundations. Choose a route based on the decision you need to make, and produce an artifact after each layer.

**Foundation route: where do semantics come from?** Read [ViT](/paper%20shorts/2020/10/01/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale.html), [CLIP](/paper%20shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html), [SigLIP](/paper%20shorts/2023/10/01/sigmoid-loss-for-language-image-pre-training-siglip.html), [LLaVA](/paper%20shorts/2023/04/01/visual-instruction-tuning-llava.html), and [MM1](/paper%20shorts/2024/03/14/mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training.html). Produce an information-flow diagram that marks the training unit, visual-token budget, frozen modules, and datasets responsible for alignment versus behavior.

**Unification route: what should share parameters?** Read [Chameleon](/paper%20shorts/2024/05/16/chameleon-mixed-modal-early-fusion-foundation-models.html), [Transfusion](/paper%20shorts/2024/08/20/transfusion-predict-the-next-token-and-diffuse-images-with-one-multimodal-model.html), [Janus](/paper%20shorts/2024/10/17/janus-decoupling-visual-encoding-for-unified-multimodal-understanding-and-generation.html), [TokenFlow](/paper%20shorts/2024/12/04/tokenflow-unified-image-tokenizer-for-multimodal-understanding-and-generation.html), and [Native Multimodal Scaling Laws](/paper%20shorts/2025/04/10/scaling-laws-for-native-multimodal-models.html). Build a matrix of shared tokenizers, encoders, trunk parameters, experts, objectives, and output heads. Then specify the matched-budget experiment that would justify each split.

**Robot-policy route: what motor prior transfers?** Read [Open X-Embodiment](/paper%20shorts/2023/10/13/open-x-embodiment-robotic-learning-datasets-and-rt-x-models.html), [Octo](/paper%20shorts/2024/05/20/octo-an-open-source-generalist-robot-policy.html), [OpenVLA](/paper%20shorts/2024/06/01/openvla-open-source-vision-language-action-model.html), [DROID](https://arxiv.org/abs/2403.12945), [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html), [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html), and [Pi0.5](/paper%20shorts/2025/04/22/pi0-5-vision-language-action-model-with-open-world-generalization.html). Produce two tables: one for dataset variation and one for the action interface. If a paper omits control rate, normalization, or target-data size, write “not reported” rather than smoothing over the gap.

**Scaling route: which result deserves the large run?** Read [Generative Mixed-Modal Scaling](/paper%20shorts/2023/01/10/scaling-laws-for-generative-mixed-modal-language-models.html), [Optimal Data Mixtures](/paper%20shorts/2025/07/12/scaling-laws-for-optimal-data-mixtures.html), and [TorchTitan](/paper%20shorts/2024/10/09/torchtitan-one-stop-pytorch-native-solution-for-production-ready-llm-pre-training.html). Reconstruct the measured range, extrapolated range, confidence interval, throughput assumption, and kill criterion. The deliverable is a go/no-go memo for one target-scale run.

Then move to [Part III: Post-Training VLAs](/blog/2026/07/16/post-training-vision-language-action-models-zero-to-hero.html). Pretraining decides what the policy can represent and which behaviors are nearby. Deployment reveals which nearby behaviors are actually useful.

## The research thesis

The strongest multimodal robot model will probably not make every modality identical. It will share the parameters that benefit from transfer and specialize the interfaces where fidelity, time, geometry, and control impose different requirements.

My preferred starting hypothesis is a shared transformer with modality-specific input/output experts, explicit per-objective accounting, and a data mixture chosen by proxy scaling rather than intuition. For physical intelligence, I would preserve metric, temporally persistent entity representations alongside open-vocabulary semantics. The generative model can imagine; the world model must preserve consequences; the policy must act; and the training system must reveal when one capability is improving at another's expense.

The practical standard is demanding but clear: a literature review should end as an experiment plan. It should say what one training unit is, where information is compressed, how data becomes gradient, what transfer is expected, which matched control is missing, and what result would reverse the architecture decision. Anything less is a tour of papers, not a pretraining strategy.

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
- [Pi0.5: A Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054)
- [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864)
- [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset](https://arxiv.org/abs/2403.12945)
- [TorchTitan: One-stop PyTorch Native Solution for Production Ready LLM Pre-training](https://arxiv.org/abs/2410.06511)
