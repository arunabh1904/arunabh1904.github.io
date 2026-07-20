---
title: 'From Seeing to Acting: A Reading Guide to Vision-Language Models'
date: '2026-07-05T20:00:00.000Z'
section: blog
postSlug: from-seeing-to-doing-the-evolution-of-vision-language-models
legacyPath: /blog/2026/07/05/from-seeing-to-doing-the-evolution-of-vision-language-models.html
tags:
  - Research
  - Vision-Language Models
summary: A reading guide to the interfaces that turned image-text models into grounded visual assistants and, eventually, robot policies.
---

# From Seeing to Acting: A Reading Guide to Vision-Language Models

A model can name the mug, explain that mugs hold liquid, and still drive a gripper into the table beside it. That failure is not surprising once we stop treating “vision-language” as a single capability. Naming an object, grounding it in pixels, estimating its pose, predicting contact, and controlling a wrist are different contracts with different tolerances for lost information.

The standard history—CLIP, visual chat, video, driving, robotics—makes the field look like a sequence of bigger demonstrations. The more useful history follows what the visual representation became responsible for. CLIP needed an image to land near the right sentence. LLaVA needed visual features to steer a decoder for hundreds of tokens. A grounded assistant must retain which pixels justify which words. A robot policy must preserve enough state, geometry, and time to act before its observation becomes stale.

Each step inherits the previous one, but none is a free upgrade. Global semantic alignment can discard location. Fluent generation can hide weak eyesight. More video frames can consume context without teaching causality. Language can specify a goal while saying nothing about control frequency. The point of this guide is to make those fault lines visible before the papers blur them together.

![Reading map from VLMs to deployed robot policies](/assets/images/multimodal-vla-reading-map.svg)
_The three-part series follows four contracts: visual evidence, robot experience, an action distribution, and closed-loop feedback. The question under each block is the one worth carrying into the papers._

This is Part I of a three-part reading course. [Part II](/blog/2026/07/15/omni-model-pretraining-decisions.html) asks how multimodal and robot data should shape a pretrained policy. [Part III](/blog/2026/07/16/post-training-vision-language-action-models-zero-to-hero.html) asks how deployment failures can justify a policy update. You can read this article alone, but the sequence matters: post-training cannot recover visual evidence that pretraining threw away, and pretraining cannot anticipate every state the deployed policy will create.

![Vision-language model stack](/assets/images/vlm-stack-schematic.svg)
_The common stack hides several different contracts. Retrieval aligns representations; visual chat conditions a decoder; embodied systems must also preserve state, time, and action consequences._

The scope is deliberately narrow. I follow five transitions: labels to language, alignment to generation, coarse semantics to grounded evidence, answers to decisions, and decisions to actions. I care about the architectural and data choices that survive model generations, not a complete leaderboard. Reported results belong to the cited papers; the interface taxonomy and reading order are my synthesis.

## How to use the guide

Do not read every paper with the same question. For alignment papers, identify the unit of comparison. For visual-assistant papers, trace where spatial detail can disappear. For video papers, ask whether time is merely compressed or actually modeled. For driving and robotics papers, write down the action interface, control deadline, and source of closed-loop evidence.

The recurring exercise is simple: draw the path from raw observation to evaluated output, then circle every irreversible compression step. That picture usually explains more than the model name.

## 1. A VLM is an interface, not one architecture

“Vision-language model” covers several systems with different outputs:

| System contract | Typical output | What vision must preserve | Failure that matters |
| --- | --- | --- | --- |
| Image-text retrieval | similarity score | global semantic identity | wrong neighborhood in embedding space |
| Visual assistant | text tokens | evidence needed across a conversation | fluent answer unsupported by the image |
| Grounded perceiver | boxes, points, regions, text | identity plus location and detail | right noun, wrong object or position |
| Decision support | rationale, plan, trajectory proposal | state, rules, geometry, uncertainty | plausible explanation for a bad decision |
| Embodied policy | action chunk or control distribution | state transitions and embodiment | compounding closed-loop error |

This table is the first defense against vague claims. A retrieval model can be excellent without being able to generate text. A visual assistant can describe a scene while failing to localize a small object. A model can localize an object yet lack the metric precision or control rate required to manipulate it.

The shared implementation pattern is simple. An image is converted into visual tokens or features. A connector maps those features into a space that can interact with text. A loss then decides what “interaction” means: contrastive agreement, next-token prediction, region grounding, denoising, or action imitation.

The loss is the contract. Architecture matters because it determines what evidence remains available to satisfy that contract.

## 2. Language replaced the fixed label set

Before large-scale image-text pretraining, a vision classifier typically learned a closed vocabulary. Its final layer represented the categories chosen by the dataset designer. Adding a new class meant collecting labels and training again.

[CLIP](/paper%20shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html) changed the interface. It trains an image encoder $f_I$ and text encoder $f_T$ so matched image-caption pairs receive high similarity and mismatched pairs receive lower similarity. In simplified form,

$$
s_{ij}=\frac{f_I(I_i)^\top f_T(T_j)}{\tau},
$$

where $\tau$ controls the sharpness of the similarities. At inference, class names become text prompts. Classification is retrieval against language rather than a fixed learned head.

The important result was not merely zero-shot ImageNet accuracy. CLIP showed that noisy natural language at web scale can define a broad visual supervision space. A prompt can name a class, attribute, style, or relation that never appeared as a dedicated label in the training interface.

The tradeoff sits inside the objective. Batch-softmax contrastive learning treats the other examples in a batch as negatives. Large and diverse batches improve that comparison set, but they demand synchronization and can introduce false negatives when two captions describe compatible content.

[SigLIP](/paper%20shorts/2023/10/01/sigmoid-loss-for-language-image-pre-training-siglip.html) changes the pairwise objective. Each image-text pair receives an independent positive or negative sigmoid loss rather than participating in a single batch-wide softmax. That reduces the objective's dependence on enormous synchronized negative sets. The broader lesson is useful beyond SigLIP: **an alignment loss determines the unit of competition**. If the unit is the batch, distributed systems and batch composition become part of the model design.

What retrieval pretraining does not provide is equally important. A shared embedding can tell us that an image and sentence belong together without preserving which patch supports which phrase, how objects relate spatially, or how to generate a multi-sentence answer. Global semantic alignment is a strong prior, not a complete visual interface.

## 3. Visual chat turned alignment into conditional generation

[LLaVA](/paper%20shorts/2023/04/01/visual-instruction-tuning-llava.html) made the next transition legible. Start with a pretrained visual encoder and an instruction-tuned language model. Learn a projector that maps visual features into the language model's token space. Then fine-tune on image-instruction-response examples.

The minimal architecture looks almost underwhelming:

$$
H_v=W f_I(I), \qquad
p(y\mid I,x)=\prod_t p(y_t\mid H_v,x,y_{<t}).
$$

The projector $W$ is small relative to the encoders around it. Yet the system feels qualitatively different because the output contract changed. The model must use visual evidence while maintaining the conversational behavior already learned by the language model.

This is a recurring pattern in foundation models: a modest interface plus the right post-training data can elicit a capability that the base components nearly support. It is tempting to conclude that the connector is the central research problem. Controlled studies in [MM1](/paper%20shorts/2024/03/14/mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training.html) point elsewhere. In its studied recipe, image encoder quality, resolution, number of visual tokens, and the composition of interleaved, caption, and text-only data matter more than endlessly elaborating the connector.

That result should change experiment allocation. Sweep the variables that determine what evidence enters the language model before polishing the bridge that carries it.

### Instruction tuning teaches behavior, not eyesight

Visual instruction data solve a behavioral problem: how should the model respond when an image and request arrive together? They do not automatically repair information discarded by the visual encoder or compression scheme.

If a document image is reduced to too few tokens, no amount of eloquent instruction tuning can recover the missing characters. If the pretraining data rarely require spatial reference, the model can learn the statistical shape of grounded answers without consistently binding phrases to regions. If captions name only salient foreground objects, the model learns that background detail is often safe to ignore.

This distinction explains why modern VLM recipes separate several data roles:

- image-text pairs establish broad semantic alignment;
- interleaved documents teach relationships across images and surrounding text;
- OCR, chart, document, and grounding data force fine-detail retention;
- instruction data teach interaction format and task following;
- preference or rejection data shape helpfulness, calibration, and refusal behavior;
- video data add temporal evidence and compression constraints.

[Eagle 2](/paper%20shorts/2025/01/01/eagle-2-post-training-data-strategies-for-frontier-vision-language-models.html) is valuable for this reason. Its central object is not a novel visual encoder; it is the post-training data strategy. The paper treats task balance, data quality, and staged training as first-class determinants of VLM behavior. A smaller model with a deliberate curriculum can compete with a larger model trained on a poorly organized mixture.

The operational question is no longer “How many multimodal examples do we have?” It is “Which behavior does each dataset teach, and which capability regresses when we increase its weight?”

## 4. Visual tokens became the scarce resource

Text tokenization compresses language into a sequence of discrete symbols. Images do not arrive with an obvious equivalent. A $1024\times1024$ image can be divided into patches, encoded into a smaller feature grid, tiled at several resolutions, or compressed through a learned tokenizer. Every choice trades detail for sequence length.

The total context is roughly

$$
T=T_{text}+T_{image}+T_{video}.
$$

That simple sum drives real costs. More visual tokens preserve small objects and text but consume attention, memory, and latency. Fewer tokens improve throughput but may create an irreversible perceptual bottleneck.

[PaliGemma](/paper%20shorts/2024/07/10/paligemma-a-versatile-3b-vlm-for-transfer.html) is a clean study of transfer: a SigLIP vision encoder feeds a compact Gemma language model, and resolution upcycling lets later stages pay more visual compute where tasks need it. [Qwen2-VL](/paper%20shorts/2024/09/01/qwen2-vl-enhancing-vision-language-model-perception-of-the-world-at-any-resolution.html) makes token count depend on input resolution and adapts positional treatment for images and video. [DeepSeek-VL2](/paper%20shorts/2024/12/01/deepseek-vl2-mixture-of-experts-vision-language-models.html) combines dynamic tiling with sparse language capacity and a more economical attention design.

These systems differ in implementation, but they answer one shared question: **where should variable visual complexity be paid for?** A fixed low-resolution budget is predictable and blunt. Dynamic resolution preserves detail but makes serving cost input-dependent. Tiling retains local evidence but can obscure global layout or duplicate boundary content. Token compression improves throughput but shifts risk into the compressor.

A convincing comparison must therefore match more than parameter count. It should report visual tokens, input resolution, training and inference FLOPs, latency, and performance on tasks that isolate small text, counting, localization, and global scene reasoning. Otherwise, a “better architecture” may simply be buying more pixels.

## 5. Grounding is the bridge from words to evidence

A model can answer “the traffic light is red” for at least three reasons: it localized the light and read its state, it used a language prior about the scene, or it guessed from dataset regularities. Standard answer accuracy often fails to distinguish them.

Grounding asks the model to bind language to a region, point, track, or spatial relation. [LocCa](/paper%20shorts/2024/03/28/locca-visual-pretraining-with-location-aware-captioners.html) pushes location information into caption-style pretraining. [Cambrian-1](/paper%20shorts/2024/06/01/cambrian-1-vision-centric-exploration-of-multimodal-llms.html) treats visual representation and connector choices as controlled variables rather than assuming a stronger language model will absorb every visual weakness. [Molmo](/paper%20shorts/2024/09/01/molmo-and-pixmo-open-weights-and-open-data-for-state-of-the-art-vision-language-models.html) makes point-based grounding and openly described high-quality data central to the model's capabilities.

The deeper lesson is about supervision granularity. Caption loss can be satisfied by global semantics. Region captions require spatial binding. Pointing data make the binding externally inspectable. Tracking and temporal localization ask whether the binding persists as the scene changes.

This gives a useful hierarchy for evaluating visual evidence:

1. **Recognition:** is the concept present?
2. **Localization:** which pixels or region support the answer?
3. **Relation:** how do the relevant entities interact spatially?
4. **Persistence:** are they tracked consistently over time?
5. **Counterfactual dependence:** does the answer change when the relevant evidence changes?

The fifth level is the hardest and most revealing. If the answer remains unchanged after the traffic light is masked or its state is edited, the model's explanation was not causally grounded in that evidence.

## 6. Video adds time, but not automatically causality

Video looks like “more images,” yet it changes the representation problem. Adjacent frames are highly redundant, important events can be brief, and the correct sampling rate depends on the question. Uniformly encoding every frame wastes context; aggressive sampling can delete the event.

[VideoLLaMA 3](/paper%20shorts/2025/01/01/videollama-3-frontier-multimodal-foundation-models.html) builds from image-text alignment and reduces redundant temporal tokens. This is a sensible systems choice, but compression must be evaluated against event duration. A tokenizer that works for slow scene description may miss a pedestrian entering the road or the instant a gripper loses contact.

Video understanding also invites an overclaim: that predicting or describing video means the model has learned a world model. Temporal coherence is not enough. A useful world model must preserve state and respond consistently to interventions. If two different actions lead to the same plausible continuation, the model may understand video statistics without representing controllable dynamics.

For video VLMs, I would separate four tests:

- event localization: when did the relevant change occur?
- state persistence: what remained true across frames?
- causal ordering: which event enabled or prevented the next?
- intervention consistency: does changing an action or state change the predicted consequence?

The first three measure temporal understanding. The fourth begins to test a model of action-conditioned worlds.

## 7. Driving reveals the gap between an answer and a decision

Autonomous driving compresses nearly every VLM weakness into one domain: small distant objects, geometry, rules, rare hazards, temporal prediction, uncertainty, and hard latency limits.

The literature explores several placements for language. [GPT-Driver](/paper%20shorts/2023/10/01/gpt-driver-learning-to-drive-with-gpt.html) and [Driving with LLMs](/paper%20shorts/2023/10/01/driving-with-llms-fusing-object-level-vector-modality.html) represent driving state in a form a language model can reason over. [DriveVLM](/paper%20shorts/2024/02/01/drivevlm-convergence-of-autonomous-driving-and-large-vision-language-models.html) uses chain-of-thought-style scene reasoning alongside conventional planning. [AsyncDriver](/paper%20shorts/2024/06/01/asyncdriver-asynchronous-llm-enhanced-planner-for-autonomous-driving.html) addresses the latency mismatch by decoupling the slower language-model path from faster planning. [VLM-AD](/paper%20shorts/2024/12/19/vlm-ad-end-to-end-autonomous-driving-through-vision-language-model-supervision.html) uses VLM supervision to improve a deployable driving policy rather than placing the full VLM in the control loop.

These designs should not be collapsed into “VLMs for driving.” They correspond to different system bets:

| Placement of language | Main benefit | Primary risk |
| --- | --- | --- |
| Online planner | semantic flexibility and explicit reasoning | latency and geometric imprecision |
| High-level route or behavior selector | separates semantics from low-level control | interface errors between levels |
| Auxiliary explanation head | inspectable training signal | rationale may not cause the action |
| Offline teacher or data labeler | rich supervision without online cost | teacher errors become training targets |
| Unified end-to-end policy | fewer hand-built interfaces | difficult attribution and safety validation |

My current read is that the strongest near-term use is hybrid. VLMs contribute semantic knowledge, intent understanding, rare-scenario interpretation, and supervision. Metric perception, motion forecasting, safety constraints, and high-rate control retain explicit structure. This is not an argument that end-to-end systems cannot work. It is a statement about evidence: fluent rationales and open-loop trajectory metrics do not yet establish reliable closed-loop control.

## 8. Benchmarks are testing whether the model looked

[DriveBench](/paper%20shorts/2025/01/01/are-vlms-ready-for-autonomous-driving-drivebench.html), [IDKB](/paper%20shorts/2024/09/01/can-lvlms-obtain-a-drivers-license-idkb.html), [TOD3Cap](/paper%20shorts/2024/03/01/tod3cap-towards-3d-dense-captioning-in-outdoor-scenes.html), and [AutoTrust](/paper%20shorts/2024/12/01/autotrust-benchmarking-trustworthiness-in-large-vision-language-models-for-autonomous-driving.html) attack different versions of the same problem: plausible language can hide weak evidence use.

The benchmark question should be phrased as a causal audit:

- Does performance fall when the relevant region is corrupted?
- Does the answer follow a changed sign, light state, object position, or instruction?
- Can the model distinguish “not visible” from “not present”?
- Are traffic-rule answers consistent across paraphrases and scene variations?
- Does confidence track the severity of distribution shift?
- Does the rationale identify evidence that actually changes the decision?

A benchmark that tests only final-answer agreement can reward memorized priors. Corruption tests, counterfactual edits, and evidence localization make it harder to pass without looking.

The practical implication is uncomfortable: a higher aggregate VLM score may be less valuable than a lower score with better calibration and causal grounding. Deployment cares about the shape of failure, not only its average frequency.

## 9. Robotics changes the output space

Robotics completes the transition from description to intervention. Once the model emits actions, its outputs change the next observation. Errors compound under the state distribution created by the policy itself.

[OpenVLA](/paper%20shorts/2024/06/01/openvla-open-source-vision-language-action-model.html) adapts a VLM backbone into an open robot policy. [RT-2](/paper%20shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html) expresses actions as tokens, making semantic and control outputs share an autoregressive interface. [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html) compresses continuous action chunks in the frequency domain to shorten those sequences. [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html) instead keeps a semantic VLM trunk and uses a continuous flow-based action expert. [DexVLA](/paper%20shorts/2025/02/01/dexvla-vision-language-model-with-plug-in-diffusion-expert.html) similarly gives dexterous control a specialized generative route.

The action representation is not an output-format detail. It defines what the policy can express and how it can be trained:

| Action interface | Advantage | Cost |
| --- | --- | --- |
| Per-step regression | simple and fast | averages multimodal actions |
| Discrete tokens | exact autoregressive likelihood | quantization and sequential latency |
| Compressed action tokens | shorter horizon | may remove abrupt corrections |
| Diffusion or flow chunks | expressive continuous distributions | iterative sampling and harder RL likelihoods |
| Separate action expert | specialization without discarding VLM semantics | extra parameters and coordination path |

A caption has no control frequency. An action does. A robot policy must fit sensing, inference, communication, and actuation inside a deadline. It must also decide how long an action chunk remains valid before new evidence should interrupt it. Longer chunks reduce inference calls and improve temporal coherence; shorter chunks respond faster to disturbances.

This is where “language as a universal interface” reaches its limit. Language can carry task semantics. It does not make units, embodiment, contact dynamics, or control latency disappear.

## 10. How to read a VLM paper

I use six questions to avoid being carried away by a capability collage.

1. **What is the output contract?** Retrieval score, text, location, video, trajectory, or action?
2. **What is one training unit?** Pair, interleaved sequence, region, frame, clip, or action chunk?
3. **Where can visual information be lost?** Resolution, crop, tokenizer, connector, context compression, or temporal sampling?
4. **Which component actually changed?** Encoder, connector, language model, data, objective, post-training, or decoding?
5. **What matched control supports the claim?** Same data, tokens, parameters, compute, and evaluation protocol?
6. **Does the evaluation require the model to use the claimed evidence?** Or can language priors solve the benchmark?

These questions turn a model paper into a decision record. If the answer to question four is “several things,” then the paper may demonstrate a strong recipe without identifying why it works. That is still useful, but it supports adoption more than causal understanding.

## 11. A reading course, not a bibliography

The list below is ordered by conceptual dependency. Each layer has a deliverable; without it, “reading the paper” too easily becomes collecting model names.

**Layer 1: alignment.** Read [CLIP](/paper%20shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html) and [SigLIP](/paper%20shorts/2023/10/01/sigmoid-loss-for-language-image-pre-training-siglip.html). Derive the two losses and write down the unit of competition. Your deliverable is a one-page note explaining how batch composition becomes part of the learning algorithm.

**Layer 2: conditional generation.** Read [LLaVA](/paper%20shorts/2023/04/01/visual-instruction-tuning-llava.html) and [MM1](/paper%20shorts/2024/03/14/mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training.html). Build an ablation table with four columns: visual encoder, connector, pretraining mixture, and instruction data. The exercise forces you to separate a good recipe from a causal result.

**Layer 3: visual evidence.** Read [LocCa](/paper%20shorts/2024/03/28/locca-visual-pretraining-with-location-aware-captioners.html), [Cambrian-1](/paper%20shorts/2024/06/01/cambrian-1-vision-centric-exploration-of-multimodal-llms.html), [PaliGemma](/paper%20shorts/2024/07/10/paligemma-a-versatile-3b-vlm-for-transfer.html), [Qwen2-VL](/paper%20shorts/2024/09/01/qwen2-vl-enhancing-vision-language-model-perception-of-the-world-at-any-resolution.html), and [Molmo](/paper%20shorts/2024/09/01/molmo-and-pixmo-open-weights-and-open-data-for-state-of-the-art-vision-language-models.html). For one image, estimate the visual-token budget under each recipe. Then mark which datasets require pointing, OCR, relations, or only a plausible caption.

**Layer 4: temporal and safety-critical use.** Read [VideoLLaMA 3](/paper%20shorts/2025/01/01/videollama-3-frontier-multimodal-foundation-models.html), [DriveVLM](/paper%20shorts/2024/02/01/drivevlm-convergence-of-autonomous-driving-and-large-vision-language-models.html), [VLM-AD](/paper%20shorts/2024/12/19/vlm-ad-end-to-end-autonomous-driving-through-vision-language-model-supervision.html), [DriveBench](/paper%20shorts/2025/01/01/are-vlms-ready-for-autonomous-driving-drivebench.html), and [AutoTrust](/paper%20shorts/2024/12/01/autotrust-benchmarking-trustworthiness-in-large-vision-language-models-for-autonomous-driving.html). Design one counterfactual edit that should flip a decision and one irrelevant edit that should not. If the benchmark cannot express that test, note the gap.

**Layer 5: action.** Read [RT-2](/paper%20shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html), [OpenVLA](/paper%20shorts/2024/06/01/openvla-open-source-vision-language-action-model.html), [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html), and [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html). Reconstruct the action distribution, horizon, control rate, and inference path. The deliverable is a latency budget, not another summary.

After these five layers, move to [Part II: Pretraining Multimodal Models for Robotics](/blog/2026/07/15/omni-model-pretraining-decisions.html). The VLM literature tells you what visual and semantic priors are available. The robotics literature asks which of those priors survive contact with embodiment.

## The research thesis

VLM progress is a sequence of interface contracts. Contrastive learning made images addressable through language. Visual instruction tuning made that representation conversational. Grounding tried to reconnect fluent words to visible evidence. Video introduced persistence and compression. Driving and robotics exposed every shortcut because a plausible answer can become a bad physical decision.

My strongest bet is not that one token stream will erase every modality boundary. It is that semantic reasoning will become a shared layer, while high-bandwidth perception and high-rate control retain specialized representations and losses. The decisive systems will know what to share and what not to compress.

My bet is a shared semantic layer with explicit high-bandwidth routes for geometry, time, and control. A fully unified token stream should replace that hybrid only when, under matched data, parameters, tokens, compute, and latency, it wins on fine grounding, temporal counterfactuals, and closed-loop recovery without losing calibration. Until then, “one model for everything” names an experiment. It does not settle the architecture.

## References

- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [Sigmoid Loss for Language Image Pre-Training (SigLIP)](https://arxiv.org/abs/2303.15343)
- [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485)
- [MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](https://arxiv.org/abs/2403.09611)
- [PaliGemma: A Versatile 3B VLM for Transfer](https://arxiv.org/abs/2407.07726)
- [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)
- [Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](https://arxiv.org/abs/2409.17146)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [Pi0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)
