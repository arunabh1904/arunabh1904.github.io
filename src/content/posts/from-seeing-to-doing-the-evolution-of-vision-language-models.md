---
title: 'Tracing the VLM Progression'
date: '2026-07-05T20:00:00.000Z'
section: blog
blogGroup: research-guides
postSlug: from-seeing-to-doing-the-evolution-of-vision-language-models
legacyPath: /blog/2026/07/05/from-seeing-to-doing-the-evolution-of-vision-language-models.html
tags:
  - Research
  - Vision-Language Models
summary: A technical history of how vision-language models moved from task-specific fusion and image-text alignment to visual assistants, grounded reasoning, video, and robot action.
---
# Tracing the VLM Progression

Vision-language models have become incredibly popular over the last few years. Rightfully so! Grounding language in images or video is a huge generalization unlock. It has pushed one model family across classification, detection, alignment, generation, grounding, temporal reasoning, and finally action.

VLMs really started to click with CLIP's image-text alignment, moved through LLaVA's visual instruction tuning, and continued into video understanding and robotics. That arc is useful but incomplete. CLIP did not begin vision-language learning; earlier models already fused detected regions with words. Its real shift was scale: natural language became an open vocabulary for classification. Visual chat required a second bridge, connecting pretrained vision encoders to language models before instruction tuning turned them into assistants.

A better way to understand this progression is to ask what the representation needs to preserve. We start by connecting detected regions to words, then move through image-text alignment, conditional generation, and instruction tuning. Grounding and video add location, detail, and time. Decision and action add geometry, control timing, and consequences. Each step unlocks a new output, but it can also throw away information the next step needs. Fluent generation can hide weak visual evidence, longer context does not guarantee temporal understanding, and reasoning cannot recover pixels the encoder never preserved. Let's unpack how these capabilities came together.

## Image-text alignment

CLIP is the natural place to start because it made image-text alignment work at internet scale. But it was not the first model to connect images and words.

Earlier models started with regions from an object detector and made those regions interact with words. [ViLBERT](/paper%20shorts/2019/08/06/vilbert-pretraining-task-agnostic-visiolinguistic-representations.html) kept vision and language in separate streams, then connected them through co-attention. [LXMERT](/paper%20shorts/2019/08/20/lxmert-learning-cross-modality-encoder-representations.html) used separate object, language, and cross-modal encoders. [UNITER](/paper%20shorts/2019/09/25/uniter-universal-image-text-representation-learning.html) put both modalities in one transformer and trained several alignment objectives together. These models could already answer visual questions, retrieve images, resolve referring expressions, and reason over detected objects.

The recipe was powerful but heavy. The detector decided which regions reached the model, so a missed proposal was gone before language ever saw the image. Every image-text pair then had to run through cross-modal attention. That worked for curated tasks, but it was a poor way to search across hundreds of millions of images and captions.

CLIP removed the expensive fusion step by encoding images and text independently, then comparing the resulting vectors. Retrieval became a nearest-neighbor lookup instead of a fresh transformer pass for every image-text pair. The same design reduced the pressure to preserve spatial correspondence because the loss only required the correct caption to match the correct image.

CLIP's success came from its simplicity. Contrast images against the text found alongside them on the internet, then build a similarity index of sorts in a shared vector space. This completely sidesteps a fixed class ontology. Adding a concept no longer means collecting labels and retraining a classifier head. Do it across a large enough corpus and batch, and voilà: classification becomes retrieval against language.

[CLIP](/paper%20shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html) trained this recipe on 400 million image-text pairs. In each batch, an image encoder $f_I$ and a text encoder $f_T$ map both sides into the shared space. The contrastive loss pulls matched pairs together and pushes mismatched pairs apart. The score for image $i$ and text $j$ is their normalized embedding dot product divided by temperature $\tau$:

$$
s_{ij}=\frac{f_I(I_i)^\top f_T(T_j)}{\tau},
$$

Temperature $\tau$ controls how sharply the model separates the similarities. At inference, class names are written as prompts and embedded by the text encoder. The predicted class is the prompt with the highest similarity to the image.

![Figure 1 from the CLIP paper, showing contrastive pretraining and zero-shot transfer through text prompts](/assets/images/clip-paper-figure-1-contrastive-pretraining.png)
*CLIP aligns image and text encoders during pretraining, then replaces the fixed classifier head with written class prompts. source: [Learning Transferable Visual Models From Natural Language Supervision](/paper%20shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html)*

CLIP's loss still couples every example in the batch. Each image competes against every caption, and every unmatched pair is treated as a negative. Larger batches give the model harder comparisons, but they also require heavy synchronization across devices. They can even create false negatives when two captions describe compatible images.

[SigLIP](/paper%20shorts/2023/10/01/sigmoid-loss-for-language-image-pre-training-siglip.html) replaced CLIP's batch-wide softmax with an independent sigmoid loss for every image-text pair. Positive pairs receive a positive label and the remaining pairs receive a negative label. Removing the shared denominator reduces the dependence on batch size and cross-device synchronization, which makes the objective easier to scale across distributed systems.

The sigmoid objective still rewards global image-text alignment. An image and caption can match even when the encoder ignores a small object or its exact location. [SigLIP 2](/paper%20shorts/2025/02/20/siglip-2-multilingual-vision-language-encoders.html) keeps the dual-encoder architecture and adds captioning, self-supervision, multilingual data, curation, and multi-resolution training. These additions improve the features used for localization and dense visual tasks while retaining the same image-text retrieval interface.

The open vocabulary learned through image-text alignment also transferred into detection. [Detic](/paper%20shorts/2022/01/07/detic-detecting-twenty-thousand-classes-using-image-level-supervision.html) expanded a detector with image-level labels, including classes without box annotations. [OWL-ViT](/paper%20shorts/2022/05/12/owl-vit-simple-open-vocabulary-object-detection-with-vision-transformers.html) started from contrastive image-text pretraining and fine-tuned a Vision Transformer for text-conditioned detection. Both models extend language-conditioned recognition from image-level classification to localization.

Localization still does not provide a generative interface. A shared embedding can measure whether an image and sentence belong together, while a detector can place a box around a named object. Neither produces a multi-sentence description or answers a follow-up question. Those capabilities required a language decoder conditioned on visual features.

## Connecting vision to a language generator

A pretrained language model already contained most of the machinery needed for text generation. The remaining problem was to map visual features into its input space without retraining the full vision-language stack.

The first approach was to keep the language model frozen. [Multimodal Few-Shot Learning with Frozen Language Models](/paper%20shorts/2021/06/25/multimodal-few-shot-learning-with-frozen-language-models.html) learned a visual prefix that the decoder could treat like extra context. [Flamingo](/paper%20shorts/2022/04/29/flamingo-visual-language-model-for-few-shot-learning.html) built a stronger bridge. A Perceiver Resampler compresses an image or video into 64 visual tokens, and gated cross-attention lets the frozen language model read them between its existing blocks. The gates start near zero, so training begins with the original language model almost unchanged.

[BLIP](/paper%20shorts/2022/01/28/blip-bootstrapping-language-image-pretraining.html) trained more of the path and cleaned the data at the same time. The same encoder-decoder handles image-text contrast, image-text matching, and caption generation under different attention masks. Its CapFilt pipeline generates better captions for noisy web images and filters weak pairs. Captioning now does two jobs: it trains the model to generate text and improves the data used for the next round of training.

![BLIP's shared encoder-decoder and three pretraining objectives](/assets/images/blip-paper-figure-2.png)
*BLIP uses the same core model in three modes. Contrastive loss aligns image and text, matching loss learns pairwise interaction, and language-modeling loss trains the decoder to turn visual evidence into words. source: [BLIP](/paper%20shorts/2022/01/28/blip-bootstrapping-language-image-pretraining.html)*

These models differ mainly in where they spend the training. Flamingo keeps both large endpoints frozen and learns the bridge. BLIP lets the multimodal path adapt. [PaLI](/paper%20shorts/2022/09/14/pali-jointly-scaled-multilingual-language-image-model.html) trains the visual encoder and language encoder-decoder across many tasks and languages. Underneath the model names are three choices: turn the image into a short prefix, expose it through cross-attention, or train vision and language together.

### From Q-Formers to MLP projectors

Connecting a vision encoder to a language model requires two decisions. The connector has to map visual features into the language model's embedding space, and it may also reduce the number of visual tokens. Q-Formers learned both operations together. Later projectors separated them and showed that the mapping itself could remain simple.

[BLIP-2](/paper%20shorts/2023/01/30/blip-2-bootstrapping-language-image-pretraining.html) introduced the Querying Transformer, or Q-Former, to connect a frozen image encoder with a frozen language model. Its 32 learned queries cross-attend to the image patches and produce 32 visual tokens regardless of the input resolution. The Q-Former is therefore a connector and a learned compression module.

$$
Z=\operatorname{QFormer}(Q,f_I(I),T), \qquad |Q|=32.
$$

BLIP-2 trains the Q-Former with three objectives. Image-text contrast aligns the query outputs with text. Image-text matching allows full interaction and predicts whether an image-text pair is genuine. Image-grounded generation lets the text attend to the visual queries and previous words while masking future words. The Q-Former weights are shared across all three objectives, while the attention mask determines how image and text tokens interact.

The second stage maps the query outputs into the language model's embedding space through a linear projection. The projected queries act as soft visual prompt tokens, while the language model remains frozen. This design made it possible to reuse two strong unimodal models without updating either one. The cost is a fixed bottleneck: every image is reduced to 32 query outputs before the language model receives it.

![BLIP-2's two-stage Q-Former training recipe](/assets/images/blip-2-paper-figure-1.png)
*BLIP-2 first teaches the Q-Former to select visual evidence that matches text. The second stage maps those query outputs into a frozen language model. source: [BLIP-2](/paper%20shorts/2023/01/30/blip-2-bootstrapping-language-image-pretraining.html)*

[InstructBLIP](/paper%20shorts/2023/05/11/instructblip-general-purpose-vision-language-instruction-tuning.html) also conditions the Q-Former on the user's instruction before visual compression. A counting question, a description request, and a localization prompt can therefore select different evidence through the same 32-query bottleneck. The instruction affects both the visual features passed to the language model and the response generated from them.

The original [LLaVA](/paper%20shorts/2023/04/01/visual-instruction-tuning-llava.html) removed the learned query bank. It projects each CLIP patch feature directly into the language model and keeps the resulting visual-token sequence. The connector was a single linear layer:

$$
H_v=W f_I(I), \qquad
p(y\mid I,x)=\prod_t p(y_t\mid H_v,x,y_{<t}).
$$

This change also came with a different training contract. LLaVA first freezes the vision encoder and language model while training the projector on 595,000 filtered CC3M image-caption pairs. It then keeps the vision encoder frozen and updates both the projector and language model on 158,000 image-instruction-response examples. BLIP-2 concentrates multimodal learning inside a pretrained query module because both large endpoints remain frozen. LLaVA gives the language model room to adapt during instruction tuning, so the connector does not have to carry the entire alignment problem.

![The LLaVA paper's visual instruction-tuning pipeline](/assets/images/visual-instruction-tuning-llava-paper-figure.png)
*LLaVA separates data generation from model adaptation: GPT-4 first converts image metadata into instruction data, then those examples tune the projected vision-language model. source: [Visual Instruction Tuning](/paper%20shorts/2023/04/01/visual-instruction-tuning-llava.html)*

LLaVA did not invent image-conditioned generation. Frozen, Flamingo, BLIP, PaLI, and BLIP-2 had already built that machinery. Its contribution was a much simpler interface and a practical instruction-tuning recipe. A frozen vision encoder, a projector, and an adaptable language model were enough to produce a capable visual assistant.

[LLaVA-1.5](/paper%20shorts/2023/10/05/improved-baselines-with-visual-instruction-tuning-llava-1-5.html) replaced the linear layer with a two-layer MLP using a GELU nonlinearity:

$$
H_v=W_2\,\operatorname{GELU}(W_1 f_I(I)).
$$

The added layer gives the connector more capacity to transform each patch feature before it enters the language model. In the paper's stepwise ablation, changing only the connector improved GQA from 46.8 to 47.3, MME from 1323.8 to 1355.2, and MM-Vet from 26.3 to 27.8. The full LLaVA-1.5 improvement was larger, but it also included a 336-pixel vision encoder, more academic VQA data, response-format prompts, and a larger language model. The experiment supports the MLP over a linear projector within this recipe. It does not show that an MLP alone explains the final model.

![LLaVA-1.5 combines a CLIP vision encoder, an MLP projector, and a Vicuna language model](/assets/images/improved-baselines-visual-instruction-tuning-paper-figure-1.png)
*LLaVA-1.5 keeps the direct projector architecture and strengthens the surrounding recipe with higher visual resolution and academic VQA data. source: [Improved Baselines with Visual Instruction Tuning](/paper%20shorts/2023/10/05/improved-baselines-with-visual-instruction-tuning-llava-1-5.html)*

A direct projection also supports tasks beyond chat. [PaliGemma](/paper%20shorts/2024/07/10/paligemma-a-versatile-3b-vlm-for-transfer.html) sends SigLIP patch features directly into a Gemma decoder and expresses captioning, VQA, detection, and segmentation as prefix-to-suffix generation. It is first trained as a compact base VLM, then extended from 224 to 448 and 896 pixel inputs before task-specific transfer. Its results show how much capability can come from the data mixture, target format, and visual resolution without adding a more complex connector.

![PaliGemma projects SigLIP patch features directly into a Gemma decoder](/assets/images/paligemma-a-versatile-3b-vlm-for-transfer-paper-figure.png)
*PaliGemma keeps the interface direct: projected image tokens and text tokens enter one decoder sequence. The model is a transferable base VLM rather than primarily a chat assistant. source: [PaliGemma](/paper%20shorts/2024/07/10/paligemma-a-versatile-3b-vlm-for-transfer.html)*

Controlled experiments in [MM1](/paper%20shorts/2024/03/14/mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training.html) put the connector result in context. MM1 compared average pooling followed by a linear projection, learned attention pooling, and a convolutional mapping while also varying image resolution and visual-token count. Resolution and token count had the larger effect in its experiments, while connector type had little conclusive effect. MLP projectors became a strong default because they are cheap, preserve the patch-token interface, and train well with the language model. More elaborate compression is still useful when the visual-token budget is the actual constraint.

### How an image becomes visual tokens

Language reaches a decoder as tokens. An image first has to be turned into them. A fixed-resolution encoder cuts every image into the same patch grid. Tiling runs that encoder over several high-resolution crops. A resampler compresses the grid into a fixed number of learned tokens. Dynamic-resolution models keep a sequence whose length grows with the image. This choice decides what the language model can still see before it generates a single word.

More tokens preserve small text, crowded objects, and local layout, but increase attention cost, memory, and latency. Fewer tokens make the model cheaper to serve, but any detail removed here is gone for the rest of the stack. A larger language model cannot recover a road sign that disappeared during resizing or a second mug that was merged into the background.

The input recipes make the alternatives concrete. [PaliGemma](/paper%20shorts/2024/07/10/paligemma-a-versatile-3b-vlm-for-transfer.html) uses separate 224, 448, and 896 pixel checkpoints. [Qwen2-VL](/paper%20shorts/2024/09/01/qwen2-vl-enhancing-vision-language-model-perception-of-the-world-at-any-resolution.html) keeps the native aspect ratio, converts each image into a variable number of patches, packs those sequences for training, and merges neighboring patches before they enter the language model. [Qwen2.5-VL](/paper%20shorts/2025/02/19/qwen2-5-vl-technical-report.html) extends the same dynamic treatment to documents and video, while [InternVL3.5](/paper%20shorts/2025/08/25/internvl3-5-reasoning-and-efficiency.html) learns when higher resolution is worth the extra compute. Qwen3-VL also injects features from several vision layers so local detail does not have to survive only through the final semantic layer.

This is why visual tokens deserve their own accounting. A comparison between two VLMs is difficult to interpret unless it matches the input pixels, resizing policy, visual-token count, compute, and latency. Otherwise, an apparent architectural improvement may simply come from letting one model look more closely.

### Vision moved earlier into training

Once the connector worked, the next gains came from what the model saw and when it saw it. [Eagle 2](/paper%20shorts/2025/01/01/eagle-2-post-training-data-strategies-for-frontier-vision-language-models.html) varies data quality, task balance, filtering, and training order instead of treating every instruction example as interchangeable. A smaller model trained on a deliberate curriculum can compete with a larger model trained on a poorly organized mixture. The useful accounting is not only how many examples we have, but what behavior each source teaches and which capabilities regress as its weight increases.

Early visual assistants still attached a pretrained vision encoder to a pretrained language model. Newer models begin mixing vision and language earlier in training.

That does not mean starting every weight from scratch. [InternVL3](/paper%20shorts/2025/04/14/internvl3-native-multimodal-pretraining.html) still initializes from strong pretrained components. It calls the recipe native multimodal pretraining because text and multimodal capabilities develop in the same training stage instead of adding vision at the end. Here, *native* describes when the modalities learn together, not where the initial weights came from.

The small visual prefix is also disappearing. [Qwen3-VL](/paper%20shorts/2025/11/26/qwen3-vl-technical-report.html) uses long interleaved context, dynamic visual tokenization, explicit video timestamps, and features from multiple vision layers. The model can revisit visual evidence throughout the sequence instead of consuming one compressed image prefix at the start.

This creates a spectrum rather than a binary distinction:

1. frozen visual encoder plus frozen language model;
2. trainable connector between mostly frozen components;
3. jointly tuned multimodal stack;
4. multimodal data present throughout pretraining;
5. shared generative model over multiple modalities.

Moving down this list gives vision and language more chances to adapt to each other. It also makes the result harder to explain. Better data, more visual tokens, a stronger language model, longer context, and joint training often arrive in the same model release.

## Detour: models that also generate images

Before continuing from generated answers to grounded answers, there is a separate branch worth keeping. The models above treat images as inputs and text as the primary output. Unified multimodal models ask a different question: can the same model also generate images?

One approach converts every modality into discrete tokens. [Unified-IO 2](/paper%20shorts/2023/12/28/unified-io-2-autoregressive-multimodal-model.html) represents vision, language, audio, and action tasks in a shared token space. [Chameleon](/paper%20shorts/2024/05/16/chameleon-mixed-modal-early-fusion-foundation-models.html) trains on interleaved image and text tokens. [Emu3](/paper%20shorts/2024/09/28/emu3-next-token-prediction-multimodal-model.html) applies next-token prediction across tokenized text, images, and video.

This design allows one context to mix text and images in any order, with one transformer attending across both. It also forces the shared representation to serve different objectives. Visual recognition benefits from invariance to texture and lighting, while image generation must preserve texture, color, and local appearance. Discrete image sequences are also long, so autoregressive generation pays a sequential decoding cost for every visual token.

[Transfusion](/paper%20shorts/2024/08/20/transfusion-predict-the-next-token-and-diffuse-images-with-one-multimodal-model.html) separates the shared context from the output objective. It applies next-token prediction to text and diffusion to continuous image patches inside the same transformer. In its comparisons, this hybrid scales better than autoregressive prediction over quantized image tokens. The model can therefore share cross-modal context while retaining an output objective suited to each modality.

The same conflict appears in visual encoding. [Janus](/paper%20shorts/2024/10/17/janus-decoupling-visual-encoding-for-unified-multimodal-understanding-and-generation.html) uses separate visual encoders for understanding and image generation, then shares the autoregressive transformer. Fully tokenized models remain an active branch, but they have not displaced connector-based visual assistants. Current systems more often share semantic context while specializing the encoder, loss, or output head for modality-specific requirements.

![Janus shares one transformer while separating the visual paths for understanding and image generation](/assets/images/janus-decoupling-visual-encoding-for-unified-multimodal-understanding-and-generation-paper-figure.png)
*Janus keeps the autoregressive transformer shared but gives visual understanding and image generation separate encoders. source: [Janus](/paper%20shorts/2024/10/17/janus-decoupling-visual-encoding-for-unified-multimodal-understanding-and-generation.html)*

This branch changes what the model generates. The main VLM line still has an unresolved problem on the input side: can the words in an answer be tied back to the visual evidence that produced them?

## Grounding

Grounding means attaching a word or phrase to the part of the image that supports it, usually a box, point, mask, region, or track. This makes the answer inspectable. A model that says “the traffic light is red” can also show which light it read, making it harder to hide a guess behind fluent language.

Location-aware captioning provided one bridge from generation to grounding. An ordinary caption may say *two dogs playing* without specifying which dog is on the left. [LocCa](/paper%20shorts/2024/03/28/locca-visual-pretraining-with-location-aware-captioners.html) also asks the decoder to describe a region and emit its box, or to generate text for a specified region. The caption now has to preserve which pixels support which words.

![LocCa trains captioning, referring expressions, and grounded captions through one decoder](/assets/images/locca-visual-pretraining-with-location-aware-captioners-paper-figure.png)
*LocCa asks one encoder-decoder to caption the image, describe a boxed region, and ground a phrase with coordinates. Each caption now carries more information about location. source: [LocCa](/paper%20shorts/2024/03/28/locca-visual-pretraining-with-location-aware-captioners.html)*

[MDETR](/paper%20shorts/2021/04/26/mdetr-modulated-detection-for-end-to-end-multimodal-understanding.html) aligns phrases with detected objects end to end. [GLIP](/paper%20shorts/2021/12/07/glip-grounded-language-image-pretraining.html) expresses detection categories through language. [Kosmos-2](/paper%20shorts/2023/06/26/kosmos-2-grounding-multimodal-language-models.html) places coordinates inside generated text, so one autoregressive response can alternate between words and locations. [Molmo](/paper%20shorts/2024/09/01/molmo-and-pixmo-open-weights-and-open-data-for-state-of-the-art-vision-language-models.html) uses pointing data to teach the same association directly, and [Molmo 2](/paper%20shorts/2026/01/15/molmo-2-video-understanding-and-grounding.html) carries those points into video as tracks and timestamps.

The supervision determines what the representation must preserve. A caption can often be produced from global semantics. A region caption adds correspondence. A box adds extent, a mask adds shape, and a track adds identity through time. [Cambrian-1](/paper%20shorts/2024/06/01/cambrian-1-vision-centric-exploration-of-multimodal-llms.html) makes the same point from the architecture side by varying the vision encoder, connector, and vision-heavy data instead of assuming that a stronger language model will compensate for weak visual features.

Grounding is still not geometry. A point such as $(0.63, 0.41)$ identifies a pixel location, but not its depth, camera pose, object orientation, free space, or uncertainty in physical units. [SpatialVLM](/paper%20shorts/2024/01/22/spatialvlm-spatial-reasoning-capabilities.html) adds spatial relations and measurements rather than relying on ordinary captions. Driving and robotics go further by using depth, multiple views, calibration, maps, object pose, and proprioception. Grounding tells us where the evidence appeared in an image. Geometry tells us what that evidence means in the world.

The cleanest test is a counterfactual. If changing the light from red to green does not change the answer, the model was not using the grounded evidence. If masking the light causes the model to abstain, while changing an irrelevant car leaves the answer alone, the prediction is much easier to trust.

## Video-language models: sampling, packing, and time

A video-language model does not receive time for free. It samples frames, converts each frame or short tube into patches, compresses or merges those patches, packs the remaining visual tokens into the language-model context, and adds some representation of order or time. Every step can remove the brief event needed to answer the question.

[LLaVA-OneVision](/paper%20shorts/2024/08/06/llava-onevision-easy-visual-task-transfer.html) reuses one visual interface across images, groups of images, and sampled video frames. [Qwen2-VL](/paper%20shorts/2024/09/01/qwen2-vl-enhancing-vision-language-model-perception-of-the-world-at-any-resolution.html) encodes short temporal tubes and packs variable-length visual sequences. [Qwen2.5-VL](/paper%20shorts/2025/02/19/qwen2-5-vl-technical-report.html) adds absolute time, which lets the decoder distinguish both order and duration. [VideoLLaMA 3](/paper%20shorts/2025/01/01/videollama-3-frontier-multimodal-foundation-models.html) merges redundant tokens across frames, while [Molmo 2](/paper%20shorts/2026/01/15/molmo-2-video-understanding-and-grounding.html) exposes temporal evidence through timestamps and tracks.

Packing more frames helps only if the model preserves what changes between them. Adjacent frames are mostly redundant, so uniform sampling spends tokens on repeated content. Aggressive sampling saves context but can skip a brief handoff, collision, or state change. Token merging has the same tension: remove repeated background, but keep the small region whose motion changes the answer.

| Temporal task | What the representation must preserve | Common shortcut |
| --- | --- | --- |
| Scene description | objects and activities across sampled frames | rely on a few representative frames |
| Event localization | when a visible change occurred | infer the event from its before or after state |
| State tracking | identity and state across time | re-detect each frame independently |
| Action-conditioned prediction | how one action changes the future | generate a plausible future that ignores the action |

The first three tasks test whether the model represents time. The fourth begins to test a world model. If two different actions produce the same plausible continuation, the model may understand video statistics without representing which changes are controllable.

## Detour: JEPA predicts in representation space

The video-language models above still turn visual evidence into words. JEPA changes the pretraining target instead. A joint-embedding predictive architecture hides part of a video and predicts its representation from the visible context. The target can discard unpredictable texture while preserving the state and motion needed to understand what happens next.

[V-JEPA 2](/paper%20shorts/2025/06/11/v-jepa-2-self-supervised-video-models.html) first learns this objective from video without action labels. It then trains a smaller action-conditioned predictor on robot trajectories and uses that predictor for image-goal planning. The video data teaches how scenes tend to change. The robot data teaches which changes an action can cause.

![V-JEPA 2 moves from action-free video prediction to an action-conditioned robot world model](/assets/images/v-jepa-2-paper-figure-1.png)
*V-JEPA 2 first learns a predictive video representation, then adds an action-conditioned predictor for planning. source: [V-JEPA 2](/paper%20shorts/2025/06/11/v-jepa-2-self-supervised-video-models.html)*

A useful exchange between [Rohan Anil](https://x.com/_arohan_/status/2007597891381031029) and [Yann LeCun](https://x.com/ylecun/status/2007907701989232684) gets at the deeper point. JEPA is not defined by opposition to language models or generative decoders. It changes where the predictive burden sits. Instead of asking a decoder to reproduce every unpredictable detail in input space, the encoder learns a latent target that keeps what is useful and suppresses what is not. Preventing that latent space from collapsing is therefore part of the learning problem, not an implementation detail.

A predictive representation is a learned bottleneck. Its value comes from throwing away variation that does not help prediction. Its risk is throwing away the local geometry and motion that a later controller needs.

The global-versus-local problem therefore returns even without language. A strong video embedding can still be poor at depth or local motion. [V-JEPA 2.1](/paper%20shorts/2026/03/15/v-jepa-2-1-dense-video-features.html) adds dense prediction and self-supervision at intermediate layers so depth, anticipation, and interaction do not have to survive only through a global target.

JEPA gives us a predictive latent state rather than a language answer. It will reappear in the pretraining story for world models, but it is not the next step in the VLM lineage. To return to that line, the next output contract changes from describing a scene to deciding what should happen in it.

## From answers to decisions and actions

### Vision-language-action models for driving

Autonomous driving puts nearly every VLM weakness in the same frame: small distant objects, metric geometry, traffic rules, rare hazards, temporal prediction, uncertainty, and a hard latency limit.

There is no agreement on where language should sit. [GPT-Driver](/paper%20shorts/2023/10/01/gpt-driver-learning-to-drive-with-gpt.html) and [Driving with LLMs](/paper%20shorts/2023/10/01/driving-with-llms-fusing-object-level-vector-modality.html) turn driving state into a form a language model can reason over. [DriveVLM](/paper%20shorts/2024/02/01/drivevlm-convergence-of-autonomous-driving-and-large-vision-language-models.html) combines scene reasoning with a conventional planner. [AsyncDriver](/paper%20shorts/2024/06/01/asyncdriver-asynchronous-llm-enhanced-planner-for-autonomous-driving.html) lets the slower language path run separately from faster planning. [VLM-AD](/paper%20shorts/2024/12/19/vlm-ad-end-to-end-autonomous-driving-through-vision-language-supervision.html) uses a VLM as supervision without putting the full model in the control loop.

Calling all of these “VLMs for driving” hides the actual design choice:

| Placement of language | Main benefit | Primary risk |
| --- | --- | --- |
| Online planner | semantic flexibility and explicit reasoning | latency, instability, and geometric imprecision |
| High-level route or behavior selector | separates semantics from low-level control | interface errors between levels |
| Auxiliary explanation head | inspectable supervision and debugging | rationale may not cause the action |
| Offline teacher or data labeler | rich supervision without online cost | teacher errors become training targets |
| Unified end-to-end policy | fewer hand-built interfaces | difficult attribution, calibration, and safety validation |

My current read is that the near-term system will remain hybrid. Let the VLM handle semantics, intent, rare-scenario interpretation, annotation, and auxiliary supervision. Keep metric perception, motion forecasting, constraints, and high-rate control explicit.

End-to-end systems may still win. We just do not get that conclusion from a fluent rationale or a lower open-loop trajectory error. The evidence has to survive closed-loop driving.

#### Testing whether the model uses visual evidence

Fluent language can hide weak dependence on visual evidence. [DriveBench](/paper%20shorts/2025/01/01/are-vlms-ready-for-autonomous-driving-drivebench.html), [IDKB](/paper%20shorts/2024/09/01/can-lvlms-obtain-a-drivers-license-idkb.html), [TOD3Cap](/paper%20shorts/2024/03/01/tod3cap-towards-3d-dense-captioning-in-outdoor-scenes.html), and [AutoTrust](/paper%20shorts/2024/12/01/autotrust-benchmarking-trustworthiness-in-large-vision-language-models-for-autonomous-driving.html) probe different parts of this problem, including visual corruption, 3D detail, and trustworthiness in driving scenes.

The same issue appears outside driving. OCR benchmarks can sometimes be solved through document templates. Visual question answering can reward answer priors. Video benchmarks can leak the event through a single frame. Robotics benchmarks can reward memorized scene layouts.

A useful benchmark should test whether the answer changes with the relevant evidence:

- Does performance fall when the relevant region or frame is corrupted?
- Does the answer follow a changed sign, object state, position, timestamp, or instruction?
- Can the model distinguish “not visible” from “not present”?
- Do irrelevant edits leave the answer unchanged?
- Does confidence track evidence quality and distribution shift?
- Does the rationale identify evidence that actually changes the decision?
- Does an open-loop gain survive closed-loop execution?

Final-answer accuracy can reward memorized priors. Corruptions, counterfactual edits, evidence localization, and calibrated abstention make visual dependence part of the evaluation.

Post-training can then reward the behavior exposed by those tests. [Visual-RFT](/paper%20shorts/2025/03/03/visual-rft-visual-reinforcement-fine-tuning.html) uses task-specific signals such as intersection-over-union for detection. [GRIT](/paper%20shorts/2025/05/21/grit-teaching-mllms-to-think-with-images.html) interleaves language reasoning with explicit region references, making parts of the trace visually inspectable. These methods can teach the decoder to use visual features more reliably. They cannot recover a sign, object, or motion that the visual encoder already discarded.

A controlled diagnostic helps separate those failures. Supplying the relevant crop or a structured scene description tests whether the model can reason once the evidence is explicit. Masking or editing the relevant pixels tests whether the original answer depended on them. The first intervention measures inference; the second measures visual dependence.

For deployment, an aggregate score is less informative than the conditions under which the model fails. Evidence sensitivity, calibrated uncertainty, and recovery behavior can matter even when they do not improve the overall average. [Part III](/blog/2026/07/16/post-training-vision-language-action-models-zero-to-hero.html) takes the post-training story further, covering rewards, preferences, interventions, critics, and online rollouts after a robot begins changing its own data distribution.

### Robot actions close the loop

Robotics adds a closed-loop consequence: the model's output changes its next input. A bad action moves the robot into a new state, and later predictions must now recover from the result of the earlier error.

[PaLM-E](/paper%20shorts/2023/03/06/palm-e-embodied-multimodal-language-model.html) expanded the language model's input with visual observations and continuous state estimates. This allowed embodied planning and question answering to reuse the pretrained language model while retaining continuous robot state as a separate input representation.

[RT-2](/paper%20shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html) then put control on the output side by writing robot actions as tokens. The same autoregressive decoder could answer a question or emit a command. [OpenVLA](/paper%20shorts/2024/06/01/openvla-open-source-vision-language-action-model.html) made that modern VLA recipe open and inspectable.

Once the model had to act, tokenizing the action became a major design choice. [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html) keeps the semantic VLM and gives continuous actions a flow-based expert. [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html) takes the opposite route and compresses action chunks in the frequency domain so the language decoder emits fewer tokens. [DexVLA](/paper%20shorts/2025/02/01/dexvla-vision-language-model-with-plug-in-diffusion-expert.html) adds a diffusion expert for dexterous control, while [GR00T N1](/paper%20shorts/2025/03/18/groot-n1-open-foundation-model-for-humanoid-robots.html) separates semantic reasoning from continuous action generation. These models can share a backbone without forcing words and motor commands into the same distribution.

#### From imitation to generalization, experience, and scale

Later VLA models changed the composition and source of the training data while retaining related decoder families.

[π0.5](/paper%20shorts/2025/04/22/pi0-5-vision-language-action-model-with-open-world-generalization.html) mixes web data, multiple robots, object detections, semantic subtasks, and low-level actions to push beyond a closed set of tasks. [π*0.6](/paper%20shorts/2025/11/18/pi-star-0-6-vla-learns-from-experience.html) adds autonomous rollouts and expert corrections, turning deployment failures into new training data. [π0.7](/paper%20shorts/2026/04/16/pi0-7-steerable-generalist-robotic-foundation-model.html) lets strategy, metadata, and subgoal images steer how the same policy completes a task.

Robot pretraining data also need not originate from a deployed robot. [Xiaomi-Robotics-1](/paper%20shorts/2026/07/16/xiaomi-robotics-1-scaling-vla-with-real-world-trajectories.html) reports more than 100,000 hours of real-world UMI trajectories, automatically labeled with descriptions of the observed state change. A later cross-embodiment stage aligns those behaviors with robot actions and imperative instructions. The reported scale matters, but the more general idea is to collect diverse manipulation through a cheaper interface and align it with robot control afterward.

This is where pretraining and post-training separate cleanly. Pretraining gives the policy a broad starting point. Post-training adapts it to one robot, command distribution, safety envelope, and deployment environment. Experience adds the failures and recoveries that demonstrations missed.

The action representation decides what motions the policy can express, how it is trained, and how quickly it can react.

| Action interface | Advantage | Cost |
| --- | --- | --- |
| Per-step regression | simple and fast | averages multimodal action distributions |
| Discrete action tokens | exact autoregressive likelihood and shared vocabulary | quantization and sequential decoding latency |
| Compressed action tokens | shorter sequences over long horizons | compression may remove abrupt corrections |
| Diffusion or flow chunks | expressive continuous distributions | iterative generation and harder likelihood-based RL |
| Separate action expert | specialization without discarding VLM semantics | extra parameters and coordination path |

A caption has no control frequency. An action does. A policy must fit sensing, inference, communication, and actuation inside a deadline. It must also decide how long an action chunk remains valid before new evidence should interrupt it. Longer chunks reduce inference calls and improve temporal coherence. Shorter chunks respond faster to disturbances.

Only part of the web-trained model transfers cleanly:

- semantic concepts, object knowledge, and instruction following can transfer;
- embodiment, contact, calibration, and control timing must be learned or represented explicitly;
- action data are expensive because they must cover not only desired behavior but recovery states created by the policy.

Language is a good interface for the task. It does not make units, embodiment, contact, or control latency disappear.

## Recap: what the representation must preserve

The history becomes easier to read when each paper is reduced to four questions. What does the model produce? Where can visual evidence be lost? Which supervision forces the new capability? Does the evaluation actually require that evidence? The table below applies those questions to the main literature threads.

| Literature thread | Start with | Main output | Reading exercise |
| --- | --- | --- | --- |
| Cross-modal pretraining | [ViLBERT](/paper%20shorts/2019/08/06/vilbert-pretraining-task-agnostic-visiolinguistic-representations.html), [LXMERT](/paper%20shorts/2019/08/20/lxmert-learning-cross-modality-encoder-representations.html), [UNITER](/paper%20shorts/2019/09/25/uniter-universal-image-text-representation-learning.html) | fused word-region representation | draw both streams, mark where they interact, and list the assumptions introduced by detector regions |
| Image-text contrastive learning | [CLIP](/paper%20shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html), [SigLIP](/paper%20shorts/2023/10/01/sigmoid-loss-for-language-image-pre-training-siglip.html), [SigLIP 2](/paper%20shorts/2025/02/20/siglip-2-multilingual-vision-language-encoders.html) | similarity or retrieval score | derive the softmax and sigmoid losses, then state which spatial information each objective can ignore |
| Generative vision-language models | [BLIP](/paper%20shorts/2022/01/28/blip-bootstrapping-language-image-pretraining.html), [Flamingo](/paper%20shorts/2022/04/29/flamingo-visual-language-model-for-few-shot-learning.html), [PaLI](/paper%20shorts/2022/09/14/pali-jointly-scaled-multilingual-language-image-model.html), [BLIP-2](/paper%20shorts/2023/01/30/blip-2-bootstrapping-language-image-pretraining.html), [PaliGemma](/paper%20shorts/2024/07/10/paligemma-a-versatile-3b-vlm-for-transfer.html) | generated text | compare the trainable components, visual bottleneck, and route into the language decoder |
| Instruction-tuned multimodal LLMs | [LLaVA](/paper%20shorts/2023/04/01/visual-instruction-tuning-llava.html), [InstructBLIP](/paper%20shorts/2023/05/11/instructblip-general-purpose-vision-language-instruction-tuning.html), [LLaVA-1.5](/paper%20shorts/2023/10/05/improved-baselines-with-visual-instruction-tuning-llava-1-5.html), [MM1](/paper%20shorts/2024/03/14/mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training.html), [Eagle 2](/paper%20shorts/2025/01/01/eagle-2-post-training-data-strategies-for-frontier-vision-language-models.html) | assistant response | separate the effects of the visual encoder, connector, pretraining mixture, instruction mixture, and preference data |
| Unified multimodal generation, a parallel branch | [Unified-IO 2](/paper%20shorts/2023/12/28/unified-io-2-autoregressive-multimodal-model.html), [Chameleon](/paper%20shorts/2024/05/16/chameleon-mixed-modal-early-fusion-foundation-models.html), [Transfusion](/paper%20shorts/2024/08/20/transfusion-predict-the-next-token-and-diffuse-images-with-one-multimodal-model.html), [Emu3](/paper%20shorts/2024/09/28/emu3-next-token-prediction-multimodal-model.html), [Janus](/paper%20shorts/2024/10/17/janus-decoupling-visual-encoding-for-unified-multimodal-understanding-and-generation.html) | text, images, or mixed sequences | mark what is actually shared: context, transformer, tokenizer, objective, encoder, or output head |
| Open-vocabulary detection and grounding | [Detic](/paper%20shorts/2022/01/07/detic-detecting-twenty-thousand-classes-using-image-level-supervision.html), [OWL-ViT](/paper%20shorts/2022/05/12/owl-vit-simple-open-vocabulary-object-detection-with-vision-transformers.html), [MDETR](/paper%20shorts/2021/04/26/mdetr-modulated-detection-for-end-to-end-multimodal-understanding.html), [Kosmos-2](/paper%20shorts/2023/06/26/kosmos-2-grounding-multimodal-language-models.html), [Molmo](/paper%20shorts/2024/09/01/molmo-and-pixmo-open-weights-and-open-data-for-state-of-the-art-vision-language-models.html), [SpatialVLM](/paper%20shorts/2024/01/22/spatialvlm-spatial-reasoning-capabilities.html) | boxes, points, grounded text, or metric relations | estimate the visual-token budget and mark the strongest supervision: caption, box, point, mask, relation, or metric target |
| Video-language models | [LLaVA-OneVision](/paper%20shorts/2024/08/06/llava-onevision-easy-visual-task-transfer.html), [VideoLLaMA 3](/paper%20shorts/2025/01/01/videollama-3-frontier-multimodal-foundation-models.html), [Molmo 2](/paper%20shorts/2026/01/15/molmo-2-video-understanding-and-grounding.html) | text, timestamp, or track | design one temporal edit that changes the answer and one irrelevant edit that should not |
| Predictive video representations, a parallel branch | [V-JEPA 2](/paper%20shorts/2025/06/11/v-jepa-2-self-supervised-video-models.html), [V-JEPA 2.1](/paper%20shorts/2026/03/15/v-jepa-2-1-dense-video-features.html) | latent state or predicted representation | identify what the target is allowed to discard, then test whether local motion and geometry survive |
| Vision-language decision and action models | [DriveVLM](/paper%20shorts/2024/02/01/drivevlm-convergence-of-autonomous-driving-and-large-vision-language-models.html), [VLM-AD](/paper%20shorts/2024/12/19/vlm-ad-end-to-end-autonomous-driving-through-vision-language-model-supervision.html), [RT-2](/paper%20shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html), [OpenVLA](/paper%20shorts/2024/06/01/openvla-open-source-vision-language-action-model.html), [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html), [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html) | rationale, plan, trajectory, or action | reconstruct the action distribution, horizon, control rate, inference path, and recovery mechanism |

The frame-by-frame explainer below holds the mug-and-tray scene fixed and changes only the required output:

<div class="architecture-comparison blog-frame-explainer" data-blog-frame-explainer="blog-vlm-evidence-contract.gif"><div class="blog-frame-explainer__viewport"><a href="/assets/images/blog-explainer-frames/blog-vlm-evidence-contract/frame-01.webp"><img src="/assets/images/blog-explainer-frames/blog-vlm-evidence-contract/frame-01.webp" alt="Manual explainer comparing the outputs of CLIP, LLaVA, Molmo, and Pi0 on the same mug-and-tray scene"></a></div></div>

*CLIP compares an image with text. LLaVA generates text from visual tokens. Molmo uses pointing data to bind a phrase to a location. Pi0 produces continuous robot actions through a flow-based action expert. The panels compare outputs, not a literal model lineage. Explanatory synthesis based on [CLIP](/paper%20shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html), [LLaVA](/paper%20shorts/2023/04/01/visual-instruction-tuning-llava.html), [Molmo](/paper%20shorts/2024/09/01/molmo-and-pixmo-open-weights-and-open-data-for-state-of-the-art-vision-language-models.html), and [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html).*

Across this history, every new capability asks vision to preserve something it could previously ignore. Region-based models connected words to detected objects. CLIP made images searchable through language at web scale. Generative bridges gave those features to a language model, and instruction tuning turned the result into an assistant. Grounding tied words back to pixels. Video added time. Driving and robotics made the remaining shortcuts expensive because a plausible sentence could now produce a bad physical decision.

My strongest bet is a shared semantic model with separate high-bandwidth paths for geometry, time, image generation, and control. I would replace that hybrid with one token stream only when it wins under matched data, pixels, tokens, parameters, compute, and latency. The win also has to hold on fine grounding, metric spatial reasoning, temporal counterfactuals, calibration, and closed-loop recovery.

Until then, “one model for everything” is a research program, not an architectural result.

For further reading, I split the robotics side of this story into two posts. [Pre-Training for Robotics](/blog/2026/07/15/omni-model-pretraining-decisions.html) looks at how multimodal and robot data shape a base policy. [Post-Training for Robotics](/blog/2026/07/16/post-training-vision-language-action-models-zero-to-hero.html) looks at how deployment feedback and failures refine that policy. Together, they show how the VLM capabilities in this post are carried into robot behavior.
