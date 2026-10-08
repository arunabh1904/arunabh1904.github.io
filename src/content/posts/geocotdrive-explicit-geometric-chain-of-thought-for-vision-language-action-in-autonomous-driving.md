---
title: 'GeoCoTDrive: Explicit Geometric Chain-of-Thought for Vision-Language-Action in Autonomous Driving'
date: '2026-10-07T00:00:00.000Z'
section: paper-shorts
postSlug: geocotdrive-explicit-geometric-chain-of-thought-for-vision-language-action-in-autonomous-driving
legacyPath: /paper shorts/2026/10/07/geocotdrive-explicit-geometric-chain-of-thought-for-vision-language-action-in-autonomous-driving.html
tags: [Autonomous Driving, VLA]
field: 'Autonomous Driving: VLA & Planning'
summary: '2026 – Explicit Geometric Chain-of-Thought for Vision-Language-Action in Autonomous Driving'
---

## 2026 – GeoCoTDrive: Explicit Geometric Chain-of-Thought for Vision-Language-Action in Autonomous Driving

**Paper:** [arXiv:2610.10390](https://arxiv.org/abs/2610.10390) · [Full text, v1](https://arxiv.org/html/2610.10390v1) · [Official code](https://github.com/TabGuigui/GeoCoTDrive)

## Summary

> GeoCoTDrive makes a driving VLA predict planning-relevant image regions, retrieve geometry features inside those regions, and use the retrieved features to plan. On NAVSIM v1, this raises PDMS from 85.9 to 87.6; the paper's global-geometry alternative scores 85.8. The result supports selective geometry retrieval in this setup. It also creates a dependency: a missed region can exclude the geometry needed for a safe action, and training retrieves from correct boxes while inference uses predicted boxes.

## Core Insights

### Ground the part of the scene that constrains the path

A box around a pedestrian is useful, but planning also depends on spaces that are not ordinary objects. A curb constrains a turn. An occluded opening creates uncertainty about traffic. GeoCoTDrive trains the VLA to identify these decision-relevant regions before it predicts the ego trajectory. Its grounding vocabulary therefore includes critical objects, road boundaries, conflict areas, occluded or unknown areas, and dense object areas.

The model receives a front-view image, a navigation command, and ego status. The paper uses LLaVA-1.5-7B on nuScenes and Bench2Drive, and Qwen2.5-VL-3B on NAVSIM. Ego-motion context can appear in the prompt, but the proposed visual setup does not explicitly model a multi-view image history. This matters for regions whose state cannot be resolved from the current front view.

How does a predicted region change the action context? Source Figure 2 follows the path from a grounding answer to a geometry sampler, then back into the language model. The geometry model supplies continuous features. The grounding answer supplies the locations at which to retrieve them.

![GeoCoTDrive source Figure 2: predicted image boxes select geometry features that are inserted before trajectory generation](/assets/images/geocotdrive-source-figure-2.png)
*Fig 1: The model predicts planning-relevant boxes, samples geometry features within each box, and inserts the projected features into the autoregressive context before planning. | source: [GeoCoTDrive, Figure 2](https://arxiv.org/abs/2610.10390)*

Figure reproduced without modification under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). [Open the full-size figure](/assets/images/geocotdrive-source-figure-2.png).

### Project feature channels while retaining sampled positions

The geometry encoder is a frozen Depth Anything 3 METRIC-LARGE model or vanilla VGGT. The sampler reads its last backbone feature map, rather than treating a predicted depth image as a list of exact 3D coordinates. Let this map be $G\in\mathbb{R}^{H_g\times W_g\times C_g}$. Each predicted box contains normalized image coordinates $(x_{\min},y_{\min},x_{\max},y_{\max})$.

For each of $N$ boxes, the sampler places a regular grid of $K$ points inside the box. Bilinear interpolation retrieves one $C_g$-dimensional feature per point. An MLP then maps each feature to the language model's embedding width $d_{\mathrm{LM}}$:

$$
\widetilde Z_i=\operatorname{Sample}(G,b_i)\in\mathbb{R}^{K\times C_g},
\qquad
Z_i=\operatorname{MLP}(\widetilde Z_i)\in\mathbb{R}^{K\times d_{\mathrm{LM}}}.
$$

Concatenating the boxes produces $NK$ geometry tokens. The projection changes channel width; it does not combine all sampled positions into one wide token. A curb region can therefore contribute several local features along its extent. This distinction matters when adapting a smaller encoder to a language model: width alignment and spatial compression are separate operations.

The VLA first generates its grounding answer. A special `<GEO_COT>` marker triggers retrieval, and the projected geometry tokens enter the sequence before trajectory prediction continues. For nuScenes and NAVSIM, the model generates trajectory coordinates autoregressively. Bench2Drive uses a different action path: an ORION planner receives VLM and localized geometry tokens as conditions, with the detection and mapping heads removed. Its result should not be attributed to an identical coordinate decoder across all three benchmarks.

The separate sampling ablation makes the token trade-off concrete. One point per region gives 86.4 PDMS; a $2\times2$ grid gives 87.0; a $4\times4$ grid gives 87.3. Increasing to $5\times5$ retains 87.3. These are the values from Table 11, whose setting should be kept separate from the main 87.6 result. The evidence supports retaining some regional detail, with no measured gain from the densest tested grid.

### Train grounding before geometry-conditioned planning

PlanningGrounding contains 146,000 question-answer annotations: 28,000 from nuScenes, 102,000 from NAVSIM, and 16,000 from Bench2Drive. Gemini 3.1 proposes planning-relevant regions from the image and driving context. The construction pipeline parses the coordinates, aligns each explanation with its box, checks the output format, and applies human verification for validity, tightness, and planning relevance. These are generated and reviewed annotations, rather than direct measurements of which region caused an expert action.

The paper also constructs a 600-example, manually refined grounding test set. It reports the source composition and validation process, but does not provide a detailed scene-level account of leakage controls or annotator agreement. That limits independent assessment of how much the grounding test isolates generalization beyond its source scenes.

Training has two stages. First, both the vision encoder and language model learn driving QA for three epochs. Each benchmark combines PlanningGrounding with its driving data: OmniDrive for nuScenes, NAVSIM-Traj for NAVSIM, and B2D-Chat for Bench2Drive. The cosine schedule peaks at $4\times10^{-5}$.

Second, the vision and geometry encoders stay frozen while the language model and geometry aligner train for three epochs. The peak learning rate is $2\times10^{-5}$, again with a cosine schedule. Autoregressive supervision covers the grounding answer and planning answer; prompt, image, and inserted continuous geometry features are excluded from the token loss. The experiments use eight A800 GPUs.

During this second stage, the sampler uses ground-truth boxes. At inference, it uses the model's boxes. That change is a concrete exposure gap: the planner learns with correctly retrieved regions, then must tolerate retrieval errors at deployment. The grounding evaluation reaches 38.93 mIoU and 21.04 recall at IoU 0.7, so precise region selection remains far from complete. Perturbation experiments also show planning performance declining as grounding quality falls.

### Separate selective-retrieval evidence from driving safety

The main geometry-integration ablation compares three NAVSIM v1 systems: the baseline at 85.9 PDMS, global geometry at 85.8, and GeoCoTDrive at 87.6. This is the clearest evidence for the paper's changed mechanism. It does not establish that every global fusion design is inferior, particularly without a matched account of token budgets and inference cost.

The broader results use different evaluation protocols and should remain separate:

| Evaluation | Reported result | What the comparison establishes |
| --- | --- | --- |
| nuScenes open-loop planning | Average L2 error 0.32 m; average collision rate 0.11% | Predicted trajectories improve selected geometric safety measures without rolling out an interacting vehicle |
| NAVSIM v1 non-reactive simulation | PDMS 85.9 → 87.6 in the geometry ablation | Local retrieval improves the aggregate planning score in this benchmark |
| Bench2Drive closed-loop driving | ORION driving score 77.74 → 78.68; success rate 54.62% → 55.42% | The ORION-based implementation gains modestly under interactive rollout |

The nuScenes results also contain a useful boundary. GeoCoTDrive's average intersection rate is 2.02%, whereas SpaceDrive reports 1.27%. A lower collision rate does not make every road-geometry metric best. The closed-loop gains likewise do not establish real-vehicle safety or quantify the uncertainty of the improvement.

Compared with [Geo-VLA's map-grounded training](/paper%20shorts/2026/08/18/geo-vla-geometry-aware-vision-language-action-planning-via-internalization-of-map-semantics.html), GeoCoTDrive retains a geometry encoder and a retrieval step at inference. Geo-VLA transfers map semantics through QA and then uses its ordinary action decoder. The technical choice is where to pay for geometry: in training supervision alone, or also in the deployed computation. GeoCoTDrive does not report a measured end-to-end latency that resolves that cost trade-off.

As checked on 8 October 2026, the official repository lists code and checkpoints as released, with PlanningGrounding and the training-script release still pending. Evaluation artifacts help inspect the method, but a complete retraining reproduction still depends on those missing releases.

## High-Level Takeaways

- GeoCoTDrive uses predicted 2D regions to select continuous geometry evidence. Its strongest controlled result is 87.6 versus 85.9 PDMS, with the tested global-feature alternative at 85.8.
- The aligner projects each sampled feature to the language width. The $NK$ sampled positions remain separate tokens, and the grid ablation shows diminishing returns beyond 16 samples per region.
- Ground-truth retrieval during training and predicted retrieval during inference create a specific failure boundary. Better trajectory loss alone cannot recover evidence from a region that was never selected.
- My synthesis: the next decisive comparison should match token budget and measured latency across regional and global retrieval, then test missed and displaced boxes. If the regional advantage disappears under that control, the argument for the extra grounding stage becomes weaker.
