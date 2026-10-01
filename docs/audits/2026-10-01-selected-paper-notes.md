# October 1 selected-paper release audit

Scope: the 12 Add to Blog selections from the current recovery batch. Four older queued papers are outside this release. All notes use `section: paper-shorts`, Markdown, preserved queue slugs/dates/legacy routes, and existing field taxonomy.

Canonical full texts and relevant technical appendices were reviewed. Source figures were downloaded from the corresponding arXiv HTML and converted to local WebP; no generated diagrams were substituted. Colosseum follows v4, while the other papers follow v1. Missing recipe details and source inconsistencies are identified in the notes.

Validation: canonical `npm run ci`, figure/caption/source checks, and visual inspection of all 12 rendered pages at 800px and 390px content widths. All 18 source images loaded; no page-level horizontal overflow or KaTeX errors were observed. Wide tables use the existing contained scrolling behavior. Full-resolution figure links support reading dense source labels.

## Notes and figure purposes

### [AD-E2E-JEPA: A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving](https://arunabh1904.github.io/paper%20shorts/2026/09/28/ad-e2e-jepa-a-joint-embedding-predictive-architecture-for-end-to-end-autonomous-driving.html)

- Source: [arXiv:2609.34085](https://arxiv.org/abs/2609.34085).
- Field: Autonomous Driving: VLA & Planning. Date: 2026-09-28T09:00:00.000Z. Tags: Autonomous Driving, Research.
- Images: 1.
- Figure purpose: A shared projector compresses current and target visual features. Planning searches candidate actions through the learned dynamics; a separate experiment transfers the projector to an imitation model.

### [An overview of 3D Vision-Language Models](https://arunabh1904.github.io/paper%20shorts/2026/09/04/an-overview-of-3d-vision-language-models.html)

- Source: [arXiv:2609.05583](https://arxiv.org/abs/2609.05583).
- Field: Vision-Language Models. Date: 2026-09-04T09:00:00.000Z. Tags: 3D Vision, Vision-Language Models, Survey.
- Images: 2.
- Figure purpose: A shared 3D embedding is supervised by image and text matching. The diagonal pairs are positives; the other batch entries supply competing candidates.
- Figure purpose: The same broad family serves categorization, retrieval, localization, language interaction, generation, and control. Each output requires its own evaluation contract.

### [AnchorReasoning: A Visual Grounding and Causal Reasoning Dataset in Long-Tail Autonomous Driving Scenarios](https://arunabh1904.github.io/paper%20shorts/2026/09/23/anchorreasoning-a-visual-grounding-and-causal-reasoning-dataset-in-long-tail-autonomous-driving-scen.html)

- Source: [arXiv:2609.28366](https://arxiv.org/abs/2609.28366).
- Field: Autonomous Driving: VLMs & Evaluation. Date: 2026-09-23T09:00:00.000Z. Tags: Autonomous Driving, Research.
- Images: 1.
- Figure purpose: Selected image regions are linked to attributes, implications, a combined rationale, and a driving plan. Human, model-assisted, and rule-based stages contribute different labels.

### [CausalDriveBench: Evaluating Causal Reasoning in Vision-Language-Action Models for Autonomous Driving](https://arunabh1904.github.io/paper%20shorts/2026/09/26/causaldrivebench-evaluating-causal-reasoning-in-vision-language-action-models-for-autonomous-driving.html)

- Source: [arXiv:2609.32157](https://arxiv.org/abs/2609.32157).
- Field: Autonomous Driving: VLMs & Evaluation. Date: 2026-09-26T09:00:00.000Z. Tags: Autonomous Driving, Research.
- Images: 1.
- Figure purpose: Graph extraction, question generation, and alternative-trajectory synthesis depend on shared scene evidence. Human graph review and separate QA and trajectory audits constrain different sources of annotation error.

### [Colosseum V2: Benchmarking Generalization for Vision-Language-Action Models](https://arunabh1904.github.io/paper%20shorts/2026/10/01/colosseum-v2-benchmarking-generalization-for-vision-language-action-models.html)

- Source: [arXiv:2605.27759](https://arxiv.org/abs/2605.27759).
- Field: Robot Post-Training & Evaluation. Date: 2026-10-01T09:00:00.000Z. Tags: Robotics, Benchmark.
- Images: 2.
- Figure purpose: The benchmark crosses manipulation tasks with controlled changes to observations, instructions, and physical configuration. Separate single-arm and bimanual suites expose different coordination demands.
- Figure purpose: Perturbation effects differ across model families and robot setups. This figure filters tasks by minimum baseline success, so its averages should be read alongside the all-task base scores.

### [FIVE-VLA: Fast and EffectIVE Autonomous Driving with Recurrent Action Memory](https://arunabh1904.github.io/paper%20shorts/2026/09/16/five-vla-fast-and-effective-autonomous-driving-with-recurrent-action-memory.html)

- Source: [arXiv:2609.18623](https://arxiv.org/abs/2609.18623).
- Field: Autonomous Driving: VLA & Planning. Date: 2026-09-16T09:00:00.000Z. Tags: Autonomous Driving, Research.
- Images: 1.
- Figure purpose: Recurrent memory modifies the action-query input while retaining the pretrained visual encoder and visual-language adapter. Separate decoders translate contextualized action tokens into path and speed waypoints.

### [Generation of Vectorized Maps Beyond Vehicle View](https://arunabh1904.github.io/paper%20shorts/2026/09/07/generation-of-vectorized-maps-beyond-vehicle-view.html)

- Source: [arXiv:2609.07511](https://arxiv.org/abs/2609.07511).
- Field: BEV Perception & Mapping. Date: 2026-09-07T09:00:00.000Z. Tags: Autonomous Driving, Mapping.
- Images: 2.
- Figure purpose: The map supplies global context and the endpoint encoder identifies where continuation begins. Geometry and topology heads jointly constrain the autoregressive output.
- Figure purpose: The learned continuation can follow longer-range curvature while retaining substantial geometric and topological errors. Lane splits remain a visible failure.

### [NeuroSymbEAD: A Large Scale Neuro-Symbolic Caption Dataset for Omni-Directional Embodied Autonomous Driving](https://arunabh1904.github.io/paper%20shorts/2026/09/15/neurosymbead-a-large-scale-neuro-symbolic-caption-dataset-for-omni-directional-embodied-autonomous-d.html)

- Source: [arXiv:2609.16919](https://arxiv.org/abs/2609.16919).
- Field: Autonomous Driving: VLMs & Evaluation. Date: 2026-09-15T09:00:00.000Z. Tags: Autonomous Driving, Research.
- Images: 2.
- Figure purpose: The graph exposes semantic and geometric relations between the ego vehicle, individual objects, and object groups. Those relations become the ingredients of captions.
- Figure purpose: Object attributes are assembled into explicit ego-relative descriptions. This is the caption-generation panel of source Figure 3, separated from its spatial-relation panel.

### [Planning-Aligned Pretraining of BEV Representations with Sparse Action-Conditioned Targets for End-to-End Autonomous Driving](https://arunabh1904.github.io/paper%20shorts/2026/09/19/planning-aligned-pretraining-of-bev-representations-with-sparse-action-conditioned-targets-for-end-t.html)

- Source: [arXiv:2609.22868](https://arxiv.org/abs/2609.22868).
- Field: Autonomous Driving: VLA & Planning. Date: 2026-09-19T09:00:00.000Z. Tags: Autonomous Driving, Research.
- Images: 1.
- Figure purpose: Candidate motions index sparse LiDAR evidence. A masked camera-BEV branch predicts two target ratios, and the entire auxiliary branch is removed after pretraining.

### [RAF-VLA: Representation Alignment with the Future for End-to-End Autonomous Driving](https://arunabh1904.github.io/paper%20shorts/2026/09/15/raf-vla-representation-alignment-with-the-future-for-end-to-end-autonomous-driving.html)

- Source: [arXiv:2609.17728](https://arxiv.org/abs/2609.17728).
- Field: Autonomous Driving: VLA & Planning. Date: 2026-09-15T09:00:00.000Z. Tags: Autonomous Driving, Research.
- Images: 1.
- Figure purpose: Separate experts retain their own transformations while exchanging context through masked joint attention. The action branch reads the vision-language context without feeding action tokens back into it.

### [Vision-Language-Action Autonomous Driving Agent with Language-based Memory](https://arunabh1904.github.io/paper%20shorts/2026/09/29/vision-language-action-autonomous-driving-agent-with-language-based-memory.html)

- Source: [arXiv:2609.38641](https://arxiv.org/abs/2609.38641).
- Field: Autonomous Driving: VLA & Planning. Date: 2026-09-29T09:00:00.000Z. Tags: Autonomous Driving, Research.
- Images: 2.
- Figure purpose: The driving target includes a memory description that can be reused at later keyframes. Relevant object state is made explicit in language alongside reasoning and trajectory output.
- Figure purpose: Parallel rollouts share recorded environmental observations but maintain separate generated memories. Driving tokens receive local credit, while memory tokens receive credit for later driving and terminal question answering.

### [WZPlanner: Safe End-to-End Path Planning for Autonomous Driving in Work Zones](https://arunabh1904.github.io/paper%20shorts/2026/09/16/wzplanner-safe-end-to-end-path-planning-for-autonomous-driving-in-work-zones.html)

- Source: [arXiv:2609.19393](https://arxiv.org/abs/2609.19393).
- Field: Motion Forecasting & Planning. Date: 2026-09-16T09:00:00.000Z. Tags: Autonomous Driving, Research.
- Images: 2.
- Figure purpose: Human scenario design and trajectory recording supply the geometry that WAVE transforms into calibrated training labels. Synthetic variation expands coverage without making those labels independently observed human driving decisions.
- Figure purpose: The original architecture tests whether trajectories should share boundary slots or receive a dedicated global decoder. This figure depicts BF; the later BF++ changes the backbone, geometry representation, and typed queries.

## Evidence boundaries retained

- Benchmark notes trace source data, sample units, label generation, human/model/rule roles, split boundaries, metrics, and audit coverage.
- The 3D VLM tutorial covers representations, five paradigms, alignment and generative interfaces, representative families, applications, data routes, and the absence of a harmonized evaluation table.
- Driving-method notes distinguish visual connectors, auxiliary alignment/compression projectors, action heads, and memory; they include reported inputs, stage objectives, frozen components, schedules, hardware, and evaluation regimes.
- Important qualifications include privileged maps/goals/boxes, replay versus closed-loop control, weak OOD generalization, differing metric denominators, and unreported settings.

The installed paper-note and website-publishing guidance was updated separately in the local skill repository; personal skill sources are not included in this public website change.
