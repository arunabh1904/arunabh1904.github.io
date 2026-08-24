---
title: 'G-MARK: Grounded Multi-Agent Reasoning for Cooperative Driving via Knowledge Graphs'
date: '2026-08-20T09:00:00.000Z'
section: paper-shorts
postSlug: g-mark-grounded-multi-agent-reasoning-for-cooperative-driving-via-knowledge-graphs
legacyPath: /paper shorts/2026/08/20/g-mark-grounded-multi-agent-reasoning-for-cooperative-driving-via-knowledge-graphs.html
tags:
  - Autonomous Driving
  - Cooperative Perception
  - Knowledge Graphs
field: 'Autonomous Driving: VLMs & Evaluation'
summary: '2026 – G-MARK: Grounded Multi-Agent Reasoning for Cooperative Driving via Knowledge Graphs'
---

## 2026 – G-MARK: Grounded Multi-Agent Reasoning for Cooperative Driving via Knowledge Graphs

**arXiv:** [2608.19964](https://arxiv.org/abs/2608.19964)

**Code:** [bhavyagupta98/g-mark](https://github.com/bhavyagupta98/g-mark)

## Summary

> G-MARK delays cooperative evidence fusion until the downstream task is known. It converts processed multi-vehicle observations into a graph that keeps each object hypothesis tied to its observing agents, visibility, uncertainty, disagreement, and path relevance. On V2V-GoT-QA, this explicit evidence improves eight of nine reported tasks over V2V-GoT, including a 42.2% relative gain in occlusion-reasoning F1, while future-trajectory error is 3.4% worse. The graph payload is 25.6× smaller than V2V-GoT's reported communication, but the study begins after perception and does not evaluate closed-loop driving.

## Core Insights

![G-MARK pipeline from per-agent evidence graphs through conservative association and context enrichment to task heads](/assets/images/g-mark-cooperative-knowledge-graph-framework.png)
*G-MARK preserves agent, observation, and hypothesis nodes through association, then adds provenance and planning context before task-specific inference. This makes delayed fusion inspectable instead of collapsing evidence into one object list. org/abs/2608.19964). source: [G-MARK paper](https://arxiv.org/abs/2608.19964)*

![Figure 1 from G-MARK: Grounded Multi-Agent Reasoning for Cooperative Driving via Knowledge Graphs](/assets/images/g-mark-grounded-multi-agent-reasoning-for-cooperative-driving-via-knowledge-graphs-source-figure-1.webp)
*Figure 1 Fig. 1: Overview of the G-MARK cooperative KG reasoning framework. source: [G-MARK: Grounded Multi-Agent Reasoning for Cooperative Driving via Knowledge Graphs](https://arxiv.org/abs/2608.19964)*

![Figure 2 from G-MARK: Grounded Multi-Agent Reasoning for Cooperative Driving via Knowledge Graphs](/assets/images/g-mark-grounded-multi-agent-reasoning-for-cooperative-driving-via-knowledge-graphs-source-figure-2.webp)
*Figure 2 Fig. 2: Planning accuracy versus communication cost. source: [G-MARK: Grounded Multi-Agent Reasoning for Cooperative Driving via Knowledge Graphs](https://arxiv.org/abs/2608.19964)*


### The graph preserves how an object became believable

G-MARK operates above an existing cooperative perception stack. Its inputs are processed boxes or tracks, confidence scores, agent poses, timestamps, and trajectory context—not raw camera or LiDAR streams. Each vehicle contributes agent nodes and observation nodes. Object-hypothesis nodes remain separate until a conservative association gate finds an exact class match within a class-dependent distance threshold.

The association stage does not discard unmatched evidence. It keeps a weak partner-only observation as an uncertain candidate, records every supporting observation, and summarizes confidence with a bounded noisy-or rule. Context enrichment then adds agent-relative visibility, provenance, disagreement, motion cues, and distance to the ego path. Lightweight rankers, regressors, and classifiers read the resulting features for object retrieval, motion prediction, control selection, and trajectory forecasting.

This representation changes the role of a graph relative to nearby driving work. [DriveLM](/paper%20shorts/2023/12/21/drivelm-driving-with-graph-visual-question-answering.html) connects questions across perception, prediction, and planning; G-MARK structures the object evidence those questions would depend on. [LangCoop](/paper%20shorts/2025/04/18/langcoop-collaborative-driving-with-language.html) compresses collaboration into generated language; G-MARK sends a compact typed record that retains which agent supported each claim.

### Evidence-sensitive tasks improve; long-horizon planning does not

The evaluation uses the official V2V4Real-derived V2V-GoT-QA split: about 110,000 training questions and 31,000 validation questions. The primary comparison uses V2V-GoT's reported results. G-MARK improves the tasks that can directly query provenance, visibility, and object state, while future-trajectory forecasting remains slightly behind.

| V2V-GoT-QA task | G-MARK | V2V-GoT | Relative change |
| --- | ---: | ---: | ---: |
| Occluding objects, F1@0.5 m ↑ | 0.428 | 0.301 | +42.2% |
| Invisible objects, F1@0.5 m ↑ | 0.494 | 0.440 | +12.3% |
| Object motion, average L2 ↓ | 3.822 | 7.610 | +49.8% |
| Control settings, action L1 ↓ | 0.076 | 0.088 | +13.1% |
| Future trajectory, average L2 ↓ | 2.710 | 2.620 | -3.4% |

The ablations identify which structure matters. Removing partner evidence drives invisible-object F1 from 0.494 to 0.000. Removing provenance lowers that F1 to 0.396 and doubles control error from 0.076 to 0.152. Replacing the graph with flat object features also hurts both tasks, though less sharply. These controls support provenance-aware cooperative evidence; they do not show that a knowledge graph is the only representation capable of storing it.

G-MARK reports a 0.0159 MB structured payload per sample, compared with roughly 0.4 MB for feature-fusion and language-mediated baselines. Its future-trajectory point is therefore a useful bandwidth trade: near-parity rather than best error, at much lower reported communication. The CPU task solver adds less than 1.4 ms per sample, but those timings exclude upstream perception; temporal tasks still take 32.4–33.4 ms mainly because previous-frame context must be loaded.

## High-Level Takeaways

- G-MARK informs whether cooperative driving should transmit one fused state or preserve the evidence behind that state. Its atomic unit is an agent-attributed object observation, which can remain weak until a task decides that partner-only evidence matters.
- The strongest causal result is the provenance ablation, not the headline benchmark average. Partner evidence is necessary for hidden-object discovery, and source attribution materially changes both retrieval and control.
- The 25.6× communication result covers serialized structured evidence, not an end-to-end V2V protocol with packet loss, delay, quantization, synchronization, or security overhead. The perception stack is also outside the measured system.
- A matched alternative should give a flat set encoder the same provenance, visibility, uncertainty, and path features under the same byte budget. Reject the graph-specific claim if that baseline matches accuracy and traceability, or if the gains vanish in closed-loop planning when upstream detections are noisy.
