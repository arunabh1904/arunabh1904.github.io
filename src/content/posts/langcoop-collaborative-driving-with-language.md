---
title: 'LangCoop: Collaborative Driving with Language'
date: '2025-04-18T02:03:14.000Z'
section: paper-shorts
postSlug: langcoop-collaborative-driving-with-language
legacyPath: /paper shorts/2025/04/18/langcoop-collaborative-driving-with-language.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – LangCoop: Collaborative Driving with Language"
---
## 2025 – LangCoop

**arXiv:** [2504.13406](https://arxiv.org/abs/2504.13406)

**Project and code:** [LangCoop](https://xiangbogaobarry.github.io/LangCoop/)

## Summary

> LangCoop makes natural language the communication medium between collaborating vehicles. Its $M^3$CoT component structures zero-shot vision-language reasoning, while LangPack turns selected information into concise language messages. In CARLA, the paper reports up to a 96% reduction in communication bandwidth, with messages under 2 KB, while retaining competitive closed-loop driving performance. The evaluation does not inject packet delay, loss, hallucinated objects, or an adversarially helpful-looking instruction, so the result establishes a semantic communication trade-off rather than a safety channel.

## Core Insights

The paper changes what travels across the vehicle-to-vehicle link. Rather than transmitting raw images or dense learned features, one agent packages a language description intended to preserve decision-relevant evidence for another. This is attractive for bandwidth and heterogeneous sensor stacks because language is compact and semantically structured. It also gives the receiver a lossy, generated representation whose omissions and ambiguities can change a maneuver.

### Four prompts before one packet

The Mixture Model Modular Chain-of-thought ($M^3$CoT) module decomposes scene understanding into four prompts: driving-scene description, interactive-object description, navigation-goal interpretation, and future-intent description. The first two emphasize visual recognition; the latter two turn that recognition into a goal and an intended maneuver. LangPack then combines the four outputs with agent metadata such as location, velocity, and acceleration.

The modularity is more than a naming choice. Different stages can use different LVLMs, so a stronger model can be reserved for the reasoning step that actually needs it. The experiments replace GPT-4o-mini in the scene/object stages with Gemini-2.0 or Qwen-2.5 and Llama-3.2 in the goal/intent stages while retaining GPT-4o-mini for driving-signal generation. This creates a practical cost knob, though it also makes the packet depend on several model prompts and their failure modes.

![LangCoop modular reasoning and language packet pipeline](/assets/images/langcoop-collaborative-driving-with-language-source-figure-1.webp)
*Fig 1: Each CAV turns a front-view observation into scene, object, goal, and intent descriptions, packages them with metadata, exchanges the result, and predicts a driving signal from local and received information. | source: [LangCoop: Collaborative Driving with Language, Figure 1](https://arxiv.org/abs/2504.13406)*

### LangPack is a semantic packet with a control contract

A LangPack contains the sender’s metadata, scene description, object description, navigation goal, and future intent. On receipt, a vehicle transforms the other agent’s coordinates and aligns the information in time before combining it with its own front-view perception. The receiver can then ask the LVLM for one of three output forms: discrete waypoints, a continuous trajectory parameterized by speed and curvature, or direct throttle/brake/steer controls.

The unit of communication is therefore not simply a caption. It is a packet that says what another vehicle sees, which objects matter, where its goal lies, and how it intends to move. In the paper’s example, CAV 1 decides to slow down at an intersection and communicates that intent; CAV 2 changes from continuing forward to slowing down. The receiver still sees its own front-view RGB image—the BEV panels in Figure 2 are visualization, not an additional sensor input—so the language message acts as a social prior over the local control decision rather than a replacement for perception.

![LangCoop communicated intent changing a following vehicle’s behavior](/assets/images/langcoop-collaborative-driving-with-language-source-figure-2.webp)
*Fig 2: At two timestamps, the sender’s intent changes to “slow down” near an intersection and the following vehicle’s planned behavior adapts while both retain local BEV and front-view context. | source: [LangCoop: Collaborative Driving with Language, Figure 2](https://arxiv.org/abs/2504.13406)*

### Continuous trajectories beat zero-shot coordinates

The signal-format ablation in Table 2 exposes an important interface constraint. Discrete trajectories score only 5.0/1.3 driving score for Vehicles 1/2 with 23.1%/19.4% route completion. Continuous trajectories reach 33.1/48.8 and 74.9%/90.3%; direct control reaches 33.7/18.1 and 89.0%/70.2%. The authors use continuous trajectories as the default because an LVLM is poor at emitting a long sequence of smooth, dynamically feasible coordinate pairs, while speed and curvature map more naturally into the vehicle controller.

This choice also separates communication from actuation. LangPack carries semantic context, but the receiver’s output still passes through a trajectory or control interface with physical constraints. A language packet can help a vehicle decide to slow or yield without requiring the LVLM to solve numerical waypoint interpolation from scratch.

### The bandwidth comparison is a trade-off, not a universal win

The CARLA evaluation uses ten Town05 scenarios, two connected vehicles, traffic-manager vehicles, pedestrians, and cyclists, front RGB cameras at 800×600, and a simulated 200 m communication range. The default setup in Section 4.1 uses GPT-4o-mini, concise $M^3$CoT, continuous trajectories, and both JPEG front-view images and LangPack. Table 4 then compares no collaboration, JPEG image sharing, LangPack-only sharing, and combined image+LangPack. With LangPack alone, Vehicle 1 scores 35.1 DS/71.6% route completion and Vehicle 2 scores 42.8/80.1 while transmitting 1.8 KB on average. JPEG image sharing uses 43.1 KB and gives 15.3/38.9 and 31.3/60.7. Image+LangPack uses 44.9 KB and reaches 33.1/74.9 and 48.8/90.3.

The 1.8 KB packet is about a 96% reduction from the 43.1 KB image packet, and it improves the reported scores over image-only in this setup. The 48.8/90.3 result belongs to the 44.9 KB image+LangPack row; the packet-only row is 42.8/80.1, so the best driving score and the 96% bandwidth reduction must not be combined. Adding the image back helps Vehicle 2, which shows that the language packet does not preserve every visual detail. The non-collaborative baseline is 13.5/33.1 and 11.35/29.44 for the two vehicles, so the comparison also contains a collaboration benefit. The clean conclusion is a Pareto point: compact semantic communication performs well for these scenes, while raw images still provide complementary evidence at much higher bandwidth.

### Prompt length, model choice, and heterogeneity matter

Table 3 compares naive prompting, standard CoT, and concise CoT. Naive prompting collapses to 2.7/0.7 driving score for the two vehicles. Standard CoT reaches 37.0/41.1 with 85.2%/80.3% route completion and a 105.2 s reported time; concise CoT reaches 33.1/48.8 and 74.9%/90.3% in 124.6 s. The paper’s TC metric measures time until route completion or terminal failure, so these numbers are route outcomes rather than model-inference latency. The preferred concise prompt is therefore not simply “more reasoning is better”: it trades Vehicle 1’s score and route completion for Vehicle 2’s higher score while shortening the generated reasoning.

The zero-shot LVLM comparison in Table 5 is highly model-dependent. GPT-4o gives 41.3/47.7 DS for the two vehicles, Claude-3.7 gives 32.0/72.1, and GPT-4o-mini gives 33.1/48.8; Gemini-2.0, Qwen-2.5, and Llama-3.2 are lower in the tested pairing. Gemini has the lowest reported TC at 46.5 s, but TC ends at route completion or terminal failure, so this is not necessarily faster successful driving and is never a model-inference latency measurement. The authors associate the short time with more aggressive behavior. Table 6 shows modular replacements remain competitive: the GPT-4o-mini-only setup reaches 33.1/48.8, Experiment 6.A 31.4/37.2, and Experiment 6.B 35.2/42.1.

Table 7 tests heterogeneous pairs. Pairing GPT-4o-mini with Gemini-2.0 improves the two vehicles from non-collaborative 18.2/12.6 to 59.1/45.3 DS; pairing it with Llama-3.2 improves 16.7/11.5 to 51.9/12.6. The Llama-3.2 vehicle gains only 1.1 DS and its route completion falls from 51.0% to 40.1%, showing that “collaboration helps both agents” can hide asymmetric benefits. Sensor and model heterogeneity is supported in this narrow simulation, but the result does not say that a weak receiver can safely trust every strong sender.

### Deployment boundaries

Language makes the packet human-readable, but readability is not integrity. The paper does not report packet delay, loss, malformed messages, decoding latency, message-length distributions beyond the average package size, or a detector for incorrect object descriptions and intents. It also uses zero-shot commercial and open LVLMs whose outputs may vary with prompt wording; the paper explicitly observes prompt sensitivity. A safety evaluation should replay the same scenes with delayed, dropped, truncated, and confidently false packets, then measure whether the receiver falls back to local perception or executes the communicated error. Equal-bandwidth feature and structured-token baselines would also clarify how much of the gain comes from language itself versus a better selected representation.

## High-Level Takeaways

- LangCoop uses $M^3$CoT to separate scene, object, goal, and intent reasoning, then sends a structured LangPack with agent metadata.
- In ten CARLA scenarios, 1.8 KB LangPack messages outperform 43.1 KB JPEG sharing on the reported routes; adding images improves some cases at 44.9 KB.
- Continuous trajectories outperform zero-shot discrete coordinates, so language supplies semantic context while the controller still handles physical actuation.
- Model choice, prompt length, asymmetric heterogeneous-agent gains, and untested packet failures define the evidence boundary for deployment.
