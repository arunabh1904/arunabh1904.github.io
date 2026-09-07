---
title: "Can LVLMs Obtain a Driver's License?"
date: '2024-09-04T00:00:00.000Z'
section: paper-shorts
postSlug: can-lvlms-obtain-a-drivers-license-idkb
legacyPath: /paper shorts/2024/09/01/can-lvlms-obtain-a-drivers-license-idkb.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2024 – Can LVLMs Obtain a Driver's License?"
---

## Summary

> IDKB tests driving knowledge as something a model must study, apply, and transfer. The dataset contains 1,016,956 questions from 206 handbook documents, driving tests, and CARLA or Bench2Drive road-sign scenes, covering 15 countries, nine languages, and four vehicle types. Driving Test Data dominates at 84.0%, while handbook data is 5.0% and road data 11.1%; 15 LVLMs are evaluated on multiple-choice and open QA tasks. GPT-4o scores 0.64 overall and the best open model, XComposer2, scores 0.45. Fine-tuning helps, and adding IDKB to nuScenes planning reduces average L2 error from 1.13 m to 0.77 m and average collision rate from 0.61% to 0.37% in the reported UniAD-metric evaluation. That supports explicit rule knowledge, but it does not turn a written-test benchmark into a closed-loop driving license.

**arXiv:** [2409.02914](https://arxiv.org/abs/2409.02914)

## Core Insights

### IDKB follows the human path from theory to practice

The Intelligent Driving Knowledge Base combines three sources. Driving Handbook Data comes from 206 documents and 23,847 pages of laws, regulations, techniques, and defensive-driving guidance. Driving Test Data collects country- and vehicle-conditioned theory questions. Driving Road Data turns traffic-sign knowledge into visual scenes using CARLA and additional sign scenes from Bench2Drive. The simulated collection drove for about 20 hours and produced roughly 400,000 camera frames before filtering to 112,388 annotated road samples.

That progression is the paper’s central idea: a capable driving system needs more than object recognition. It needs to connect a rule to a sign, a country, a vehicle, and the action an ego vehicle should take. The data covers 15 countries and nine languages across car, truck, bus, and motorcycle categories. The knowledge taxonomy assigns 22.2% to laws and regulations, 38.6% to signs and signals, 22.0% to driving techniques, and 17.1% to defensive driving.

![IDKB examples from its three data sources](/assets/images/can-lvlms-obtain-a-drivers-license-idkb-source-figure-3.webp)
*Fig 1: IDKB combines handbook explanations, country-specific driving-test questions, and visual road-sign questions from its simulated and extracted road data. | source: [Can LVLMs Obtain a Driver’s License?, Figure 3](https://arxiv.org/abs/2409.02914)*

The final inventory is 1,016,956 questions: 50,501 handbook items, 854,067 driving-test items, and 112,388 road items. Multiple-choice questions make up 830,057 items, open QA 187,435, and questions with an explanation 66,645. The split uses all handbook data plus 90% of test and road data for training, reserving the remaining 10% of the latter sources for testing. That split is suitable for measuring the dataset’s training utility, but it is not a jurisdiction-held-out test.

### The score separates written rules from sign recognition

IDKB evaluates 15 representative LVLMs with the same prompts. Driving Test Data contains single-answer and multiple-answer MCQs, instruction-following checks, and open QA scored with ROUGE-1, ROUGE-L, and SEMScore. Driving Road Data uses MCQ and QA tasks around traffic signs. The overall IDKB score averages the test-data and road-data scores, with MCQ quantities providing the weighting; instruction following is reported separately rather than folded into the score.

![Performance of 15 LVLMs on IDKB](/assets/images/can-lvlms-obtain-a-drivers-license-idkb-paper-figure.png)
*Fig 2: Overall IDKB, Driving Test, and Driving Road scores show a consistent gap between proprietary models and most open models, while road-sign performance is often higher than written-test performance. | source: [Can LVLMs Obtain a Driver’s License?, Figure 1](https://arxiv.org/abs/2409.02914)*

GPT-4o reaches an IDKB score of 0.64 and Gemini-1.5-flash 0.58. XComposer2 is the strongest open model at 0.45; most open models fall around 0.35–0.40, with BLIP2 at 0.27 and VisualGLM at 0.29. The asymmetry is informative: models generally perform better on Driving Road Data than on Driving Test Data. Recognizing a traffic sign is narrower than answering a question about law, responsibility, vehicle control, or defensive driving.

The multi-answer task is also a useful stress test. Some models improve over single-answer questions, while others fail to output the complete set and receive no partial credit. Instruction following magnifies the same issue: proprietary models reach about 0.99 on the required answer format, while Yi-VL-6B reaches only 0.11. A low score can therefore reflect knowledge, output discipline, or both; the paper reports these components separately so they are not conflated.

![IDKB distribution by source, country, language, vehicle, and knowledge domain](/assets/images/can-lvlms-obtain-a-drivers-license-idkb-source-figure-4.webp)
*Fig 3: The dataset’s source, geographic, language, vehicle, and knowledge distributions make coverage visible instead of hiding it inside one million items. | source: [Can LVLMs Obtain a Driver’s License?, Figure 4](https://arxiv.org/abs/2409.02914)*

### Fine-tuning knowledge can influence planning, within an open-loop test

The authors fine-tune Qwen-VL-chat, MiniCPM-Llama3-V2.5, XComposer2, and DeepSeek-VL-7B on IDKB. The gains are largest on written-test data: MiniCPM’s Test Data Score rises from 0.23 to 0.46, while its Road Data Score rises from 0.59 to 0.77. The experiment supports the mechanism the dataset is designed to test: explicit legal and procedural knowledge can be injected into a model that already recognizes visual scenes.

For a downstream check, Qwen-VL-chat is fine-tuned either on nuScenes planning data alone or on nuScenes plus IDKB. In the paper’s UniAD-metric table, average trajectory L2 falls from 1.13 m to 0.77 m, and average collision rate falls from 0.61% to 0.37%. The one-second, two-second, and three-second values all improve in the combined condition. The paper describes this as a 32% L2 reduction and a 65% collision reduction, but the displayed average collision values correspond to roughly a 39% relative drop; the table is the safer reference.

This is still imitation-learning, open-loop validation on nuScenes. The model identifies a “ROAD WORK AHEAD” sign, explains that the vehicle should slow down, and outputs a trajectory, but the evaluation does not show behavior under interactive traffic or a change of jurisdiction. IDKB is most valuable as a knowledge and transfer benchmark: passing its questions is evidence that the model has learned explicit driving concepts, not that it can safely control a vehicle.

## High-Level Takeaways

- IDKB tests the missing layer between visual scene understanding and rule-conditioned driving judgment.
- Its source mixture matters: a model can recognize signs while remaining weak on laws, techniques, and defensive-driving questions.
- Fine-tuning improves both written-test knowledge and an open-loop planning experiment, but the planning gain should be read from the exact metric table and protocol.
- The decisive follow-up is jurisdiction-held-out, counterfactual rule-scene evaluation followed by closed-loop interaction.
