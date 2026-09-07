---
title: 'CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving'
date: '2024-08-19T09:53:49.000Z'
section: paper-shorts
postSlug: covla-comprehensive-vision-language-action-dataset-for-autonomous-driving
legacyPath: /paper shorts/2024/08/19/covla-comprehensive-vision-language-action-dataset-for-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2024 – CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving"
---
## 2024 – CoVLA

**arXiv:** [2408.10845](https://arxiv.org/abs/2408.10845)

**Project:** [CoVLA-AD](https://turingmotors.github.io/covla-ad/)

## Summary

> CoVLA is a dataset construction paper. It turns synchronized front-camera video, vehicle signals, GNSS/IMU, and radar into 10,000 real-world driving scenes with frame-level captions and future trajectories. The selected corpus contains 6 million frames, more than 80 hours of video, and 3-second action targets. Its CoVLA-Agent baseline shows that language and trajectory prediction are coupled: predicted captions give ADE/FDE of 0.955/2.239, while ground-truth captions give 0.814/1.655. The paper also documents hallucinated objects and left-right errors in its automatic captions, making provenance part of the result.

## Core Insights

### Sensor-derived facts anchor generated descriptions

CoVLA’s contribution is to make the training example multimodal at the same timestamp. Each scene is a 30-second front-facing video clip with a future trajectory, factual behavior labels, and richer free-form descriptions. The dataset is collected around Tokyo from a front camera, CAN bus, GNSS, IMU, and radar. The sensors matter because a caption alone can say that a car is slowing, while the synchronized signals can determine whether that statement is consistent with the actual path.

![Figure 2 from CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving](/assets/images/covla-comprehensive-vision-language-action-dataset-for-autonomous-driving-source-figure-2.webp)
*Fig 1: CoVLA first detects traffic lights and leading vehicles, estimates trajectories with sensor fusion, and uses rule-based text as factual context for VLM-generated behavior and reasoning captions. | source: [CoVLA, Figure 2](https://arxiv.org/abs/2408.10845)*

The pipeline is deliberately anchored before it becomes generative. A Kalman filter fuses GNSS and IMU to estimate the ego path; radar and the front camera supply leading-vehicle speed, acceleration, and relative position; OpenLenda-s1 detects traffic-light color and arrow direction. Rule-based captions encode speed, acceleration, curvature, leading-vehicle presence, and traffic-light state. A VideoLLaMA2-7B captioner then sees eight representative frames from each 60-frame, three-second window and adds road type, weather, potential risks, and other context. The rule text supplies sensor-grounded context for the VLM, without guaranteeing that its added interpretation is correct, and token probabilities are queried for attributes such as sunny, cloudy, or rainy rather than trusting a free-form sentence blindly.

![Figure 4(a) from CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving](/assets/images/covla-comprehensive-vision-language-action-dataset-for-autonomous-driving-source-figure-4.webp)
*Fig 2: Inverse-frequency scene sampling reduces the large near-zero-speed peak and makes the retained speed distribution cover more driving regimes. The companion panel in the paper shows the analogous steering-angle balancing. | source: [CoVLA, Figure 4](https://arxiv.org/abs/2408.10845)*

### Balanced sampling changes the driving distribution

Sampling is a modeling decision, not only a data-cleaning step. From more than 1,000 raw hours, the authors retain data recorded in driving gear, below 100 km/h, with continuous GNSS. They weight samples inversely to the joint empirical distribution of maximum absolute steering angle, maximum absolute acceleration, and turn signal, using additive smoothing $\delta=50$. The result is 10,000 diverse 30-second scenes, 6,000,000 frames, and 83.3 hours. Active turn signals occur in 16.11% of frames and traffic lights in 22.90%, so the corpus intentionally gives more probability to maneuvers that a random driving log would underrepresent.

The captioning trade-off is visible in the paper’s own error audit. Auto-captioning sometimes hallucinates objects such as a wooden fence, swaps left and right, or misunderstands Japanese traffic signs and landmarks. Because the captions are later used as conditioning text, these errors are not cosmetic: the language channel can redirect the predicted trajectory. CoVLA therefore provides a useful pattern for scalable supervision, but its generated text should be treated as a noisy sensor with a rule-based check, not as ground truth.

### Read the result together with its evaluation protocol

The baseline CoVLA-Agent uses Llama-2 7B, CLIP ViT-L at $224\times224$, an MLP embedding of ego speed, and special trajectory query tokens. It outputs ten $(x,y,z)$ coordinates relative to the current position over a three-second horizon. The model is trained jointly with cross-entropy for caption generation and mean-squared error for trajectory prediction, with equal loss weights. A 70/15/15 scene split produces 302,989 training samples, 64,153 validation samples, and 64,920 test samples after sampling at 2 Hz and retaining only frames with all 60 future coordinates.

| Caption condition | ADE ↓ | FDE ↓ |
| --- | ---: | ---: |
| Predicted caption | 0.955 | 2.239 |
| Ground-truth caption | 0.814 | 1.655 |

The gap diagnoses sensitivity to caption conditioning. The same trajectory head is more accurate with the dataset’s reference captions, which are automatically constructed using richer sensor and temporal information than the single-frame caption predictor receives. This is not a clean isolation of caption correctness, and “ground truth” here does not mean every caption was verified by a human. The qualitative cases show the same coupling: on a controlled intersection, a generated “turning right” caption produces a right-turn trajectory, while the rule-based “moving straight” caption produces a straight path. That is coherent behavior, but it also means that a fluent wrong caption can make the action wrong in a consistent way.

The error analysis is concentrated around motion words. Captions containing deceleration have mean ADE/FDE 2.236/5.458, left 2.037/5.009, and acceleration 1.826/4.790; turning and right are associated with lower errors on average. The authors attribute this to estimating intention from a single frame. A useful next evaluation would separate caption correctness from trajectory correctness on temporal windows, then report how much of the improvement comes from better scene language versus direct visual and sensor evidence.

## High-Level Takeaways

- CoVLA’s scalable unit is a synchronized video frame with both a future trajectory and language that describes the same driving event.
- Rule-based captions, sensor fusion, and VLM enrichment form a sensible provenance ladder, but the final captions still contain hallucinated objects, spatial swaps, and culture-specific recognition errors.
- The baseline’s 0.955/2.239 predicted-caption errors versus 0.814/1.655 with ground-truth captions quantify the cost of trusting generated language.
- The dataset’s diversity comes partly from inverse-frequency sampling; results should therefore report both the balanced sample distribution and performance on the unbalanced deployment distribution.
