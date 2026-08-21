---
postSlug: omni-model-pretraining-decisions
sourceSha256: 8f8d862b580916df4f4cf917d4e4e0fe605fd5182b032637c790bf86b3ba9e2b
---

# Pre-Training for Robotics

Pre-training for robotics tries to give a policy useful priors before it has enough experience in one robot, task, or deployment. The promise is transfer: learn semantics, geometry, time, interaction, and action from several data sources, then reuse those priors when robot data is scarce.

A robot can inherit the word drawer from the internet. It can learn what drawers look like, where handles usually are, and which instruction refers to which object. The internet does not tell it how a sticky drawer feels, how far a particular arm reaches, what force closes its gripper, or how to recover after the object slips.

That gap is what makes robotics pretraining exciting to me, and easy to misunderstand. Internet data supplies broad semantics. Human video supplies temporal and interaction structure. Robot trajectories supply actions, contact, embodiment, and recovery. These sources are valuable because they are different. Putting them inside one large model does not erase those differences.

The central question is which priors should transfer into control, through which parameters and interfaces, and what evidence would prove that the transfer is real. A useful policy needs semantic, geometric, temporal, dynamics, affordance, and motor priors. Yet many recipes begin with years of visual-language pretraining and a randomly initialized action path. The most deployment-critical interface starts from scratch on the scarcest data.

## Start with the transfer contract

Multimodal, generalist, and foundation model describe families, not deployment capabilities. Before choosing an architecture, write the transfer contract as observable behavior. What must work with no robot-specific data? What may use a small target set? Which axes are held out: objects, scenes, task compositions, language, embodiment, recovery, or horizon? What deadline is non-negotiable? Which failure kills the program?

The contract should report zero-shot behavior, the slope of few-shot adaptation, latency, and recovery. A policy that reaches seventy percent after fifty target demonstrations can be more valuable than one that reaches seventy-five only after five thousand. For pretraining, the decisive metric is often transfer per robot-hour, not final success after unlimited adaptation.

## The pretraining ladder

Robotics pretraining is a ladder of increasingly physical supervision. Image-text data teaches semantics and relations but not contact or action consequences. Human video adds persistence and interaction while lacking robot commands. UMI capture and retargeted human motion add action-like structure but not target dynamics. In-domain robot video adds viewpoint and morphology. Action-labeled trajectories connect commands to transitions. Failures and interventions expose boundary states. Simulation supplies controlled diversity and counterfactuals but misses parts of real contact and appearance.

Each rung reduces a different uncertainty. Do not ask a dataset to supervise a variable it cannot identify. Passive video may learn motion regularities without revealing whether motion came from the robot, a human, gravity, or an unseen event. A caption can say open the drawer without specifying frame, impedance, force, or timing.

The important new bridge is abundant human interaction converted into executable priors. Latent-action methods, proprioceptive codebooks, UMI capture, and egocentric retargeting all make the same bet: a small amount of robot data can map broad human motion structure into commands. The test is transfer across held-out tasks, scenes, and embodiments, not whether a reconstructed clip looks plausible.

## Robot data is structured, correlated, and attached to hardware

Robot trajectories contain the operator's habits, controller smoothing, camera calibration, reset procedure, safety limits, and whatever state happened to be logged. More trajectories can mean more tasks, scenes, embodiments, operators, failures, or merely repetitions. Those axes do not buy the same transfer.

Open X-Embodiment, Octo, OpenVLA, DROID, Xiaomi-Robotics-1, and RoboCat explore different scaling loops: pooled embodiments, environmental diversity, human-operated capture, automatic labeling, target adaptation, and self-collected experience. Episode count alone hides the distinction.

Cross-embodiment learning needs an interface schema. Record coordinate frame, units, controller mode, frequency, horizon, joint topology, gripper semantics, and missing sensors. Padding actions and mapping every axis into the same numeric range does not create shared physics. And an episode is not an independent sample: sliding a window through one maneuver can create thousands of targets without creating thousands of new decisions.

## Representations must preserve what control needs

Recognition rewards invariance to lighting, texture, and background. Control needs sensitivity to pose, gripper offset, motion, and contact. The useful representation is invariant to nuisance appearance but equivariant to task-relevant geometry and state.

R3M, Voltron, VC-1, Theia, and dense patch-policy work show why general image quality is not enough. Measure semantics, keypoints, pose sensitivity, temporal correspondence, contact-state separability, robustness, and few-shot transfer under frozen, partial, and full tuning. A frozen encoder tests whether the prior already contains the information. Fine-tuning tests whether the initialization adapts quickly. Those are different claims.

A vision-language encoder can preserve enough semantics to answer a question while discarding the edge, depth, or motion needed for control. The connector cannot recover removed evidence. Test resolution, visual-token count, temporal sampling, and encoder adaptation before sweeping elaborate fusion modules.

## Representation, world-model, and policy pretraining are different targets
<!-- covers: Video prediction is not automatically a world model | World-action models expose where foresight enters control | Policies and world models scale differently -->

A representation model maps observation history and language into state. A world model predicts how that state changes under an action. A policy chooses the action. They may share a backbone, but they supervise different conditional distributions.

Plausible video is not automatically a world model. A planning model must preserve action-conditioned consequences, identity, contact, irreversible changes, and useful rollouts. Test interventional consistency, counterfactual action ranking, state persistence, contact fidelity, rollout stability, and actual planning value. Better-looking videos are not the relevant ablation.

World-action models make the interface explicit. Foresight may generate a visual subgoal before inverse dynamics, condition an action head through predictive features, share a model with actions, or disappear after acting as an auxiliary training loss. Joint future and action generation is promising, but it does not prove that the action head uses the predicted consequence. Stop, scramble, or remove the future path and measure recovery and action ranking.

Policies and world models also scale differently. Behavior cloning is limited by demonstrated actions. A world model is limited by state-action coverage and long-horizon consequence accuracy. One scaling exponent should not be assumed to govern both, or to predict real-robot success.

## Share semantics; specialize interfaces
<!-- covers: The shared variable matters more than the shared decoder | Pretrain the motor path, not only the semantic trunk | Measure gradient agreement instead of debating fusion abstractly -->

One model can mean one token stream, a shared trunk with embodiment-specific stems and heads, a pretrained VLM with a continuous action expert, or semantic reasoning layered above reactive control. The useful rule is to share parameters that should learn common invariances and specialize interfaces whose units, bandwidth, geometry, or deadlines differ.

Language is a strong interface for task semantics and a poor native clock for high-frequency control. The variable shared across robots matters more than whether the decoder is shared. Camera-frame tool motion, end-effector traces, object-centric motion, language descriptions, and learned latent actions make different bets about metric fidelity and transfer. The bridge used during pretraining need not be the native command served at execution.

The motor path also deserves a prior. Compare random initialization against within- and cross-embodiment action pretraining under matched data. A lower reconstruction loss is not enough; the prior should improve convergence, closed-loop success, perturbation response, or few-shot adaptation.

Finally, replace abstract arguments about early and late fusion with gradient evidence. For every objective, measure gradient norm, cosine similarity, update norm by module, and downstream capability. A lower joint loss does not mean semantics, state, action, or recovery improved.

## A dataloader share is not a learning share

Text, images, video, and robot trajectories expand into different numbers of predicted units and follow different compute paths. One sampled example from each modality does not contribute equal targets, FLOPs, or parameter updates. Overlapping windows add another distortion by repeating one decision many times.

Keep five ledgers together: sampled examples, predicted units, consumed compute, update norm by objective and module, and effective independent decisions. Normalize losses per target, per sequence, per FLOP, and per independent scene-task-embodiment unit. A mixture is balanced only relative to a capability objective. It is not balanced because the percentages sum to one hundred.

## The action interface is four decisions, not one
<!-- covers: 1. Coordinate system | 2. Temporal basis | 3. Conditional action distribution | 4. Who owns action generation? -->

Discrete versus continuous compresses four choices. First is physical space: joints, Cartesian pose, velocity, torque, impedance, or a skill consumed by another controller. Second is temporal basis: one action, a fixed chunk, a variable skill, or coefficients that reconstruct a trajectory. Longer chunks improve coherence and throughput but commit open loop; receding-horizon execution pays more inference to recover feedback.

Third is the conditional distribution. Regression is fast but averages incompatible strategies. Autoregressive tokens offer explicit likelihood while adding quantization and sequential latency. Mixtures represent modes. Diffusion and flow model continuous multimodality with an iterative or specialist serving path.

Fourth is ownership. The language decoder, a dedicated head, an embodiment expert, a flow module, or a controller hierarchy can emit the action. This choice is separate from tokenization. Every result should state physical frame, units, horizon, distribution, generator, execution policy, and end-to-end deadline. Otherwise, two continuous-action VLAs may solve different problems.

## Data mixtures and curricula

A robotics corpus is a sampling policy over sources, embodiments, tasks, quality, sequence length, success, and training stage. There is no context-free optimal percentage of text, video, and robot data.

A useful curriculum often follows the causal ladder: semantics, temporal state, embodied dynamics, behavior, then target embodiment with failures and recoveries. Staging can reduce optimization difficulty and can also cause forgetting. Compare it with an equal-compute interleaved run, and rerun retention tests after every transition.

Rehearsal should follow capability regressions. If behavior cloning damages open-vocabulary grounding, retain semantic data. If web training makes state insensitive to pose, add metric, temporal, or robot supervision. Prompt form is also part of the curriculum: a description of what happened is not the same target as an imperative command. The expensive decision is whether one candidate mixture will still win at target scale, so report uncertainty on that ranking.

## Scaling robotics data

Robotics has several scales: episodes, control steps, unique tasks and scenes, embodiments, operators, failures, active parameters, training compute, collection hours, and target adaptation hours. The scarce axis is often diversity, not raw steps.

Every scaling claim should name its fitted range, fixed architecture and interface, measured capability, diversity assumptions, residuals, uncertainty, and the experiment that would reverse the recommendation. The bottleneck can move from model capacity to unique robot data, action distortion, feedback, or latency.

Precommit kill criteria. Abandon one token stream if action latency or quantization grows faster than transfer. Abandon a shared trunk if smaller embodiments regress. Abandon a continuous expert if it adds no robustness at matched compute. Abandon a world model if interventions do not change futures or planning. Reduce internet data if semantic gains damage metric state or target adaptation. That is how paper reading becomes capital allocation.

## Proxy runs before the large run
<!-- covers: 1. Representation sweep | 2. Action-interface sweep | 3. Interaction sweep | 4. Scaling sweep -->

The proxy program should test decisions likely to reverse the architecture. Sweep representation choices such as dense versus pooled features, resolution, temporal context, adaptation, and metric state. Sweep action coordinates, horizons, tokenizers, distributions, and motor initialization. Sweep mixtures, prompt ontology, normalization, curriculum, capacity, and embodiment routing. Then train several model and data sizes and fit separate curves for grounding, transfer, prediction, policy success, and latency.

The output should be decision artifacts: an architecture memo, embodiment schema, objective-interaction matrix, scaling model with uncertainty, throughput model, capability dashboard, kill criteria, and failure playbook. A cheap simulator metric matters only after showing that its model ranking predicts the expensive real-robot decision.

## Evaluation must match the transfer claim

Random trajectory splits often leak scenes, objects, operators, and adjacent windows. Hold out explicit cells across visual appearance, semantics, task composition, perturbations, embodiment, horizon, and hardware or collection system.

For every held-out axis, report zero-shot success, few-shot curves, recovery after perturbation, latency, variance, and regression on supported tasks. Few-shot curves distinguish a broad prior from a merely large policy. Pretraining should improve zero-shot behavior, reduce target demonstrations needed for a given success rate, or improve the asymptote under the same data. If none moves, transfer did not occur where claimed.

## The training system is part of the experiment

Large multimodal runs need more than peak throughput. Detect loss spikes by source, slow capability drift, modality and embodiment domination, corrupt shards, coordinate mismatches, stalls, and evaluation regressions. Automatic responses should quarantine data, roll back, rebalance, or block promotion while preserving evidence.

Every checkpoint must bind weights to optimizer, scheduler, data cursor, mixture policy, tokenizers, action schemas, code revision, and evaluation configuration. A weight file without that lineage is not a recoverable experiment.

## A practical reading path

Read by decision and produce an artifact. Study CLIP, R3M, VC-1, Theia, MM1, and dense patch policies to locate where semantic, spatial, and temporal priors come from. Study Open X, Octo, HPT, camera-frame actions, language-action pretraining, and behavior-aligned representations to decide what transfers across robots. Study Behavior Transformer, ACT, Diffusion Policy, FAST, Pi-zero, RDT, and motor-prior work to write the action interface and latency benchmark. Study Genie, DINO-WM, V-JEPA, and world-action models to compare planning with direct policy learning. Finish with robotics scaling, mixture, proxy-simulation, and held-out generalization work to write the go or no-go memo.

Then move to post-training. Pretraining decides what the policy can represent and which behaviors are nearby. Deployment reveals which of those nearby behaviors are useful.

## A testable thesis

The strongest robot model will probably share the parameters that benefit from transfer and specialize the interfaces where geometry, bandwidth, time, and control differ. My starting hypothesis is a strong vision-language semantic trunk; metric and temporally persistent state; explicit action-conditioned dynamics when planning is required; a behavior-aligned cross-embodiment bridge; embodiment-aware inputs and continuous action experts; and an explicitly pretrained motor path or evidence that it is not the bottleneck.

The VLM can know what a drawer is. A temporal model can track that it moved. A world model must preserve what the robot's action caused. A policy must choose and execute the action before its deadline. Pretraining succeeds only when those capabilities transfer without erasing the distinctions physical control requires.

A literature review should end as an experiment plan. It should say what one training unit is, where information is compressed, how data becomes gradient, which prior should transfer, what matched control is missing, and which result would reverse the architecture. Anything less is a tour of papers, not a pretraining strategy.

## References

The spoken version skips the reference list. The complete linked bibliography remains in the written post.
