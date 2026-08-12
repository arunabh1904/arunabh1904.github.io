---
title: "Code Is Cheap. Understanding Isn't."
date: '2026-08-12T04:00:00.000Z'
section: blog
postSlug: code-is-cheap-understanding-isnt
legacyPath: /blog/2026/08/12/code-is-cheap-understanding-isnt.html
tags:
  - AI
  - Software Engineering
  - Career
topics:
  - language-systems
summary: >-
  What remains valuable when agents make implementation abundant: context,
  judgment, task decomposition, efficient orchestration, and ownership.
---

# Code Is Cheap. Understanding Isn't.

Software engineering has always been about building things. Code was the tool we happened to use.

That sounds obvious, but software people are unusually easy to seduce by the artifact itself: beautiful abstractions, elegant APIs, clever type systems, perfectly factored code. I love that stuff too. Craft matters. But nobody hired us because the source looked beautiful in a diff. We were hired to build something useful.

There is a reason needless refactoring has always been frowned upon. There is a reason nobody is nostalgic for punch cards. The interface changes; the work remains. Right now, that interface is changing very quickly.

## From reasoning to action

[Junyang Lin describes the transition](https://justinlin610.github.io/blog/from-reasoning-to-agentic-thinking/) as a move from “reasoning thinking” to “agentic thinking.” The first generation of reasoning models taught us to treat intelligence as inference-time compute: give a model a harder problem, let it deliberate for longer, and hope for a better answer. An agentic system changes the objective. The model is no longer thinking only to answer. It is thinking to act, observe what happened, and decide what to do next.

A coding agent can search, inspect a repository, run a test, revise its plan, call another tool, delegate an independent investigation, and continue. The useful unit is therefore not the model alone. It is the model plus its harness, tools, environment, context, evaluators, and feedback loop. Lin makes the same systems point about training: once a policy interacts with terminals, browsers, sandboxes, and other stateful environments, the environment becomes part of the capability stack rather than a passive verifier.

That changes what it means to be good at using AI for software. Most of us have not calibrated yet. We are still impressed by how much code an agent can produce when code is rapidly becoming the least scarce part of the system.

The scarce part is deciding what should exist.

[![An engineer routes distinct tasks to small specialist robots while reserving one large machine for the hardest work.](/assets/images/agentic-engineering-orchestration.png)](/assets/images/agentic-engineering-orchestration.png)
_Explanatory illustration: an agentic engineering system is a routing problem. Give bounded work to specialized workers, isolate their context, and reserve the most expensive capability for the decisions that need it. Original image generated with OpenAI ImageGen._

## Context becomes the scarce input

As implementation gets cheaper, context gets more valuable. An agent might remember more Kubernetes details than I do. It might know more C++ edge cases. It might produce a CUDA kernel in minutes that would have taken me half a day. What it does not automatically know is why a particular system looks the way it does.

Why is that ugly workaround still there? Which customer depends on that behavior? Why did the team reject the cleaner abstraction six months ago? Which invariant cannot break? Which service disappears next quarter? What is the business trying to accomplish?

That is context, and context engineering is not the same thing as putting more text into a prompt. I increasingly dislike giant base-level instruction files that every agent receives forever because one paragraph was useful for one task six weeks ago. More context is not necessarily better context. Irrelevant context pollutes reasoning in the same way that unnecessary shared state pollutes a software system.

The real skill is deciding what an agent needs to know for this task, right now. Sometimes good context engineering means adding a design decision, production invariant, or failure trace. Sometimes it means deleting an obsolete instruction. The goal is not maximum context. It is the smallest context that preserves the decisions the agent cannot safely rediscover.

## Do not become a slop cannon

A new developer archetype is emerging: the slop cannon.

Give it a problem and 8,000 lines of code appear. Something fails, so another agent rewrites the implementation. The pull request becomes unwieldy, so a third agent “cleans it up.” More abstractions appear. More instructions get added. More code gets generated to manage the code that was just generated.

Everyone has been extremely busy. Then somebody has to own it.

[![A cannon creates a tangled heap while an engineer builds a small bridge that carries a real load.](/assets/images/agentic-engineering-slop-cannon.png)](/assets/images/agentic-engineering-slop-cannon.png)
_Explanatory illustration: output volume is not engineering progress. The pile is easy to create; the small structure that carries a real load is the result that matters. Original image generated with OpenAI ImageGen._

There is a large difference between using a model to extend a deep understanding of your tools and using it to generate a system you do not understand. If I would be uncomfortable maintaining an unfamiliar codebase handed to me by a developer I have never met, I should be equally uncomfortable maintaining thousands of generated lines whose decisions I never examined.

Technical debt generated in thirty seconds is still technical debt. A needless abstraction written by a frontier model is still a needless abstraction. A giant diff is not evidence of productivity.

The adult in the room asks a more boring question: why does this thing need to exist? Sometimes the most successful agentic coding session should end with less code than it started with.

## Match intelligence to the task

We increasingly reach for the biggest model because it feels safer. Surely more intelligence cannot hurt.

Except sometimes I ask a giant model to change a small utility and it spends five minutes exploring half the repository, discovers three adjacent architectural sins, proposes a new abstraction hierarchy, and consumes a heroic number of tokens solving a problem that was fundamentally small. I think of this as the big-model smell.

Large models can be extraordinary. [Claude Fable 5](https://www.anthropic.com/claude/fable), for example, is explicitly positioned for ambitious, long-running coding and knowledge-work tasks. When I genuinely need difficult architectural reasoning, deep synthesis, or an insight I probably would not have reached myself, that extra capability can justify its cost.

But I do not need Fable to do my job. Much of engineering does not require maximum intelligence.

My workflow is increasingly heterogeneous. A bounded task can often go to a smaller or faster model after the hard ambiguity has been removed. [GLM-5.2](https://z.ai/blog/glm-5.2) gives me another option for sustained coding work and explicit effort control. [Claude Code](https://www.anthropic.com/claude-code) is useful for constrained asynchronous jobs. [Cursor](https://www.cursor.com/) works well when I want to remain close to the implementation. [Grok 4.5](https://x.ai/news/grok-4-5) is another day-to-day coding option. In OpenAI's current GPT-5.6 line, [Terra and Sol occupy deliberately different capability and cost positions](https://openai.com/index/previewing-gpt-5-6-sol/), and reasoning effort adds another routing dimension.

Those exact names will age. That is fine. The principle will not: match the intelligence to the task.

Use the expensive model when ambiguity, consequence, or deep reasoning justifies it. Use faster and cheaper models for execution after the problem has been reduced to something mechanical. Use subagents when work is genuinely independent. Do not send a genius to alphabetize a list.

This is not only thrift. It is architecture.

## Treat tokens as a system resource

We already reason about CPU, memory, bandwidth, latency, storage, GPU utilization, and cloud spend. Intelligence cost belongs in the same design conversation. Tokens are a resource.

Good agentic engineering does not give every worker the full history of the repository. It does not pay a frontier model to rediscover what another investigation already established. It keeps context boundaries clean, routes easy work to cheaper models, and escalates hard decisions. It distinguishes an extra million tokens buying useful thought from an extra million tokens buying beautifully articulated wandering.

The agent stack may end up looking familiar to anyone who has designed distributed systems: specialized components, constrained interfaces, careful information flow, isolation where useful, observability, retries, escalation paths, and expensive resources reserved for the places that require them. The point is not to minimize token use at all costs. The point is to spend intelligence deliberately.

A company will eventually notice the difference between an engineer who needs enormous amounts of frontier-model inference for every task and one who produces the same or better outcome with a well-designed hierarchy of models and agents. Efficiency will matter because it exposes the quality of the decomposition underneath the bill.

## Parallelism changes the engineer's job

Humans are mostly single-threaded. Agents do not have to be.

Instead of researching, designing, implementing, testing, and debugging in a serial loop, one engineer can run several independent investigations at once. One agent can trace an unfamiliar call path while another studies an upstream dependency. A third can implement a bounded component. A fourth can test assumptions against telemetry or review the resulting diff.

This is where the agent manager becomes more than a metaphor. The engineer moves up one level: understand the problem, decompose it, decide what can safely run in parallel, give each agent a purpose, protect its context, route hard decisions upward, inspect the output, integrate the pieces, and kill bad branches early.

Lin's argument goes further: the important system is increasingly the orchestrator, specialized agents, tools, environments, and feedback loops working together. [The Engine Shop](https://noumena.com/essays/the-engine-shop-part-3) develops a related metaphor around candidate implementations that are isolated, compared, repaired, pressure-tested, and rejected. The future is not only better models. It is better systems of models.

But that future hides an uncomfortable requirement. A manager who cannot recognize bad work is not managing anything.

## Keep learning load-bearing

AI makes it possible to accomplish things we do not understand. That is an incredible superpower. It is also a trap.

For most of programming history, doing the work and learning the work were inconveniently coupled. To fix a sufficiently difficult bug, I eventually had to understand something about why it existed. That coupling is weakening. I can modify a distributed system without understanding its consistency model, generate a CUDA kernel without learning much about memory access, or ship code in an unfamiliar framework without understanding the framework.

The task gets completed. The learning never happens.

In a 2026 conversation about the move from vibe coding to agentic engineering, [Andrej Karpathy returns to a useful distinction](https://www.youtube.com/watch?v=96jN2OCOfLs): thinking can be outsourced more readily than understanding. That distinction is the defense against passive competence. Do hard things sometimes. Read the implementation. Follow the stack trace yourself. Debug without immediately reaching for an agent. Go one abstraction lower. Learn enough that when a model proposes something stupid, some part of your brain feels the mismatch before the tests do.

The better agents become, the easier it becomes to stop learning. The less I learn, the worse I become at directing them. Learning is therefore not a nostalgic attachment to manual work; it is part of the control system.

## Some things still take time

AI can compress implementation time. It cannot compress experience in the same way.

In [“Some Things Just Take Time,”](https://lucumr.pocoo.org/2026/3/20/some-things-just-take-time/) Armin Ronacher uses an old tree to make the distinction concrete. A person can buy a sapling but cannot buy decades of growth. Software has equivalents: trust, taste, community, and the knowledge earned by operating a system after the excitement of creating it has disappeared.

I can generate ten architectures overnight. I cannot generate ten years of operating one. I cannot manufacture the judgment that comes from outages, migrations, failed launches, organizational mistakes, ugly compromises, and maintenance that continues long after the original authors move on.

There is something important in the friction itself. If a machine can generate code instantly, it becomes tempting to ask why review should take so long, why design needs discussion, or why deployment requires permission systems and rollout plans. Sometimes friction is waste. Sometimes friction is the mechanism by which people notice what they are about to do. Engineering judgment includes knowing the difference.

Speed is useful. Tenacity is different.

## What remains valuable

The amount of human labor required to produce software will probably fall. Pretending otherwise does not help anyone. If one engineer can direct five, ten, or twenty capable agents, the economics of engineering organizations change—perhaps unevenly and not all at once, but in a direction that is hard to ignore.

Imagine two engineers in that organization. One has become extraordinarily good at generating things. Every task invokes the largest model. Every problem becomes a giant agent run. The engineer produces enormous diffs, consumes enormous amounts of inference, and understands a shrinking fraction of what ships.

The other understands the system and the business. They decompose ambiguous problems, know when a small model will do, keep context clean, run independent work in parallel, delete aggressively, challenge the output, and take responsibility for what reaches production.

Which one do you retain?

The answer says a great deal about where the profession is going. The future does not belong to whoever can generate the most code. Code is becoming abundant. The durable skill is understanding what should be built, why it should exist, what context matters, and how to marshal increasingly capable machines to build it well.

Writing software was always about building. Code was a tool. Agents are tools too.

Do not become a slop cannon. Do not worship the largest model. Do not outsource your learning. Spend intelligence deliberately. Understand the system. Understand the business. Build things that deserve to survive.

Code is getting cheap. Understanding is not.
