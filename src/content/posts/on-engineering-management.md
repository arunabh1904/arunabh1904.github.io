---
title: On Engineering Management
date: '2026-08-20T04:00:00.000Z'
section: blog
blogGroup: essays
postSlug: on-engineering-management
legacyPath: /blog/2026/08/20/on-engineering-management.html
tags:
  - Engineering Management
  - Leadership
  - Career
summary: How judgment, evidence, trust, delegation, and visible progress turn engineering management into compounding team capability.
---

# On Engineering Management

An engineering manager’s job is to execute within constraints. The role includes coaching, architecture, hiring, planning, conflict resolution, performance management, communication, and process design. But those activities are mechanisms, not the final output. The output is a team that repeatedly identifies an important problem, makes a sound decision, and turns that decision into a working system. Kamil Sindi’s [Managers Playbook](https://github.com/ksindi/managers-playbook) is a resource I keep coming back to, and it has shaped parts of how I think about the job.

Execution without judgment is motion. Judgment without execution is theater. An engineering manager is responsible for bringing the two together.

I do not measure management by the number of meetings I attend, the decisions that pass through me, or how indispensable I appear. I measure it by whether the team makes better decisions, moves with greater conviction, and becomes more capable over time.

Management, at its best, is a compounding system. Good decisions create progress. Progress creates information. Information improves judgment. Better judgment produces greater autonomy, which allows the team to take on harder problems. My job is to keep that loop moving.

## Code is only as good as the problem it solves

I love writing code. Building things is still one of the best ways to understand them. But code is only as good as the problem it solves.

It is easy for engineers, especially strong engineers, to become attached to the elegance of a solution. We can spend weeks optimizing an architecture, cleaning up an abstraction, or adopting a new technique without confronting the more important question: does this matter?

Management keeps technical work connected to its purpose. What user problem are we solving? What capability does this unlock? What constraint are we removing? What becomes possible once the work is complete?

Not every technical decision needs an immediate revenue calculation. Foundational work, research, reliability, infrastructure, and technical debt all matter. But even foundational work needs a theory of impact. “This is technically interesting” is not, by itself, a strategy.

After roughly ten years of building systems, I have accumulated some battle scars. I have seen elegant ideas fail under production constraints, rushed systems become permanent systems, and six-month projects begin with assumptions that could have been disproven in a week. Those experiences are useful, but experience can also become a trap.

Battle scars should sharpen your questions, not fossilize your conclusions. Strong opinions, weakly held: have a point of view, explain it clearly, and update it quickly when reality disagrees.

Staying technical keeps those instincts honest. That might mean reviewing a difficult design, building a prototype, inspecting traces, fixing an annoying bug, or writing the first version of a document. It does not mean placing myself on the critical path of every important project.

Leading by example is different from leading through dependency. If I always take over the hardest problems, the team learns that important work eventually routes through me. The goal is not to prove that I can still be the strongest engineer in the room. It is to create more people who can solve problems I could not solve alone.

## Go slow to go fast

Engineering has two distinct modes: deciding where to go and moving in the chosen direction. Confusing them creates waste.

Before committing to an expensive direction, I want us to explore broadly. Read the literature. Understand previous attempts. Establish a baseline. Write down the assumptions. Examine the failure modes. Build the smallest prototype that can change our confidence.

Many teams spend thirty minutes framing a problem and six months implementing the answer. I would rather spend enough time making the problem legible, especially when the decision has a large blast radius.

This is what “go slow to go fast” means to me. It means matching the depth of investigation to the cost and reversibility of the decision. A small, reversible product change does not require weeks of consensus-building. Make a reasonable decision, instrument it, and learn. An architecture that will define the system for years deserves more care.

Expensive decisions deserve cheap evidence. A prototype, benchmark, ablation, trace, or small production experiment can turn an abstract disagreement into an empirical question. The manager is not building the entire system; the manager is creating enough evidence for the team to make a better decision.

Exploration still needs an exit condition. Research becomes procrastination when nobody knows what evidence would be sufficient to act. Before investigating, we should know what uncertainty we are trying to reduce, how much time we will spend, and what result would change the decision. Careful thinking should make execution more decisive, not postpone it.

## Argue before the decision, rebase after it

Before a decision, I want lots of voices. I want engineers to challenge the framing, question the data, expose missing assumptions, and propose alternatives. Silence is not alignment. It may simply mean that people do not believe disagreement is worth the cost.

Teams often fail in one of two directions. Some suppress disagreement in the name of speed, producing superficial alignment and delayed resistance. Others let discussion continue forever in the name of inclusion, until nobody knows who owns the decision. Inclusion does not give every person veto power. It gives relevant perspectives a genuine chance to change the team’s model of the problem.

> Before the decision, optimize for truth. After the decision, optimize for coordination.

Once we decide, everyone rebases onto the chosen direction. Nobody has to pretend they originally agreed, but they do have to stop fighting yesterday’s branch. Commitment means giving the selected direction its best honest chance and helping the team learn as quickly as possible.

Disagree and commit only works when commitment is earned. People are more willing to support a decision they opposed when they understand the reasoning, know their concerns were considered, and can see what evidence would cause us to revisit it. A decision should include the outcome we want, the tradeoffs we accepted, the owner, the success criteria, and the conditions for reconsideration.

Then we execute. A team that learns from reality will usually beat one that is still defending its forecast. Time in execution often beats an incredibly smart prior.

## Make progress tangible

Progress is not merely a reporting concern. It is a management primitive.

A prototype reveals whether an interface works. A benchmark exposes the real bottleneck. A demo uncovers integration problems. A customer interaction corrects assumptions. A production deployment teaches us things that no planning document can. Tangible progress creates information.

It also creates energy. Teams behave differently when they can see themselves moving. Collaboration becomes easier, harder ideas feel possible, and ambiguity becomes manageable because the team has evidence that it can reduce it.

This is why I break large efforts into meaningful milestones. The goal is not to manufacture deadlines or celebrate ticket closure. It is to create frequent contact with reality. For implementation, that may be a working vertical slice. For research, it may be a reproduced baseline, a falsified assumption, or a completed ablation. For architecture, it may be a prototype that validates an interface or retires a major risk. Not all progress means shipping; in ambiguous work, progress often means reducing uncertainty.

Small, repeated wins also build confidence. Give people problems they can own, make the goal clear, let the result be visible, and then increase the scope. Growth often looks like a sequence of small wins long before it looks like a promotion.

## Trust is consistency over time

Trust is not built through a single vulnerable conversation, a team offsite, or a well-written values document. Trust is consistency over time: between what a manager says and does, across people and standards, and between calm periods and schedule pressure.

A manager’s first reaction to bad news matters. When someone brings me a failure, a slipped deadline, or a mistake, my response teaches the team what to do with the next one. If my first reaction is blame, I am choosing to receive bad news later. If it is curiosity and problem-solving, I make early escalation more likely. We should diagnose before we judge.

Trust creates speed. People spend less time protecting themselves, interpreting hidden motives, or documenting every interaction defensively. They ask questions sooner, challenge ideas more directly, and commit more fully.

High trust does not mean low accountability. Expectations should be clear: what good work looks like, where people have autonomy, when they need to escalate, and how performance will be evaluated. Hidden standards make every outcome feel political. Culture is not what we say we value. It is the pattern of consequences people observe.

## One-on-ones

One-on-ones are an incredible coaching tool if used correctly. But I have often seen them boil down to status meetings. Status can usually be communicated asynchronously or discussed in a project forum. A one-on-one should focus on the person, the relationship, and the conditions under which that person is doing their work.

The org chart tells me who reports to whom. One-on-ones tell me where the organization is actually stuck. They surface unclear expectations, missing context, interpersonal friction, repeated interruptions, loss of motivation, unspoken disagreement, and uncertainty about growth. These are leading indicators; by the time they appear in a missed milestone or performance review, the underlying problem may have existed for months.

I want one-on-ones to contain questions that are difficult to ask in a project meeting: What is taking more energy than it should? Where do you need more context or authority? Which part of your work is helping you grow? What are you avoiding? What am I doing that makes your job harder? What do you believe the team is getting wrong?

Good one-on-ones are not passive listening sessions. A manager should bring observations, offer context, give feedback, and provide a point of view. Coaching that consists only of questions can feel like abandonment. Sometimes the right response is, “What do you think?” Sometimes it is, “Here is how I would reason about this.” Sometimes the person needs a direct decision.

One-on-ones also require follow-through. If someone raises the same concern three times and nothing happens, the meeting teaches them that candor is performative. Trust grows when a person can see that being honest changed something.

## Delegate opportunity, not just work

Delegation is not unloading tasks. It is designing ownership.

Bad delegation transfers implementation while retaining the context and decision-making authority. Good delegation transfers a coherent problem. The person understands why the work matters, the constraints and success criteria, the decisions they can make independently, and the situations in which they should escalate.

I think about each person’s work as a mixture of leverage, stretch, and stewardship. Leverage work matters to the team or organization. Stretch work expands judgment, technical ability, or influence. Stewardship keeps the system healthy through operations, maintenance, documentation, migrations, and process improvements.

Everyone will do some stewardship work, but nobody should receive only stewardship while someone else gets all the ambiguous, visible, and exciting problems. Over time, access to ownership, learning, and meaningful work should be distributed intentionally.

Autonomy does not mean isolation. People should be able to make progress without constant approval, but they should also escalate early when blocked. A useful request for help includes the goal, what has been tried, what was observed, the current hypothesis, and the specific help needed. I want asking for help to be emotionally inexpensive but intellectually serious.

The manager remains accountable but should become less central. The test of delegation is not whether work disappeared from my plate. It is whether judgment appeared somewhere else on the team.

## Empathy by default, firmness when required

My default is empathy. I want to understand the situation before prescribing a solution because the same visible problem can have different causes: missing context, excessive parallel work, a skill gap, burnout, or avoidance. Good management begins with diagnosis.

But empathy does not remove responsibility, and kindness does not make the standard ambiguous. Empathy without standards becomes avoidance. Standards without empathy become fear.

Firmness means being clear when something needs to change. Feedback should be timely, specific, and connected to impact. The person should understand what happened, why it matters, what is expected, and what support is available. Nobody should encounter important negative feedback for the first time in a performance review.

It is kinder to have a difficult conversation early than to let uncertainty accumulate for months. Someone can be capable and still be struggling. Someone can have good intentions and still create a harmful pattern. The purpose of firmness is not punishment. It is clarity.

## Innovation needs capacity

Managers often say they want innovation while planning every engineer at full utilization, rewarding only predictable delivery, and treating failed experiments as wasted time. Teams learn what is valued from what managers fund, review, celebrate, and protect.

Innovation requires capacity. People need time to read, think, compare approaches, build prototypes, and follow observations that do not fit the current plan. Slack is not automatically waste. It is often the capacity from which improvement emerges.

But innovation is not novelty for its own sake. A good experiment starts with a real question: What assumption are we testing? How much are we willing to spend? What result would cause us to continue, stop, or change direction? The best experiments collapse uncertainty cheaply. If a two-week prototype prevents a six-month mistake, it created enormous value.

Continuous improvement should be a reflex. What slowed us down this week? What recurring work should be automated? Where are handoffs creating ambiguity? Which assumption is no longer true? Every recurring frustration is a tax. Sometimes that tax is worth paying, but it should be visible and intentional.

A high-performing team should not merely deliver more over time. It should become better at delivering. Its tools improve, its decisions become clearer, its operational burden falls, and its engineers take on broader ownership. Every few months, the team should be capable of work that would previously have overwhelmed it.

## What I am trying to build

My management style can be summarized as high context, high ownership, high standards, and low ego.

I want people to understand what matters and why, and to have enough context to make decisions without waiting for me. I want them to challenge my assumptions. I want us to do the difficult thinking before an expensive commitment, then put our heads down and execute.

Lots of voices, one direction.

I want progress to be visible because it improves learning and builds confidence. I want people to receive exciting work as well as necessary overhead. I want one-on-ones to cover judgment, growth, motivation, and friction, not project status. I want engineers to ask for help before a small blockage becomes a large surprise.

I want trust to come from consistency. I want to understand circumstances with empathy and protect standards with firmness. I want to lead by example without making myself indispensable.

Ultimately, the manager’s output is not control. It is capability.

The strongest evidence of good management is that the team makes sound decisions when the manager is not in the room. Problems surface earlier. Engineers take on larger scope. Disagreement becomes more honest. Execution becomes more focused. The system improves without waiting for one person to push it forward.

That is the kind of team I want to build: not one that depends on my answers, but one whose collective judgment compounds over time; not one that is always busy, but one that repeatedly turns important problems into tangible progress; not one without disagreement, but one that knows how to disagree, decide, rebase, and execute.

That, to me, is engineering management.
