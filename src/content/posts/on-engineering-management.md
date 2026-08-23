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

An engineering manager’s job is to help a team improve business outcomes under real constraints while creating the conditions for people to grow. The role spans coaching, architecture, hiring, planning, conflict resolution, performance management, communication, and process design. But those activities are mechanisms toward a broader goal, not the final output. The output is a team that can repeatedly identify an important problem, make a sound decision, and turn that decision into a working system. Kamil Sindi’s [Managers Playbook](https://github.com/ksindi/managers-playbook) is a resource I keep returning to, and it has deeply influenced how I think about the job.

A good engineering manager is not judged by the number of meetings they attend, the decisions that pass through them, or how indispensable they appear. The real test is whether the team makes better decisions, moves with greater conviction, and becomes more capable over time. Management, at its best, creates a compounding loop. Good decisions create progress. Progress creates information. Information improves judgment. Better judgment gives the team greater autonomy and the ability to take on harder problems. A manager’s job is to keep that loop moving.

---

I love writing code! Some of the most gratifying moments in my career have come from seeing code I wrote make a robot do something extraordinary. I also believe that building things is one of the best ways to understand them. But writing code is not the end goal. It is in service of the problem the code solves. Time and again, I have seen strong engineers become seduced by the elegance of a solution. They can spend weeks optimizing an architecture, cleaning up an abstraction, or adopting a new technique without stopping to ask: does this matter?

A good manager keeps technical work grounded in its purpose and aligned with the goals of the business or organization. What user problem does the work solve? What capability does it unlock? What constraint does it remove? What becomes possible once it is complete?

That does not mean every technical decision needs an immediate return. Foundational research, reliability work, infrastructure, and technical debt reduction all matter. But each type of work still needs a theory of impact: how will it affect the business a few weeks or a few months from now? A team may spend more time designing an interface up front so that the next integration is easier. It may also slow down an implementation because a rushed system accrues technical debt and becomes harder to replace once production depends on it. The return can be delayed, but it cannot be undefined.

A theory of impact is stronger when the engineer doing the work helps shape it. That participation matters especially for stretch goals. The manager can set the goal, explain why it matters, and make the constraints clear, but the engineer should help choose the path. Buy-in turns an assigned task into an owned decision. It also creates room for growth while keeping the engineer and manager aligned on what success means.

For much of my career, I was the engineer people turned to for the hardest, most ambiguous problems. I became used to owning the solution end to end, and that instinct did not disappear when I became a manager. I still wanted to review the difficult design, build the prototype, inspect the traces, fix the annoying bug, and write the first draft myself. I learned the hard way that staying technical does not mean putting myself on every critical path.

Leading by example is different from leading through dependency. If I take over every hard problem, the team learns that important work will eventually route through me. Staying technical should keep my judgment honest, not make the team depend on me. My job is no longer to provide the answer to every hard problem. It is to build a team that can solve problems I could not solve alone.

---

Go slow to go fast. Before committing to an expensive direction, a team should explore broadly. That means reading the literature, understanding previous attempts, establishing a baseline, writing down assumptions, and examining failure modes. The team should then build the smallest prototype that can either increase its confidence in the direction or reveal a reason to change it.

The depth of investigation should match the cost and reversibility of the decision. A small, reversible product change does not require weeks of consensus-building. The team can make a reasonable decision, instrument it, and learn. An architecture that will define the system for years deserves more care. Expensive decisions deserve cheap evidence: a prototype, benchmark, ablation, trace, or small production experiment that can turn an abstract disagreement into an empirical question. The manager’s role is not to build the whole system, but to help the team create enough evidence to make a better decision.

Before the work begins, the team should know which uncertainty it is trying to reduce, how much time it will spend, and what result would change the decision. Careful thinking should make execution more decisive, not postpone it.

Sometimes, even after months of work, the results do not justify continuing in the same direction. I like the phrase “strong opinions, weakly held” because it captures how I think about these moments. The phrase does not mean weak conviction. Deep conviction often carries difficult work forward, but it must remain accountable to evidence. When results contradict the plan, the team has to update the plan rather than defend the time already spent. Otherwise, sunk cost can keep a six-month project alive even when a one-week experiment could have disproved its central assumption. Battle scars should sharpen future questions, not fossilize conclusions that the evidence no longer supports.

---

Every person on the team, no matter how new or junior, should be able to express an opinion freely. Healthy teams question the framing of a problem, examine the data, expose missing assumptions, and propose alternatives. Silence can look like alignment, but it can also mean people no longer believe disagreement is worth the cost. I have seen teams fail at both ends of this spectrum. Some suppress disagreement in the name of speed, producing superficial alignment and delayed resistance. Others let discussion continue indefinitely in the name of inclusion until nobody knows who owns the decision. Inclusion does not give everyone veto power. It gives each relevant perspective a genuine chance to change the team’s model of the problem.

Before a decision, the goal is to surface the strongest arguments and choose the best direction. Once the team decides, the goal shifts to coordinated execution, and everyone rebases onto the chosen direction. People are more willing to support a decision they opposed when they understand the reasoning, know their concerns were considered, and can see what evidence would justify revisiting it. The decision should make the desired outcome, accepted tradeoffs, owner, success criteria, and conditions for reconsideration explicit. Once that context is clear, the team can commit to the direction and test it against reality. A team that learns from execution will usually beat one that keeps defending its forecast. Time spent executing often beats even a very strong prior.

---

Build useful metrics! Progress is a management primitive, and metrics make it visible. They tell a team whether its work is moving toward the outcome that matters and give leadership a clear view of that progress. But metrics are only as complete as the failures the team knows to look for. Ship early and often, instrument every release, and let production expose what the evaluation missed. I learned this the hard way after we shipped a model that looked excellent on every metric we tracked. In production, we found an exposure-related failure mode that none of those metrics captured. We had to define a new metric, update the evaluation, and use that signal to guide the next model update. Production revealed both a model failure and a gap in how we measured quality.

Visible progress also creates energy and builds momentum. Teams behave differently when they can see themselves moving: collaboration gets easier, harder problems feel tractable, and ambiguity becomes manageable. That is why large efforts need milestones that produce useful evidence. For implementation, that may be a working vertical slice; for research, a reproduced baseline or an overfitting experiment; for architecture, a prototype that validates an interface or retires a major risk. Not all progress means shipping. In ambiguous work, reducing uncertainty is progress. Small, repeated wins build confidence too. A manager can give an engineer a problem to own, clarify the goal, make the result visible, and then increase the scope. Growth often looks like a sequence of small wins long before it looks like a promotion.

---

Trust is built through consistent behavior over time. Keeping commitments, sharing difficult context honestly, applying the same standards across the team, and staying steady when a deadline slips all show people that they can rely on their manager. I also believe in overcommunicating, but that does not mean pulling me into every Slack thread. It means understanding who needs which context and sharing it at the right time. When that boundary is unclear, I would rather share too much than too little. Each of these actions is small on its own. Repeated over months, they build more trust than any single vulnerable conversation, team offsite, or values document.

Bad news is one of the clearest tests of that consistency. When someone reports a failure, a slipped deadline, or a mistake, the manager’s response shapes whether that person raises the next problem early. If the manager reacts with blame, people learn to protect themselves. Problems stay hidden until the situation becomes harder to fix. If the manager first tries to understand what happened, people are more likely to escalate while the problem is still manageable. Accountability still matters, but diagnosis should come before judgment.

The same consistency should apply to expectations. People should know what good work looks like, where they have autonomy, when they need to escalate, and how performance will be evaluated. When both support and standards are predictable, people spend less time protecting themselves or interpreting hidden motives. They ask for help sooner, challenge ideas more directly, and move faster.

---

I’ve often seen 1:1s become status meetings. Most updates can be written asynchronously or handled in standups. The value of a 1:1 lies in the private context around the work. I use the meeting for questions that are difficult to ask in a project forum: What is taking more energy than it should? Where do you need more context or authority? Which part of your work is helping you grow? What are you avoiding? What do you believe the team is getting wrong? I do not treat these as a checklist. A question matters only if the answer changes what one of us understands or does next. The goal is to surface issues while they are still small enough to address.

Feedback is a gift in both directions, and I try to be as intentional about receiving it as I am about giving it. When I ask, “What am I doing that makes your job harder?” I need to listen without defending my intent because my response determines whether the person answers honestly the next time. Questions are only one part of coaching. I also bring observations, context, feedback, and a point of view. Sometimes I ask, “What do you think?” Sometimes I explain how I would reason about the problem. Sometimes the person needs a direct decision. If I only ask questions and never offer judgment, I leave them alone with a problem I should be helping solve.

A 1:1 also needs follow-through. If someone raises the same concern three times and nothing changes, the meeting teaches that person that honesty is a waste of time. A 1:1 should end with a clear next step, an owner, or an explicit decision not to act. The manager should then follow up.

---

Good delegation means giving engineers meaningful problems. For example, I like to delegate work that I would be excited to do myself. The goal is not to unload tasks. It is to create real ownership around a problem the engineer can get excited about solving. Bad delegation means the manager chooses the solution, breaks it into tickets, and assigns the implementation while every meaningful decision still comes back to the manager for approval. Real ownership starts with the problem: why it matters, what success looks like, which constraints are fixed, and where the engineer can decide independently. The engineer owns the approach, the tradeoffs, and the path to the result.

Every team has necessary but less visible work: operations, maintenance, migrations, and documentation. That work should be shared, but so should the ambiguous, visible, and exciting problems. Those problems give people the chance to build judgment and take on broader scope.

Ownership does not mean leaving someone alone. Engineers need to be able to make progress without constant approval and know when to pull the manager in. They also need to escalate early when blocked, with enough context to make the conversation useful: the goal, what they tried, what they observed, their current hypothesis, and the decision or help they need. Asking for help should be easy, but thinking through the request is still part of owning the problem. The manager remains accountable for the outcome while becoming less central to each decision. That means providing context, challenging the reasoning, and removing blockers without taking the problem back.

---

Empathy is essential to good management. A missed deadline can come from missing context, too many competing priorities, a skill gap, burnout, or avoidance. A manager should understand the cause before prescribing a solution. But a manager’s job is not to be everyone’s friend or to optimize for being liked. Engineering teams solve hard problems with real commitments, and their decisions consume significant time and money. Managers owe people understanding and honest feedback, and they owe the team a clear standard.

When something needs to change, a manager has to say so clearly. Feedback needs to be timely, specific, and connected to impact: what happened, why it matters, what needs to change, and what support is available. Nobody should encounter important negative feedback for the first time in a performance review. Someone can be capable and still be struggling, and good intentions can still create a harmful pattern. Avoiding the conversation may feel kind in the moment, but it denies the person a fair chance to improve. The goal is not punishment. It is to make the problem clear while there is still time to change it.

---

Managers often say they want innovation while planning every engineer at full utilization, rewarding only predictable delivery, and treating failed experiments as wasted work. Teams learn what matters from what managers fund and protect. Capacity has to exist in the plan. If reading, prototyping, or following through on a retrospective depends on spare time after roadmap work, it will rarely happen. A manager should plan slightly below full utilization and give the remaining capacity a deliberate purpose: exploring new ideas, improving reliability or tooling, or removing recurring friction. That capacity also absorbs the uncertainty that hard technical work always carries.

Retrospectives are one way to decide where that capacity should go. In a retrospective, the team asks what slowed the team down, which recurring work should be automated, where handoffs created ambiguity, and which assumptions are no longer true. The retrospective needs to end with a concrete change and an owner, or an explicit decision to accept the tradeoff. Otherwise, the same friction returns at the next one. Over time, a high-performing team should become better at delivering, not simply deliver more. Its tools improve, its decisions become clearer, its operational burden falls, and its engineers take on broader ownership. Every few months, the team should be capable of work that would previously have overwhelmed it.

---

I keep coming back to four ideas: give people the context they need, let them own meaningful problems, be clear about the standard, and keep ego out of the way. In practice, that means explaining why the work matters, inviting disagreement before a decision, and giving engineers room to choose the path. Once the team commits to a direction, it ships, measures what happens, and lets the evidence change the plan.

Working alongside the team on technical problems lets me understand the tradeoffs, offer useful feedback, and remove blockers without taking the problem back. I want engineers to take on broader scope and the team to keep moving without waiting for me. That works only when problems surface early, expectations stay clear, and the plan leaves room to explore new ideas, improve reliability, and remove recurring friction.

I am still learning how to do this job well. These principles help me see what is working and where I need to improve. I will not always have the best answer, and I will get things wrong. What matters is recognizing it, updating my thinking, and continuing to learn from the people around me.
