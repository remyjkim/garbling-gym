# Intro Draft v1.4 — "The Signal Economy"

---

title: "The Signal Economy"
subtitle: "We spent two centuries worrying about what agents do with information. We forgot to ask who designs it."

---

*This is the third part in a series. The [first installment](/posts/014-agent-problem/) traced the intellectual history of the principal-agent problem. The [second](/posts/015-agent-solutions/) surveyed the solutions — contracts, monitoring, reputation, relational governance — and showed why each one cracks under pressure. This post introduces a failure mode none of them were built to handle.*

---

There is an assumption embedded in two and a half centuries of principal-agent theory that almost nobody has bothered to name, because until recently it was approximately true.

The assumption is this: information is a given. It exists in the world, is held by some parties and not others, and the problem is one of revelation — getting the informed party to share what they know. The tools the literature built — incentive contracts, monitoring mechanisms, reputation systems — are all revelation mechanisms. They are answers to the question: *given that the agent knows something, how do we get them to tell the truth?*

What they are not answers to is the prior question: *what does the agent choose to put in the information channel in the first place?*

This is the question the garbling literature has been quietly circling since Blackwell's 1951 informativeness theorem and Kamenica and Gentzkow's 2011 Bayesian persuasion paper. A sender who controls the information structure — the stochastic mapping from true states to observable signals — doesn't need to lie, doesn't need to withhold, doesn't need to shirk. They just need to design. Choose the right matrix. Send signals that are technically drawn from a real probability distribution over real states. Let the receiver update on those signals and act.

The sender earns every time the receiver acts. The matrix is the tool. The matrix is the business model.

This was always latent in information markets. The pharmaceutical company that designs a clinical trial to make a drug look better than it is. The financial advisor who frames a recommendation to maximize AUM. The real estate agent who leads with the kitchen because kitchens sell. These are all garbling strategies — some of them illegal, most of them legal, all of them old. The principal-agent literature saw the problem at the level of the individual transaction and tried to fix it there.

What's changed is the infrastructure.

[*A Mind Without Money Is Stuck*](/posts/010-mindpass/) described the payment rails being built for autonomous agents — the infrastructure that lets a cognitive actor-seeker query a vendor, authenticate, and settle a micropayment without human involvement. [*Let There Be Light*](/posts/011-world-building/) described what cognitive actor-seekers do with the information they receive: act on it, causally, at the speed of inference, in ways that propagate into the physical and digital world.

Put those two things together and you get a signal economy: a market where cognitive actor-seekers are the primary consumers of information, vendors are autonomous agents with their own objective functions, and the garbling decision — which matrix to use — is made at machine speed, refined by feedback from millions of transactions.

In a signal economy, the revelation problem is the wrong problem. The agents are willing to reveal. They're just revealing from a matrix they chose.

We built [garbling-gym](https://github.com/remyjkim/garbling-gym) to study the signal economy at its most basic level: two agents, one information channel, a misaligned payoff structure, and the question of what a rational buyer learns to do.

---

The game strips the problem to its skeleton.

A sender observes an asset quality — LOW, MEDIUM, or HIGH — and selects a garbling matrix, a 3×3 stochastic transformation that maps each quality state to a probability distribution over signals: BAD, NEUTRAL, or GOOD. The receiver sees only the signal and decides: BUY or PASS.

The payoffs are calibrated to create pressure. The sender earns $10 on any BUY, quality-independent. The receiver earns -$15 on BUY+LOW, +$5 on BUY+MEDIUM, +$20 on BUY+HIGH. The sender wants transactions. The receiver wants good transactions. Both are acting rationally. The conflict is structural.

Garbling-gym runs this game with nine different receiver strategies: Bayesian, Dirichlet, regret-matching, Hedge, EXP3-style bandits, UCB-based, level-k reasoning, and two LLM-powered strategies. The receiver strategies are a taxonomy of cognitive defenses — what does it take, epistemically, to survive an adversarial signal economy?

Some strategies are simple and predictable; they get exploited. Some are adaptive; they converge, but slowly. Some use language model reasoning to form explicit models of the sender. The variance across strategies is the finding.

The rest of this post shows what we found.
