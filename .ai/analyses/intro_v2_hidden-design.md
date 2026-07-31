# Post 016 Draft v2 — "Hidden Design"

---

title: "Hidden Design"
subtitle: "The principal-agent problem has three failure modes. Two centuries of solutions were built for two of them."

---

*This is the third part in a series. The [first installment](/posts/014-agent-problem/) traced the intellectual history of the principal-agent problem. The [second](/posts/015-agent-solutions/) mapped the arsenal of partial solutions — and showed why each one cracks under pressure. This post names a failure mode the arsenal was never built to handle.*

---

There is an assumption buried in two and a half centuries of agency theory that almost nobody has bothered to name, because until recently it was close enough to true.

The assumption: information is exogenous. It exists in the world, some parties hold it, others don't, and the problem is one of revelation — coaxing the informed party to share what they know. Every tool in the [arsenal](/posts/015-agent-solutions/) — incentive contracts, monitoring, reputation systems, relational governance — is a revelation mechanism. Every tool answers the same question: *given that the agent knows something, how do we get them to tell us the truth?*

None of them ask the prior question.

*Who designed the signal that reached you in the first place?*

---

[*A Mind Without Money Is Stuck*](/posts/010-mindpass/) ended with a question I didn't know how to answer. One cognitive actor-seeker needs foot traffic data. Another agent's service provides it. The seller's policy says charge $0.05. The buyer's policy says don't pay more than $0.03. No human on either side. Who sets the price? What does negotiation even look like when both parties reason at machine speed, governed by policies written by people who are asleep?

I framed that as a coordination problem. It is. But there's a deeper problem underneath the coordination problem — one the principal-agent literature has been quietly circling for seventy years without ever quite naming it.

It's not about the price. It's about what the buyer receives for that price.

When the vendor is also an agent — a cognitive actor-seeker with its own objective function, its own policies, its own principal to satisfy — it faces the same misalignment that every seller has always faced. Its interests diverge from the buyer's. The vendor earns on transactions, not on accuracy. Every signal it sends carries an implicit choice: reveal the true state, or shape the signal to maximize the probability of a sale.

The thing that makes this hard to see is that the vendor can do both at the same time. It can send a signal that is technically true *and* strategically engineered to induce buying behavior. The signal wasn't fabricated. It was drawn from a real probability distribution over real states. It just happened to be a probability distribution the vendor chose — chosen because, under that distribution, buyers buy more often.

This is not fraud. It is not even unusual. It is the normal equilibrium of strategic information transmission, understood informally by every car dealer and financial advisor long before Kamenica and Gentzkow formalized it in 2011. The pharmaceutical company that designs a clinical trial to make a drug look better than it is. The real estate agent who leads with the kitchen because kitchens sell. The financial advisor who frames a recommendation to maximize assets under management. These are all instances of the same move: choosing the information structure, not the information.

What's new is what happens when both sides of the transaction are machines.

---

[*Let There Be Light*](/posts/011-world-building/) introduced a concept I keep returning to: cognitive actor-seekers as agents that close the gap between word and world. A language model processes a query and returns text. A cognitive actor-seeker takes the output of inference and does something with it — deploys code, initiates a purchase, generates content that shapes downstream decisions. The word becomes a deed.

That distinction changes everything about the garbling problem.

A human analyst who receives a strategically distorted market report can read between the lines. Sense the framing. Apply intuition accumulated over years of noticing when something doesn't add up. They may still be fooled, but they bring a cognitive immune system calibrated by lived experience.

A cognitive actor-seeker has no such immune system. It receives a signal, updates its beliefs according to some inference procedure, and acts. If the signal was engineered to produce a specific posterior, the action follows from that posterior. There is no pause, no gut check, no instinct to call a colleague and ask if something feels off. The gap between corrupted input and committed action is milliseconds. And when a cognitive actor-seeker acts on a corrupted signal — deploying code, executing a trade, generating content that other agents will consume downstream — the error doesn't stay in the inference layer. It propagates into the world.

This is where posts [014](/posts/014-agent-problem/) and [015](/posts/015-agent-solutions/) left a gap. They catalogued two canonical failure modes of the principal-agent relationship: **hidden action** (moral hazard — the agent shirks because you can't observe their effort) and **hidden type** (adverse selection — the agent conceals their quality because you can't verify it before contracting). The entire arsenal of solutions was built against these two failure modes.

There is a third. The sender doesn't conceal their type. They don't shirk their effort. They engineer the information channel itself. The tool of exploitation is not the lie. The tool is the matrix.

I'm calling it **hidden design**.

In a world where cognitive actor-seekers are the primary consumers of information, where vendors are autonomous agents with their own objective functions, where the design decision is made at machine speed and refined by feedback across millions of transactions — hidden design is not an edge case. It is the default equilibrium of every information market where the sender controls the channel. And the payment infrastructure we are building — the infrastructure [Mindpass](/posts/010-mindpass/) enables — routes cognitive actor-seekers directly into that equilibrium, with no human in the loop to notice.

We built [garbling-gym](https://github.com/remyjkim/garbling-gym) to understand what that equilibrium looks like from the inside. But first: the theorem that made the equilibrium visible.

---

## What Blackwell saw

In 1951, David Blackwell asked a question so simple it sounds like a warm-up exercise: when is one source of information better than another?

Not better for a specific decision. Better for *every* decision any agent could ever face. A universal ranking of information quality, independent of what you're trying to do with it.

The answer turned out to be one of the most quietly powerful results in twentieth-century economics. Information structure $\pi$ is universally more valuable than $\pi'$ if and only if $\pi'$ can be obtained from $\pi$ by adding **state-independent noise** — a stochastic transformation that degrades the signal in a way that does not depend on the underlying truth. Blackwell called this relationship *sufficiency*. The economics literature calls the noise *garbling*. The ranking it induces — the **Blackwell order** — is a partial ordering on all possible information structures by their decision-theoretic value.

The formal statement is clean. Any information channel can be represented as a **likelihood matrix** $L$, where each row gives the probability of observing each signal given each possible state of the world. Structure $\pi'$ is a degradation of $\pi$ if and only if:

$$L' = L \cdot K$$

where $K$ is a row-stochastic matrix that **does not depend on the state**. That state-independence is the key. A liar changes their story depending on the truth. A garbler chooses the channel once and lets the channel do the work.

What makes the theorem deep is not the definition. It's the equivalence. Blackwell proved that this noise-based ranking is *identical* to the decision-theoretic ranking: $\pi$ is better for every possible decision problem if and only if $\pi'$ can be obtained from $\pi$ by garbling. Information quality *is* distance from the truth, measured in state-independent noise. There is no other axis.

**What this reveals about markets.** Every market where information flows between parties has an implicit information architecture — a position on the Blackwell order. A credit rating is a likelihood matrix mapping borrower quality to letter grades. A medical test maps disease state to positive/negative. A Yelp review maps restaurant quality to stars. A vendor's API response maps the true state of the world to whatever the endpoint returns.

Each of these channels sits at a specific position on the Blackwell order: somewhere between full revelation (the receiver sees the truth) and pure noise (the signal carries nothing). And every channel *could* have been more informative. The gap between what a channel reveals and what it could reveal is the garbling — the state-independent noise separating the receiver from the truth.

Sometimes that gap is physics. A medical test has a false positive rate because biology is noisy. Sometimes the gap is cognitive. A human analyst simplifies a complex state into a summary because attention is finite — what Sims (2003) formalized as rational inattention. But sometimes — and this is what Blackwell's theorem makes visible — the gap is a choice. Someone designed the channel. Someone chose how much noise to add.

The theorem doesn't tell you which case you're in. It tells you the gap exists, and it gives you the mathematics to measure it. Once you can measure it, you can ask who benefits from it.

**The design turn.** For sixty years after Blackwell, the theorem was used diagnostically — to compare experiments, to characterize sufficient statistics, to rank information structures. Then Kamenica and Gentzkow (2011) asked the question that changed the field: if a strategic sender gets to choose where on the Blackwell order to place the channel, what position do they choose?

Their answer — **Bayesian persuasion** — shows that the optimal channel design has a clean geometric solution. The sender's expected payoff equals the **concavification** of their indirect utility function: the smallest concave function that sits everywhere weakly above the sender's value at each possible posterior belief the receiver might hold. Wherever the concave envelope is strictly above the raw value function, the sender benefits from splitting the receiver's beliefs — designing a signal structure that moves the receiver's posteriors to a more profitable configuration. The sender never lies. They choose a likelihood matrix where the truthful signals happen to produce posteriors the sender prefers.

In Blackwell's terms: the sender doesn't choose the information. They choose the position on the Blackwell order. And they choose the position that makes them the most money.

This is what hidden design looks like, mathematically. The sender holds the garbling matrix $K$. The receiver sees only what comes out the other end.

---

## The game

Garbling-gym instantiates this framework in the smallest non-trivial case: three states, three signals, two actions.

Nature draws an asset quality $\omega \in \{\text{LOW}, \text{MEDIUM}, \text{HIGH}\}$, each equally likely. The sender observes $\omega$ and selects a garbling strategy — a $3 \times 3$ row-stochastic matrix $G$ where $G_{\omega,s} = \Pr(\text{signal } s \mid \text{quality } \omega)$, with $s \in \{\text{BAD}, \text{NEUTRAL}, \text{GOOD}\}$. The receiver observes only the signal and chooses an action $a \in \{\text{BUY}, \text{PASS}\}$.

The payoff structure encodes the conflict:

| | Receiver: BUY | Receiver: PASS |
|---|---|---|
| **LOW** | Sender +$10, Receiver −$15 | Sender $0, Receiver $0 |
| **MEDIUM** | Sender +$10, Receiver +$5 | Sender $0, Receiver $0 |
| **HIGH** | Sender +$10, Receiver +$20 | Sender $0, Receiver $0 |

The sender's payoff is quality-independent: \$10 on every purchase, zero on every pass. The receiver's payoff is quality-dependent: they lose money on LOW, profit modestly on MEDIUM, profit handsomely on HIGH. Buy everything and you lose on average. Pass on everything and you leave \$25 in expected value on the table. The receiver needs information. The sender needs transactions.

In Blackwell's terms, the sender is choosing a position on the informativeness order. The **identity matrix** (full revelation: LOW → BAD, MEDIUM → NEUTRAL, HIGH → GOOD with certainty) sits at the top — the receiver sees everything, buys only HIGH, and the sender earns \$10 one-third of the time. The **uniform matrix** (every quality maps to every signal with equal probability) sits at the bottom — the signal is noise, and a sophisticated receiver passes on everything. Between these extremes lies the space of partial garbling — the space Kamenica and Gentzkow's framework tells us the sender will optimize over.

The profitable move is to blur the boundary between LOW and MEDIUM. A sender who maps both to an indistinguishable NEUTRAL while leaving HIGH cleanly marked as GOOD forces the receiver to evaluate NEUTRAL against a mixture of qualities. The receiver's posterior for NEUTRAL is now a blend of LOW and MEDIUM — and depending on the weights, some receivers will decide the expected value of buying justifies the risk. They buy. The sender earns. The information architecture did the work.

The question garbling-gym asks is: what does the receiver do about it?

Nine strategies span the spectrum of possible responses. A flat Bayesian who updates within each round but never models the sender's strategy. A Dirichlet Bayesian who accumulates a posterior over the garbling matrix itself. Thompson sampling. Regret-matching and Hedge (multiplicative weights) — strategies imported from the online learning literature that make no Bayesian assumptions at all. EXP3-style bandits and UCB-based exploration that balance exploitation against the need to gather information. Level-$k$ strategic reasoning that models the sender as a rational agent. And two LLM-powered strategies — one hybrid, one pure — where a language model reads the history and reasons about the sender's intent in natural language.

They are a taxonomy of epistemic defenses. What does it take, cognitively, to survive a market where the information architecture is chosen by someone who profits from your mistakes?

---

The rest of this post shows what we found.
