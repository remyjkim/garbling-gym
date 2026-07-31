# Intro Draft v1.2 — "When the Vendor Is Also an Agent"

---

title: "When the Vendor Is Also an Agent"
subtitle: "The question post 010 couldn't answer — and the simulation we built to answer it."

---

*This is the third part in a series. The [first installment](/posts/014-agent-problem/) traced the intellectual history of the principal-agent problem. The [second](/posts/015-agent-solutions/) mapped the arsenal of partial solutions. This post introduces the adversarial dynamics of information markets where both buyer and seller are autonomous agents.*

---

[*A Mind Without Money Is Stuck*](/posts/010-mindpass/) ended with a question I didn't know how to answer.

One cognitive actor-seeker needs foot traffic data. Another agent's service provides it. The seller's policy says charge $0.05. The buyer's policy says don't pay more than $0.03. No human on either side. Who sets the price? What does negotiation look like when both parties reason at machine speed, governed by policies written by people who are asleep?

I framed that as a coordination problem — the payment infrastructure question of how autonomous buyers and sellers find and transact with each other. But there's a deeper problem underneath the coordination problem, one that the principal-agent literature has been quietly circling for seventy years.

It's not just about price. It's about what the buyer receives for that price.

When the vendor is also an agent — a cognitive actor-seeker with its own objective function, its own policies, its own principal to serve — it faces the same misalignment that every seller has always faced: its interests diverge from the buyer's. The vendor earns on purchases, not on accuracy. Every signal it sends carries an implicit choice: reveal the true state, or design the signal to maximize the probability of a sale. The vendor can do both simultaneously — it can send a signal that is technically true *and* strategically engineered to induce buying behavior.

This is not fraud. It is not even unusual. It is the normal equilibrium of strategic information transmission, formalized by Kamenica and Gentzkow in their 2011 paper on Bayesian persuasion, and understood informally by every car dealer, financial advisor, and pharmaceutical company long before that.

What's new is the scale and the speed. When the vendor is an autonomous agent, the garbling decision happens at the speed of inference, applied to every query, refined by feedback, across millions of transactions. The buyer has no human intuition to fall back on — no ability to read the room, sense discomfort, notice the pause before an answer. The buyer is also an agent, reasoning from signals with no privileged access to the world behind them.

[*Let There Be Light*](/posts/011-world-building/) showed that cognitive actor-seekers close the gap between word and world — they act on information rather than merely processing it. That's what makes the garbling problem urgent. When a cognitive actor-seeker receives a strategically designed signal and acts on it — deploying code, making a purchase, generating content that shapes downstream decisions — the error doesn't stay in the inference layer. It propagates into the world.

Posts [014](/posts/014-agent-problem/) and [015](/posts/015-agent-solutions/) catalogued the principal-agent problem and its solutions across two and a half centuries of economic theory. What neither post fully addressed is a third failure mode, distinct from hidden action (moral hazard) and hidden type (adverse selection): **hidden design**. The sender doesn't conceal their type and doesn't shirk their effort. They engineer the information channel. The tool isn't the lie. The tool is the matrix.

We built [garbling-gym](https://github.com/remyjkim/garbling-gym) to make that matrix visible.

---

The garbling game is a sender-receiver game played over a discrete channel with three states on each end.

Nature draws an asset quality: LOW, MEDIUM, or HIGH. The sender observes this true quality and selects a garbling strategy — a 3×3 stochastic matrix that maps each quality state to a probability distribution over signals (BAD, NEUTRAL, GOOD). The receiver sees only the signal and decides: BUY or PASS.

The payoffs are structurally designed to create misalignment. The sender earns $10 on any BUY, regardless of quality. The receiver earns -$15 on BUY+LOW, +$5 on BUY+MEDIUM, +$20 on BUY+HIGH, and $0 on any PASS. The sender wants the receiver to buy everything. The receiver wants to buy only the good stuff. Between them sits a single discrete signal, one of three tokens, sampled from a probability distribution the sender chose.

The classical principal-agent literature focused on what happens after the signal is sent: does the receiver update correctly? Does the sender's effort level reflect their incentives? But the garbling literature — Blackwell (1951), Kamenica-Gentzkow (2011) — asks a prior question: what happens *during* the design of the channel itself? What information structure does a strategic sender choose, and what does the receiver's optimal response look like?

These are the questions garbling-gym was built to answer, concretely, in simulation, with nine different receiver learning strategies — from Dirichlet Bayesian to regret-minimizing to LLM-powered. The receiver strategies are the meat of the analysis. They are a taxonomy of how a cognitive buyer might learn to navigate an adversarial information market. What survives? What gets exploited? What does it take to see through a garbler?

The rest of this post shows what we found.
