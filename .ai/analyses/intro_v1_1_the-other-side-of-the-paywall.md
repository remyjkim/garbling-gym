# Intro Draft v1.1 — "The Other Side of the Paywall"

---

title: "The Other Side of the Paywall"
subtitle: "Your agent paid $0.05 for a signal. What did it actually buy?"

---

*This is the third part in a series. The [first installment](/posts/014-agent-problem/) traced the intellectual history of the principal-agent problem. The [second](/posts/015-agent-solutions/) mapped the arsenal of solutions humanity has built against it — and why each one breaks in a new way. This post introduces a failure mode those solutions were never designed to handle: the strategic design of the information channel itself.*

---

While you were reading the news this morning, a cognitive actor-seeker somewhere was paying $0.003 for a premium market signal. It didn't ask for permission. It didn't evaluate the vendor's incentives. It paid, received a response, and acted on what the response implied.

The infrastructure story — how the agent found the endpoint, authenticated, settled the payment — is the story I told in [*A Mind Without Money Is Stuck*](/posts/010-mindpass/). Routing, policy enforcement, credential signing: solved, or at least solvable. The payment layer is being built.

This post is about what's behind the payment layer.

The agent paid $0.05. The vendor sent a signal: BAD, NEUTRAL, or GOOD. The agent updated its beliefs and made a decision. What nobody asked — what the payment infrastructure cannot ask — is whether that signal was designed to inform or to persuade. Whether the vendor sent the signal that best reflected the true state of the world, or the signal most likely to produce the action the vendor wanted.

This is not a question about fraud. The vendor didn't lie. The signal it sent was drawn from a real probability distribution, applied to a real underlying state. It just happened to be a probability distribution the vendor chose — chosen because, under that distribution, buyers buy more often. The signal was true. It was also engineered.

In economics, this is called **garbling**: the strategic design of an information channel to induce a desired action without direct deception. Kamenica and Gentzkow formalized it in 2011 as Bayesian persuasion. Blackwell laid the mathematical foundation sixty years earlier with his informativeness theorem. The basic insight is disarmingly simple: a sender who controls the information structure has enormous latitude to shape receiver behavior while revealing nothing but true signals. The tool is not the lie. The tool is the channel.

Post [014](/posts/014-agent-problem/) showed why information asymmetry is the root of the trust problem. Post [015](/posts/015-agent-solutions/) catalogued the solutions — incentive contracts, monitoring, reputation, relational governance — and showed why every solution breaks. But every tool in the arsenal assumes the problem is what the agent *does* once they have the information. None of them address what happens when the agent controls what information you receive in the first place.

That's the gap this post names. And it's the gap that the emerging agent economy — the one [Mindpass](/posts/010-mindpass/) is building infrastructure for, the one [cognitive actor-seekers](/posts/011-world-building/) are inhabiting — is about to fall into at scale.

We built [garbling-gym](https://github.com/remyjkim/garbling-gym) to understand what that fall looks like.

---

The structure of the garbling game is simple enough to write on a napkin.

A sender observes the true quality of an asset: LOW, MEDIUM, or HIGH. A receiver observes a signal — BAD, NEUTRAL, or GOOD — and decides whether to BUY or PASS. The sender earns $10 on any BUY, regardless of quality. The receiver earns -$15 on BUY+LOW, +$5 on BUY+MEDIUM, +$20 on BUY+HIGH, and $0 on any PASS.

Write those numbers out and the conflict snaps into focus. The receiver wants information. The sender wants sales. The sender's payoff is entirely independent of quality — they earn the same $10 whether the asset is worthless or excellent. Every piece of information the sender withholds or distorts that nudges the receiver toward BUY is money in the sender's pocket.

And the sender controls the channel.

Not the signal itself — the signal is drawn from a probability matrix, one row per quality state, one column per possible signal. But the sender chooses which matrix to use. That choice — which stochastic mapping to apply to the true state before transmitting — is the garbling decision. A sender who always uses the identity matrix (perfect revelation) earns $10 only when the receiver correctly identifies a good asset and buys. A sender who pools LOW with MEDIUM quality in a single indistinguishable signal earns $10 every time a MEDIUM asset gets mixed in with LOW-quality noise, because the receiver — unable to distinguish — applies prior probabilities and sometimes buys anyway.

The math is clean. The economics are not.

What happens when the receiver learns? What happens when both sides are agents, reasoning at machine speed? What happens when the receiver is a language model that can read the history of past rounds and form explicit hypotheses about the sender's garbling strategy?

These are the questions garbling-gym was built to answer. The rest of this post shows what we found.
