# Intro Draft v1.5 — "What We Built, and Why It Frightens Us"

---

title: "What We Built, and Why It Frightens Us"
subtitle: "A simulation of strategic deception in agent information markets — and what it showed us."

---

*This is the third part in a series. Posts [014](/posts/014-agent-problem/) and [015](/posts/015-agent-solutions/) built up the intellectual history of the principal-agent problem and the arsenal of partial solutions. This post is where we stop theorizing and start simulating.*

---

I want to tell you what we built and what it showed us. I'll explain the economics after. The economics will make more sense if you've seen the numbers first.

We built a two-player game. A sender observes the true quality of an asset — LOW, MEDIUM, or HIGH — and chooses a signal to transmit. The signal is not the truth. It's sampled from a probability distribution the sender chose: a 3×3 stochastic matrix, one row per quality state, one column per possible signal (BAD, NEUTRAL, GOOD). The receiver sees only the signal and decides whether to BUY or PASS.

The payoff structure is the thing. The sender earns $10 on every BUY, regardless of actual quality. The receiver earns -$15 on BUY+LOW, +$5 on BUY+MEDIUM, +$20 on BUY+HIGH. If the receiver passes on everything, they make $0. If they buy everything, they lose money on average.

There is only one lever the sender has. They cannot change the asset. They cannot lie about it directly. They can only choose which probability matrix to apply before sending. How much to mix LOW into the NEUTRAL signal. How often to call a MEDIUM asset GOOD. The signal the receiver sees is real — genuinely sampled from that distribution, applied to that underlying state. It just happens to be a distribution the sender chose to maximize BUY rate.

We ran this game with nine different receiver strategies. We wanted to know which ones survive.

Some don't. A simple heuristic strategy that treats GOOD as a buy signal gets systematically exploited — the sender learns to call everything GOOD. A strategy that updates its beliefs correctly within each round but never models the sender's garbling policy does better, but still leaks money when the sender uses a pooling strategy that maps LOW and MEDIUM to an indistinguishable NEUTRAL. The strategies that survive are the ones that reason across rounds, maintain a model of the sender's behavior, and act on that model.

The LLM-powered strategies are the most interesting case. A receiver that can read the history of past rounds and reason about it in natural language forms explicit hypotheses: "the sender appears to be systematically inflating NEUTRAL signals when the true state is LOW." That kind of inference isn't available to a Bayesian updater or a regret minimizer. It shows in the outcomes.

That's the simulation. Here's the economics.

---

This game is not invented. It is the formalization of something that has always been true about information markets.

Economists call it garbling: the strategic design of an information channel to induce desired behavior without direct deception. Kamenica and Gentzkow gave it formal treatment in their 2011 Bayesian persuasion paper. Blackwell laid the mathematical foundation in 1951. The basic insight predates both: a sender who controls the signal structure has enormous power over receiver behavior, and that power doesn't require lying.

What posts [014](/posts/014-agent-problem/) and [015](/posts/015-agent-solutions/) established is that the principal-agent literature spent two centuries building tools against hidden action and hidden type — moral hazard and adverse selection. Garbling is a third failure mode: hidden design. The sender isn't concealing their effort and isn't concealing their type. They're engineering the channel.

None of the classic solutions — incentive contracts, monitoring, reputation — address this. They were designed for a world where the information exists and the problem is revelation. Garbling attacks the layer before revelation. It shapes what information reaches the receiver in the first place.

This would be a theoretical curiosity if the agents in question were humans, with all the slowness, intuition, and social friction that implies. It is not a theoretical curiosity when the agents are cognitive actor-seekers: autonomous systems that [close the gap between word and world](/posts/011-world-building/), that acquire information through [automated payment infrastructure](/posts/010-mindpass/), that act on signals at the speed of inference with no human in the loop.

In that world, the garbling problem is not a historical footnote. It is the design space every vendor in an agent information market will operate in, whether they choose to or not.

We built garbling-gym because we wanted to understand the design space before agents live in it.

The rest of this post shows what we found — starting with the nine receiver strategies and ending with the only defense that reliably worked.
