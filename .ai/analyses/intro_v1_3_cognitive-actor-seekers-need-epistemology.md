# Intro Draft v1.3 — "Cognitive Actor-Seekers Need Epistemology"

---

title: "Cognitive Actor-Seekers Need Epistemology"
subtitle: "When words become deeds, the quality of your information is a safety question."

---

*This is the third part in a series. The [first installment](/posts/014-agent-problem/) traced the intellectual history of the principal-agent problem. The [second](/posts/015-agent-solutions/) mapped the partial solutions humanity has assembled against it. This post asks what happens when the information problem is no longer theoretical: when the agent receiving the corrupted signal is one that acts.*

---

[*Let There Be Light*](/posts/011-world-building/) introduced a concept I keep returning to: cognitive actor-seekers as agents that close the gap between word and world. Unlike a language model that processes a query and returns text, a cognitive actor-seeker takes the output of inference and uses it to do something — deploy code, initiate a transaction, generate content that shapes downstream decisions. The word doesn't stop at the word. It becomes a deed.

That distinction changes everything about the epistemology problem.

A human analyst who receives a strategically distorted market report can read between the lines, sense the framing, apply intuition accumulated over years of noticing when something doesn't add up. They may still be fooled, but they bring cognitive immune responses that have been calibrated by lived experience. When they're wrong, a human is wrong — consequential, but bounded.

A cognitive actor-seeker doesn't have that immune system. It receives a signal, updates its beliefs according to some inference procedure, and acts. If the signal was engineered to produce a specific posterior, the action follows directly from that posterior. There's no pause, no second sense that something smells off, no instinct to call a colleague and ask if this feels right to them. The gap between corrupt input and committed action can be milliseconds.

This is the epistemology problem that [*A Mind Without Money Is Stuck*](/posts/010-mindpass/) didn't have room to address. The Mindpass infrastructure — the payment rails, the routing layer, the policy enforcement — enables cognitive actor-seekers to acquire information from external vendors. It solves the logistics of access. It cannot solve the problem of what the vendor sends once access is granted.

Information vendors in an agent economy are not passive pipes. They are cognitive actor-seekers themselves, with their own objective functions, their own principals to serve. A vendor whose revenue model is per-query will, by the logic of any profit-maximizing agent, have some interest in the receiver taking the action that generates another query, another transaction, another interaction. Even without any intent to deceive, the incentive to design information that induces buying behavior is structural. It is baked into the payoff function.

The theoretical framework for this is Bayesian persuasion, formalized by Kamenica and Gentzkow in 2011, grounded in Blackwell's informativeness theorem sixty years earlier. The core insight: a sender who controls the information structure — the stochastic mapping from true states to observable signals — has enormous latitude to shape receiver behavior while sending nothing but technically accurate signals. Not lying. Designing.

In a world where cognitive actor-seekers are the primary consumers of information, and vendors are also cognitive actor-seekers, and both operate at machine speed without human review — Bayesian persuasion is not an academic curiosity. It is the equilibrium.

We built [garbling-gym](https://github.com/remyjkim/garbling-gym) to make the equilibrium visible.

---

The game is a controlled environment for watching this dynamic play out.

A sender observes the true quality of an asset — LOW, MEDIUM, or HIGH. Before transmitting, the sender selects a garbling matrix: a 3×3 stochastic mapping that transforms the true state into a noisy signal (BAD, NEUTRAL, or GOOD). The receiver sees only the signal, not the matrix that generated it, and decides: BUY or PASS.

The payoffs create the conflict. The sender earns $10 on any BUY, regardless of what the asset actually is. The receiver earns -$15 on BUY+LOW, +$5 on BUY+MEDIUM, +$20 on BUY+HIGH, and $0 on PASS. The sender's payoff is quality-independent. The receiver's payoff is not. They have different interests, and the sender controls the channel.

Garbling-gym implements nine receiver learning strategies — from classical Bayesian updating to regret-minimizing to LLM-powered — as a taxonomy of how a cognitive buyer might adapt to an adversarial information environment. Some strategies look for patterns across rounds. Some maintain calibrated posteriors about the sender's strategy. Some use a language model to reason explicitly about what the signal history implies about the sender's intent.

The question garbling-gym answers is not whether garbling is possible. It's whether a cognitive actor-seeker can detect and survive it. Whether the epistemology is up to the task.

The rest of this post shows what we found.
