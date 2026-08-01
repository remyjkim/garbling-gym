# Research Proposal 10 — Two-Sided Learning: Does an Adaptive Sender Converge to an Equilibrium Garbling Rate?

**Status:** Proposal (capability + science). **Depends on:** 08 (solver). **Defines:** the `SenderStrategy` interface (reused by 09, 11).

**One-line thesis.** The gym has a rich zoo of *receiver* learners but a sender that never learns — it samples a fixed matrix or follows a hand-written heuristic. So we have never observed the central dynamic question of agentic persuasion: when a self-interested sender *adapts* against a learning receiver, does the system converge to a repeated-game equilibrium garbling rate, cycle, or collapse? This proposal makes the sender a first-class learner symmetric to the receiver, and studies the resulting **two-sided learning dynamics** against the solver's static benchmarks.

---

## 1. Research question

Persuasion is a two-sided process: the sender chooses how much to garble, the receiver chooses how much to trust, and each adapts to the other. The gym currently freezes one side. The receiver-modes doc even notes that *"the sender's strategy choice is not learned."* That is the single biggest gap in the framework, and it hides the most interesting science.

> **Core question:** Under what pairings of sender-learner × receiver-learner does the realized garbling rate **converge**, and to what — the one-shot quasiconcave (cheap-talk) value, a repeated-game folk-theorem outcome, or a non-convergent cycle of exploit-and-collapse? Where do LLM senders sit relative to the *learned* optimum?

## 2. Positioning against the literature

- **Crawford & Sobel (1982):** the equilibrium *garbling rate* is the object of study; in the state-dependent (bias-`b`) variant we introduce, the question becomes whether two learners discover the CS partition `N(b)` and its coarsening as `b` rises.
- **Kamenica & Gentzkow (2011) / Lipnowski & Ravid (2020):** the solver's `cav`/`qcav` values are the targets a converging sender should approach (qcav without commitment-by-consistency, cav if reputation supplies it — connecting to Proposal 09).
- **Repeated-games theory (folk theorem; Kandori on garbling in repeated games):** does repetition expand the achievable set beyond the one-shot babbling collapse?
- **LLM-agent dynamics (collusion: Lin et al. 2024; cooperation: Yao et al.):** two LLMs adapting to each other may collude (under-garble together) or destabilize — we measure which, against a benchmark.
- **Explicitly out of scope:** deep-RL senders. The receiver-modes doc already establishes that DQN/PPO/LOLA need 10³⁺ episodes and are non-viable at 20–50 rounds. We restrict to learning rules that converge in tens of rounds (regret-matching, multiplicative weights, bandits, best-response-to-empirical, LLM in-context).

## 3. The `SenderStrategy` interface (shared mechanism, defined here)

A sender-side mirror of the existing `ReceiverStrategy`, so senders are swappable via config/CLI exactly like receivers.

```python
class SenderStrategy(ABC):
    @abstractmethod
    def choose_garbling(self, true_quality, round_num, total_rounds) -> Garbling:
        """Pick this round's garbling (a matrix row distribution, σ, or message policy)."""
    @abstractmethod
    def update(self, true_quality, signal, action,
               sender_payoff, receiver_payoff) -> None:
        """Learn from the completed round."""
    @abstractmethod
    def reset(self) -> None: ...
    def get_diagnostics(self) -> dict: return {}
```

**Sender learners to implement (mirroring the receiver zoo):**

| Strategy | Rule | Notes |
|---|---|---|
| `sender-regret` | Regret-matching over the garbling-strategy set | parameter-free; converges to correlated eq. |
| `sender-hedge` | Multiplicative weights over garblings | `η = √(8 ln K / T)` |
| `sender-bandit` | Per-context bandit over σ (continuous channel) | SW-UCB / EXP3 for non-stationary receiver |
| `sender-best-response` | Best-response to the *empirically estimated* receiver decision rule | model-based; fastest convergence |
| `sender-llm` | LLM choosing garbling/message in context | the behavioral arm |

The empirical-best-response sender requires estimating the receiver's `P(BUY | signal)` from history — a clean, data-efficient learner that gives the sharpest convergence signal.

**Implementation:**
```
core/agents/strategies_sender/   # mirror of strategies/ for the sender
core/agents/sender.py            # delegate to a SenderStrategy (as ReceiverAgent does)
core/game.py                     # call sender.strategy.update() after payoffs
cli/commands/run.py              # --sender-strategy flag (mirror of --receiver-strategy)
```

## 4. The state-dependent variant (so "garbling rate" has a CS meaning)

To connect to Crawford–Sobel and make convergence non-trivial, we add an optional **misalignment knob** `b`: the sender's payoff becomes `u_S = base + b·1[action=BUY]·f(θ)` so the sender's ideal action varies with the state (moving away from pure transparent motives toward CS bias). `b = 0` recovers the current transparent-motive gym; `b > 0` gives a genuine partition problem with a known `N(b)` benchmark from the solver. This lets us ask whether two learners *discover* the CS partition.

## 5. Hypotheses

- **H1 (convergence vs. a fixed receiver):** Against a fixed Bayesian receiver, a regret/bandit/best-response sender converges to the optimal feasible garbling — approaching `V_qcav` without reputation, and toward `V_cav` when the announce-mechanism of Proposal 09 supplies consistency.
- **H2 (co-adaptation can fail to converge):** Against an *adaptive* receiver, some pairings exhibit limit cycles — the sender builds trust (low garbling), exploits it (high garbling), the receiver punishes (BOCPD detects the shift), and the cycle repeats. We predict which pairings cycle vs. settle.
- **H3 (CS recovery):** In the bias-`b` variant, the realized garbling coarsens as `b` rises, tracking `N(b)`; learners approximate the partition structure without being told it.
- **H4 (the persistent honesty gap):** LLM senders under-garble relative to the *learned* optimum even with payoff feedback — the over-honesty bias is not corrected by reward signal at these horizons. The gap shrinks but does not close with persona conditioning (link to Proposal 09 H4).

## 6. Experiment design

- **Grid:** {5 sender learners} × {receiver zoo} × {b ∈ 0…} × {horizon}.
- **Convergence diagnostics:** garbling-rate trajectory and its variance over a trailing window; distance to `{V_qcav, V_cav, CS partition}`; detection of limit cycles (autocorrelation of the garbling sequence).
- **Outcomes:** terminal garbling rate; payoff vs. benchmarks; convergence/cycle/collapse classification per pairing; LLM honesty gap vs. learned optimum.
- **Repetition:** ≥ 20 seeds/cell; report the *distribution* of terminal behavior, not a point.

## 7. Contribution & venue

- Upgrades the gym from a **receiver-learning testbed** to a **two-sided information-design testbed** — a reusable capability that all future work (competition, multi-receiver, mediation) builds on.
- A characterization of *when agentic persuasion converges*, with a benchmark-anchored map of convergence/cycle/collapse across learning-rule pairings.
- A clean measurement of the LLM honesty gap that survives reward feedback — a sharper version of the "LLMs are too honest" claim than prior single-shot work.
- Target: a learning-dynamics / multi-agent venue, with the capability itself as an artifact contribution.

## 8. Risks & limitations

- **Two adaptive sides ⇒ high variance.** Mitigation: paired seeds, large repetition, distributional reporting.
- **Convergence is sometimes to a set, not a point.** We classify outcomes (settle/cycle/collapse) rather than forcing a convergence claim.
- **Best-response sender needs enough data** to estimate the receiver rule; at very short horizons it behaves like its prior. Mitigation: report convergence *rate*, not just terminal state, and sweep horizon.
- **LLM cost** for the LLM×LLM cells. Mitigation: establish dynamics with learners; sample LLM cells.
