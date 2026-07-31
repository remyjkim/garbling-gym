# Research Proposal 09 — Reputation as Endogenous Credibility: Tracing the Weak-Institutions Value Curve with Learning and LLM Agents

**Status:** Proposal (flagship). **Depends on:** 08 (solver), and the `SenderStrategy` interface from 10. **Defines:** the commitment/credibility dial (reused by 11).

**One-line thesis.** Lipnowski–Ravid–Shishkin (2022) prove that a sender's value moves along a sharp, non-smooth curve `v*_χ` as institutional credibility χ runs from cheap talk (χ=0) to full commitment (χ=1). We turn χ from an exogenous parameter into an **emergent property of repeated interaction**, and ask: does reputation among adaptive/LLM agents manufacture an *effective credibility* `χ_eff` — and where on the qcav→cav interval does it land?

---

## 1. Research question

Classical Bayesian persuasion assumes the sender can **commit** to a signal structure. The gym, by construction (state-independent +10-for-BUY sender payoff), is the opposite extreme: transparent-motive cheap talk, whose one-shot value is the quasiconcave envelope. Between the extremes sits the weak-institution model, where the announced rule is honored only with probability χ.

> **Core question:** In a repeated game with no enforced commitment, does reputation — built through observed outcomes and the receiver's evolving trust — act as an *endogenous* credibility χ_eff that lifts the sender from the cheap-talk value toward the commitment value? And does the lift exhibit the **discontinuity** the LRS theory predicts for exogenous credibility?

This is the rigorous, benchmark-anchored form of "can reputation substitute for commitment" — but instead of asking a yes/no convergence question, we **measure where on the LRS curve the system lands** and **how χ_eff is produced**.

## 2. Positioning against the literature

- **Primary anchor — LRS (2022):** we instantiate `v*_χ` and ask whether emergent reputation reproduces it. To our knowledge, no one has *empirically traced* the weak-institutions value function with agents; the curve has only ever been drawn analytically.
- **Endpoints — KG (2011) and LR (2020):** χ_eff → 1 recovers concavification; χ_eff → 0 recovers quasiconcavification. The solver (08) supplies both as exact targets.
- **FLP (2022):** treats commitment as an exogenous experimental treatment; we make it endogenous and compare emergent χ_eff to their commitment ladder.
- **Repeated-games-with-LLMs literature** (Akata et al. 2025; the "too-honest" Homo-Silicus pattern): reframes their convergence questions as "what χ_eff do LLM agents generate, and is it payoff-rational?"

## 3. The commitment/credibility dial (shared mechanism, defined here)

We add one primitive to the gym, used by this proposal and Proposal 11.

**Round structure (announce → maybe-deviate → reveal):**

1. The sender **announces** a garbling rule `π_C` (a row-stochastic matrix), observable to the receiver.
2. Nature draws an enforcement bit: with probability **χ** the realized signal is drawn from `π_C` (the institution holds); with probability **1−χ** the sender, *after* observing θ, draws from a deviation rule `π_R(·|θ)` of its choosing (the institution fails and the sender influences the report).
3. The receiver sees the **message only**, not whether it came from `π_C` or `π_R`, and acts.
4. The true state is revealed (gym's existing mechanic); both sides update.

`χ = 1` ⇒ Bayesian persuasion (cav). `χ = 0` ⇒ transparent-motive cheap talk (qcav). This is exactly the LRS technology.

**Implementation:**
```
core/agents/sender.py        # announce(π_C); choose_deviation(θ, history) → π_R
core/mechanisms/credibility.py
    CredibilityChannel(chi: float)        # mechanical enforcement
    ReputationChannel(...)                # χ_eff emerges; no enforced bit
core/game.py                 # play_round gains the announce/enforce/reveal steps
core/results.py              # store announced π_C, realized origin, message
```

## 4. Two phases

### Phase A — Mechanical validation (no learning)
Fix χ on a grid; the sender plays the LRS-optimal influenced strategy from the solver (08). **Verify the realized sender value traces `v*_χ`**, including the kink and the discontinuity. This validates the harness end-to-end against theory and is itself a publishable "the simulator reproduces the weak-institutions curve" result.

### Phase B — Endogenous credibility (the science)
Remove the enforced bit. The sender may always deviate (χ structurally 0), but the receiver maintains a **reputation/trust state** and conditions its action on the announced rule *and* the sender's track record of announcement-vs-outcome consistency. Agents:

- **Learning agents:** sender = a `SenderStrategy` learner (Proposal 10) choosing announcements and deviations to maximize discounted payoff; receiver = the existing zoo (Dirichlet/BOCPD/regret/bandit), extended with a trust state.
- **LLM agents:** sender and receiver are LLMs reasoning over the announced rule and history in natural language.

We then **recover `χ_eff`** three independent ways and triangulate:
1. **Value-fit:** `χ_eff = argmin_χ | realized_sender_value − v*_χ(μ₀) |` — where on the curve did we land?
2. **Behavioral:** the empirical fraction of messages consistent with the announced `π_C` (honesty rate), as a direct credibility proxy.
3. **Belief-based:** the receiver's revealed trust weight on the announcement.

## 5. Hypotheses (falsifiable)

- **H1 (validation):** Phase-A realized value matches `v*_χ` within solver tolerance, reproducing the discontinuity at the LRS threshold.
- **H2 (reputation lifts credibility):** Phase-B `χ_eff` is strictly between 0 and 1 — reputation *partially* substitutes for commitment. Sender value exceeds the cheap-talk floor `V_qcav` but falls short of `V_cav`.
- **H3 (a repeated-game cliff):** There is a horizon/discount threshold below which reputation collapses (`χ_eff → 0`, value → qcav) — the repeated-game analog of the LRS discontinuity. Near the threshold, small changes in horizon produce discontinuous value drops.
- **H4 (the honesty bias has a sign):** LLM senders generate `χ_eff` **higher** than the payoff-maximizing reputation equilibrium — they are "too credible," leaving sender value on the table relative to learning agents. The three χ_eff estimators disagree in a characteristic way (behavioral honesty > value-fit) when an agent is honest beyond strategic necessity.

## 6. Experiment design

- **Conditions:** {Phase A χ-grid} × {Phase B agent pairings: learner×learner, LLM×learner, LLM×LLM}.
- **Sweeps:** horizon `N ∈ {10, 20, 50, 100}`; discount; prior `μ₀`; payoff misalignment (to move `v̂`'s non-concavity); LLM model and persona conditioning.
- **Seeds/repetition:** ≥ 20 repetitions per cell; paired seeds across agent types for variance reduction.
- **Primary outcomes:** realized sender value vs. `{V_qcav, v*_χ, V_cav}`; recovered `χ_eff` and its three estimators; trust trajectory; informativeness `I(θ;s)` over rounds.

## 7. Contribution & venue

- **The first agent-based trace of the weak-institutions value function**, and a definition of *endogenous credibility* as a measurable quantity rather than a modeling primitive.
- A bridge between the LRS information-design theory and the empirical "LLMs and repeated games" literature, with an exact theoretical benchmark (rare in that literature).
- Target: an economics-of-AI / computational-social-science venue; the Phase-A validation also supports a shorter methods/tools contribution.

## 8. Required gym modifications (summary)

- The commitment/credibility dial (§3) — new mechanism module + game-loop steps.
- A **receiver trust state** added to the zoo (a scalar/vector belief about announcement reliability).
- `χ_eff` estimators in the analysis layer (consumes solver 08).
- Reuses the `SenderStrategy` interface from Proposal 10.

Scope: substantial but self-contained; the announce/enforce/reveal loop is the only change to game orchestration, and it is gated behind a channel flag so the default game is untouched.

## 9. Risks & limitations

- **χ_eff identifiability:** the three estimators may diverge. That divergence is itself a finding (it diagnoses non-strategic honesty), but we must pre-register which estimator is primary (value-fit) to avoid post-hoc selection.
- **Horizon vs. compute:** N=100 with LLM agents is expensive. Mitigation: establish the curve with learning agents (cheap), spot-check with LLMs at chosen N.
- **Equilibrium multiplicity:** repeated games have many equilibria; we measure the *realized* selection, not "the" equilibrium, and report the distribution across seeds rather than a point claim.
