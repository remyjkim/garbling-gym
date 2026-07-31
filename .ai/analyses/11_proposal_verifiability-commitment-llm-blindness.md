# Research Proposal 11 — Verifiability × Commitment: Do LLM Agents Exhibit "Commitment Blindness"?

**Status:** Proposal (replication + extension). **Depends on:** 08 (solver), the commitment dial from 09, the `SenderStrategy` interface from 10.

**One-line thesis.** Fréchette, Lizzeri & Perego (2022) show, with human subjects, that commitment has **opposite** effects on communication depending on whether messages are verifiable — and that people misperceive commitment ("commitment blindness"). We replicate their 2×2 design with **LLM agents** in the gym, testing whether the opposite comparative statics survive, and whether LLMs are commitment-blind — and crucially, in which *direction*, because the "LLMs are too honest" prior predicts a specific and partly novel signature.

---

## 1. Research question

FLP nest cheap talk, disclosure, and Bayesian persuasion in one framework with two axes: **verifiability** (can the sender make false state-specific claims?) and **commitment** ρ (probability the announced rule is honored). Their headline: informativeness *rises* in ρ under unverifiability but *falls* in ρ under verifiability, and the two regimes converge at full commitment. Their behavioral surprise: subjects over-communicate under verifiable rules and under-communicate under unverifiable ones, as if commitment were weaker/different than it is.

> **Core question:** Do LLM agents reproduce the opposite comparative statics? Do they show commitment blindness — and given their documented over-honesty, do they break the *human* pattern in a measurable way (e.g., over-communicating under unverifiability too)?

This is a clean, citable contribution: the **first LLM replication-and-extension of a named Econometrica experimental result**, with an exact theoretical benchmark from the solver.

## 2. Positioning against the literature

- **Primary anchor — FLP (2022):** we re-run their experimental logic with LLMs instead of humans; their human data is the comparison baseline.
- **Milgrom (1981) — finally load-bearing:** the **verifiable regime `Π^V`** is hard-information / unraveling territory. Milgrom's "favorable evidence cannot be ignored" is exactly what makes disclosure informative at ρ=0. The verifiable arm is built on his MLRP/disclosure backbone.
- **KG (2011) / LR (2020):** the ρ endpoints — ρ=1 ⇒ Bayesian persuasion (both regimes), ρ=0 ⇒ cheap talk (unverifiable) vs. disclosure (verifiable). The solver supplies all four corner benchmarks.
- **Homo Silicus / VBP over-honesty:** predicts the *direction* of LLM commitment blindness, generating H3 below.

## 3. The two regimes and the commitment ladder

**Commitment ρ** reuses the dial from Proposal 09 (announce → honored w.p. ρ → else post-state revision).

**Verifiability** is a constraint on the message space:
- **`Π^U` (unverifiable):** any message allowed in any state (the current gym).
- **`Π^V` (verifiable):** the sender cannot emit a message that is false for the realized state — concretely, each message carries a *hard claim* (e.g., "quality ≥ MEDIUM") that must be true when sent. Implemented as a per-state admissible-message mask; the sender may stay silent or make a weaker true claim but cannot fabricate.

**The 2×2 corners (validation targets from the solver):**

| | ρ = 0 (no commitment) | ρ = 1 (full commitment) |
|---|---|---|
| **`Π^U` unverifiable** | cheap talk → babbling | Bayesian persuasion |
| **`Π^V` verifiable** | disclosure / unraveling | Bayesian persuasion |

## 4. Two arms

### Arm 1 — Game-theoretic agents (replication of theory)
Senders/receivers are the solver-optimal or learning agents (Proposal 10). **Validate the opposite comparative statics:** informativeness `I(θ;s)` weakly increases in ρ under `Π^U` and weakly decreases in ρ under `Π^V`, converging at ρ=1. This confirms the harness reproduces FLP's theoretical predictions and grounds the LLM arm.

### Arm 2 — LLM agents (the new science)
Senders/receivers are LLMs. Measure informativeness vs. ρ in each regime and compare the *slopes* and *levels* to (a) the theory benchmark and (b) FLP's human data. Recover an **effective perceived commitment** `ρ_perceived` (analogous to χ_eff in Proposal 09): the ρ value under which theory best matches the LLM's realized behavior. Commitment blindness = `ρ_perceived ≠ ρ_true`.

## 5. Hypotheses

- **H1 (replication):** Arm-1 informativeness shows the opposite-sign comparative statics in ρ across regimes, converging at ρ=1 — matching FLP theory within solver tolerance.
- **H2 (LLMs are commitment-blind):** Arm-2 LLM behavior is best fit by `ρ_perceived ≠ ρ_true`; LLMs do not fully internalize the commitment level.
- **H3 (the direction is partly novel):** Because LLMs over-disclose, under `Π^V` they over-communicate relative to theory (same sign as humans), but under `Π^U` they **also over-communicate** rather than under-communicating as humans do. If confirmed, LLM commitment blindness is *not* a copy of human blindness — it is over-honesty masquerading as it. This is the paper's sharpest claim.
- **H4 (persona modulates):** Strategic persona conditioning shifts `ρ_perceived` toward `ρ_true` under `Π^U` (reduces over-communication) but has little effect under `Π^V` (where truth is already enforced).

## 6. Experiment design

- **Cells:** {`Π^U`, `Π^V`} × {ρ ∈ 0, .2, .5, .8, 1} × {game-theoretic, LLM} × {LLM model, persona}.
- **Prior/threshold:** start from FLP's `μ₀ = 1/3`, threshold `q = 1/2` for direct comparability, then sweep.
- **Primary outcome:** informativeness `I(θ;s)` (and receiver accuracy for the NL channel) as a function of ρ, per regime; fitted `ρ_perceived`; slope signs.
- **Baselines:** solver corner values; FLP published human means as an external comparison series.
- **Repetition:** ≥ 20 seeds/cell; paired across regimes.

## 7. Required gym modifications

- **Verifiable-evidence regime:** per-state admissible-message mask + a hard-claim message type. New `core/mechanisms/verifiability.py`; sender respects the mask; receiver knows the regime.
- Reuses the **commitment dial** (Proposal 09) and the **`SenderStrategy`/sender learners** (Proposal 10).
- `ρ_perceived` estimator in the analysis layer (consumes solver 08).
- Experiment config schema gains `regime: U|V` and `rho`.

Scope: moderate; the only genuinely new piece is the verifiability mask. Everything else is composition of 08–10.

## 8. Contribution & venue

- A **direct LLM replication of an Econometrica experiment**, with a theoretical baseline that the original lab study could only approximate — and a novel finding (H3) that LLM commitment blindness has a different signature than human blindness.
- Directly reuses the FLP interactive figures already produced for the literature review, giving the paper a ready-made exposition of the theory.
- Target: an economics-of-AI / behavioral-and-experimental venue; strong fit for the "LLMs as economic agents" conversation.

## 9. Risks & limitations

- **Faithful operationalization of `Π^V`:** "cannot make a false claim" must be enforced mechanically, not via prompt instruction (an LLM told "don't lie" is not the same as a message space that forbids it). Mitigation: the admissible-message mask is a hard constraint at the channel level.
- **External comparability to FLP human data** is approximate (different payoffs/UI). Mitigation: match `μ₀, q` and report theory-relative gaps, not raw cross-study levels.
- **H3 is the high-risk/high-reward claim.** If LLMs *do* replicate the human under-communication under `Π^U`, that is still a clean result (LLMs mimic human blindness) — the design is informative either way.
