# Research Proposal 08 — The Theory Baseline: An Equilibrium Solver and Information-Metric Layer for the Garbling Gym

**Status:** Proposal (foundation). **Depends on:** nothing. **Consumed by:** Proposals 09, 10, 11.

**One-line thesis.** Every behavioral experiment the gym can run is currently *descriptive* — we observe payoffs and a Frobenius "informativeness" number, but we never measure them against what theory says *should* happen. This proposal builds the missing analytical ground-truth layer — a solver that computes the Bayesian-persuasion (concave-envelope), cheap-talk (quasiconcave-envelope), and weak-institution (capped-concavification) values for any configured game, plus a proper information-theoretic metric module — so that all downstream experiments measure *deviation from a computed benchmark* rather than raw outcome.

---

## 1. Motivation

The gym already encodes, without naming it, a precise theoretical object. The default payoff structure gives the sender **+10 for any BUY regardless of the true state**. A state-independent sender payoff is exactly the **transparent-motives** assumption of Lipnowski & Ravid (2020). The receiver-learning-modes design doc already records the consequence — *"only the babbling equilibrium exists in one-shot; receiver floor 3.5/round, ceiling 8/round"* — which is the quasiconcave-envelope collapse, derived informally.

What the gym lacks is the machinery to compute these benchmarks for arbitrary priors and payoff matrices, and to compare a realized run against them. Without it:

- "Informativeness" is a Frobenius distance from the identity matrix — a geometric proxy with no decision-theoretic meaning and no comparability across signal types (matrix vs. continuous vs. natural-language channels, per Proposal 07).
- We cannot say whether a learning receiver reaches the cheap-talk value, the commitment value, or neither.
- We cannot quantify the "honesty gap" of an LLM sender, because there is no optimum to subtract from.

This proposal is the prerequisite that turns the gym from a simulator into an **instrument**.

## 2. Positioning against the literature

This layer operationalizes the geometry shared by the six-paper review:

- **Kamenica & Gentzkow (2011)** — the **concave envelope** `cav v̂` of the sender's indirect value is the full-commitment (Bayesian-persuasion) value. The solver must compute it.
- **Lipnowski & Ravid (2020)** — the **quasiconcave envelope** `qcav v` is the transparent-motive cheap-talk value. The gym *is* this world by construction; the solver makes its value explicit.
- **Lipnowski, Ravid & Shishkin (2022)** — the **capped concavification** `v*_χ` interpolates the two as a function of institutional credibility χ. Proposal 09 lives entirely inside this curve; the solver must produce it.
- **Crawford & Sobel (1982)** — the **partition equilibrium** and `N(b)` for the state-*dependent* variant we will add (Proposal 10/11 sweeps a bias `b`).
- **Milgrom (1981)** — the Blackwell/MLRP order on experiments grounds the *information metrics*: whether one realized channel is a garbling of another, and the mutual-information ordering.
- **Blackwell (1951/53)** — the metric module's backbone: garbling = a state-independent Markov post-processing; informativeness = mean-preserving spread of posteriors.

## 3. The formal objects to compute

Let the state be `θ ∈ Ω` (the gym's `AssetQuality`, |Ω| = 3), prior `μ₀ ∈ Δ(Ω)`, signal `s ∈ S`, receiver action `a ∈ A`. A signal/garbling structure is a row-stochastic matrix `π(s|θ)`. The receiver best-responds at posterior `μ`; the sender's **indirect value** is

```
v̂(μ) = max_{a ∈ BR(μ)} u_S(a, μ).
```

The solver computes, on a triangulated grid over the belief simplex `Δ(Ω)`:

1. **Babbling value** `v̂(μ₀)` — no information.
2. **Commitment value (KG):** `V_cav(μ₀) = cav v̂ (μ₀)` — concave envelope, via the upper concave hull of `{(μ, v̂(μ))}` over the simplex (for |Ω| = 2 this is the 1-D hull; for |Ω| = 3 it is a 2-D lower-convex-hull-of-the-hypograph computation).
3. **Cheap-talk value (LR):** `V_qcav(μ₀) = qcav v (μ₀)` where `v(μ) = max V(μ)` is the best sender payoff sustainable at `μ`. For |Ω| = 2, `qcav v(μ) = min(running-max-from-left, running-max-from-right)`. For the simplex, compute via superlevel-set convex hulls: `qcav v(μ) = sup{ s : μ ∈ conv{ μ' : v(μ') ≥ s } }`.
4. **Weak-institution curve (LRS):** for each `χ ∈ [0,1]`,
   ```
   v*_χ(μ₀) = max_{β, γ, k}  k · cav(v^{∧γ})(β) + (1−k) · v^CT(γ)
   s.t.   k·β + (1−k)·γ = μ₀,
          (1−k)·γ(θ) ≥ (1−χ)·μ₀(θ)   ∀θ,
   where  v^{∧γ}(μ) = min(v(μ), v^CT(γ)),  v^CT = qcav v.
   ```
   Solved by gridding `γ` over the simplex (then `β`, `k` are pinned by Bayes-plausibility) and taking the max subject to the credibility-feasibility constraint.
5. **CS partition (state-dependent variant):** cutoffs `a_{i+1} = 2a_i − a_{i-1} + 4b`, `N(b) = ` largest integer with `2N(N−1)b < 1`, for the bias-`b` game introduced in Proposals 10–11.
6. **Receiver benchmarks:** the existing floor (3.5/round) and ceiling (8/round) re-derived from the configured payoff/prior rather than hard-coded.

## 4. Built-in correctness tests (this is why it is trustworthy)

The solver is validated against closed-form results we already hold:

- **KG prosecutor–judge:** prior 0.3, threshold 0.5 ⇒ `V_cav = 0.60`. (Matches the interactive figure already built.)
- **LRS central bank:** `v*_χ = 3/2` for χ ≥ 3/4; `2χ` for 2/3 ≤ χ < 3/4; `1` for χ < 2/3, with the discontinuity at χ = 2/3. The solver must reproduce this piecewise curve exactly.
- **CS:** `N(b)` collapses 7→5→3→2→1 over `b ∈ {.01,.02,.05,.10,.25}`.
- **LR ≤ KG:** `V_qcav(μ₀) ≤ V_cav(μ₀)` pointwise, with equality iff `v̂` is already quasiconcave at `μ₀`.

These are not aspirational — they were computed numerically while building the literature-review figures, so the solver has a reference oracle from day one.

## 5. The information-metric module

Replace/augment the Frobenius score with decision-theoretic and information-theoretic measures, each computable from the realized empirical channel `L̂(s|θ)` estimated from a run:

- **Mutual information** `I(θ; s)` and conditional entropy `H(θ|s)` — model-free informativeness, comparable across channel tiers (Proposal 07).
- **Blackwell test:** given two realized channels, decide whether one is a garbling of the other (LP feasibility of a state-independent kernel `K` with `L' = L·Kᵀ`). Yields a partial order on runs.
- **Posterior-mean mean-preserving-spread check** — the Gentzkow–Kamenica (2016) Rothschild–Stiglitz characterization, for the binary/continuous variants.
- **Receiver-accuracy / discriminability** — the universal cross-channel proxy from Proposal 07, so matrix, continuous, and NL channels share one comparable axis.
- **Le Cam deficiency** (optional, later) — distance from Blackwell dominance.

## 6. Required gym modifications

A new theory package, cleanly separated from the simulation core:

```
src/garbling_gym/core/theory/
  envelopes.py     # cav / qcav on the simplex; CS cutoffs
  weak_inst.py     # LRS capped-concavification program v*_χ
  metrics.py       # I(θ;s), Blackwell garbling test, MPS check, deficiency
  benchmarks.py    # assemble per-config benchmark bundle (floor, ceiling,
                   #   babbling, cav, qcav, v*_χ) for any (prior, payoffs)
  estimate.py      # estimate empirical channel L̂ from a GameResults
```

Integration:

- `core/results.py` — `GameResults` gains a `benchmarks` field (the computed bundle) and `realized` field (empirical channel, MI, gaps).
- New CLI `gg solve --prior … --payoffs … [--chi-grid …]` prints the benchmark table for a config without running a game.
- `gg run` / `gg experiment` automatically attach benchmarks to stored results.
- New tests `tests/core/theory/` covering the four oracle cases in §4.

Scope: moderate. Pure computation, no agent changes, no game-loop changes. It is additive and side-effect-free, so it cannot break existing runs.

## 7. Shared-infrastructure map (how the four proposals compose)

```
08  Theory baseline (solver + metrics)            ← this doc, foundation
      │  provides benchmarks to all
      ├──────────────┬──────────────────┬───────────────────┐
      ▼              ▼                  ▼                   ▼
09  Reputation→χ   10 Sender learning   11 Verifiability×ρ   (future: competition,
    (LRS curve)       (SenderStrategy)     (FLP w/ LLMs)       multi-receiver)
      │                  │                  │
      └── commitment dial defined in 09, reused by 11
                         └── SenderStrategy interface defined in 10, reused by 09/11
```

Two shared *mechanism* primitives are defined in their primary docs and reused:
- the **commitment/credibility dial** (announce → maybe-deviate → reveal) — defined in Proposal 09, reused by 11;
- the **`SenderStrategy` interface** (a sender-side mirror of the receiver zoo) — defined in Proposal 10, reused by 09 and 11.

## 8. Deliverables & success criteria

- The `theory/` package reproduces all four oracle cases in §4 to ≤ 1e-3.
- Every `GameResults` carries `benchmarks` and `realized`; `gg solve` prints a benchmark table.
- A short methods note (this layer) usable verbatim as the "Theoretical baseline" section of any paper produced by Proposals 09–11.

## 9. Risks & limitations

- **Simplex concavification at |Ω| = 3** is a 2-D convex-hull problem; correct but needs care at the boundary. Mitigation: validate on the 2-D collapse (merge two states) against the 1-D oracle.
- **The LRS program** is solved by gridding; resolution trades accuracy for speed. Mitigation: refine around the active constraint; the central-bank oracle bounds the error.
- **Empirical-channel estimation** is noisy at 20–50 rounds. Mitigation: report posterior credible intervals on `L̂`; never claim a Blackwell ordering that the credible region does not support.
