# Materialization Plan — Turning the Four Garbling-Gym Research Proposals Into a Buildable Program

**Status:** Analysis report (read by the author before any code is written).
**Scope:** A rigorous, code-grounded review of *what it would take* to materialize Proposals 08–11, grounded in a full read of the `bayesian-persuasion/.ai/analyses` source documents and a deep audit of the `garbling-sims` codebase as it exists today.
**Method:** Every claim about "the code does/doesn't have X" is backed by a file:line reference verified against the current tree (audit performed 2026-07-31).

---

## 0. The one-sentence finding

**The four proposals describe a research program for a gym that does not yet exist in the form they assume.** The repository today is a *receiver*-learning testbed with a *fixed, non-learning* sender, no equilibrium solver, and no commitment/credibility/verifiability primitives — and the proposals correctly identify most of these as gaps to fill. But the proposal documents also leak several assumptions that the current code contradicts (3 actions where there are 2; a "commitment dial" described as reused across proposals; payoffs/priors assumed configurable that are in fact hardcoded across 5 files). Materializing the program is therefore **two layers of work, not one**: (1) build the missing infrastructure the proposals presume, and (2) reconcile the proposals' stated assumptions with the code's actual structure. This document is the concrete plan for both.

The plan below is sequenced so that each step is independently verifiable and so that the **foundation (Proposal 08) is real before any claim depending on a benchmark is made** — because the entire empirical value of 09–11 is "measured against a computed optimum," and that optimum does not yet exist.

---

## 1. What I read, and what state each artifact is in

### 1.1 The `bayesian-persuasion/.ai/analyses` source documents

| File | What it is | State |
|---|---|---|
| `01_literature_review_information_economics_katex.md` | Six-paper review: Milgrom '81 → Crawford–Sobel '82 → Kamenica–Gentzkow '11 → Lipnowski–Ravid '20 → Fréchette–Lizzeri–Perego '22 → Lipnowski–Ravid–Shishkin '22. | **Complete.** This is the intellectual scaffolding. Every proposal anchors to specific results here (KG prosecutor–judge value 0.60; LRS central-bank curve $(\tfrac32, 2\chi, 1)$; CS $N(b)$ collapse). |
| `research_proposals.html` | Interactive deck for proposals 08–11. | **The canonical source for the proposals** — more polished and slightly more precise than the `.md` drafts. Read in full. Figures embed the actual JS that *computes* the envelopes (`concaveEnv`, `quasiEnv`, `lrsV`), which is directly portable to Python. |
| `02_visual_design_system.md` | Reusable visual language (Manrope/Newsreader/Inter, paper palette, accent-per-track). | Reference for any future HTML artifacts. Not load-bearing for the research. |

### 1.2 The `garbling-sims/.ai/analyses` proposal drafts (08–11) + intro

The four `.md` proposals mirror the HTML deck and add detail (formal objects, gym-modification specs, risks). `intro_v2_hidden-design.md` is the narrative framing ("hidden design" — the sender doesn't choose the information, they choose *the matrix*). All read in full.

### 1.3 The codebase — audited claim-by-claim

The rest of this report rests on the following **verified** facts about `/Users/pureicis/dev/garbling-sims` (references to `src/...` mean `src/garbling_gym/...`):

| Claim | Evidence |
|---|---|
| **Sender is not a learner; no `SenderStrategy` ABC exists.** | `core/agents/sender.py:25` `SenderAgent(LLMAgent)` is a single concrete class. No `update()`/`learn()`. `choose_strategy()` returns one of 6 string keys. Game loop never calls any sender-learning method (`core/game.py:97-103` calls only `receiver.learn`). |
| **Garblings are 6 discrete fixed matrices, chosen by name.** | `core/strategies/builtin.py:9-55` `BUILTIN_STRATEGIES`; the game looks up `strategy_registry.get(name)` (`game.py:79-80`), not a continuous matrix. |
| **No solver / no `solve` command / no `cav`/`qcav`/Blackwell-compute / no MI or entropy anywhere.** | `grep -rniE "solve|concave env|qcav|\bcav\b|blackwell.*test|mutual info" src/` → **zero hits**. "Blackwell" appears only as prompt/docstring narrative. |
| **`scipy` is not a dependency** (needed for LP / convex hull / optimization). | `pyproject.toml`: runtime deps have numpy, pandas, matplotlib, etc.; **no scipy**. `requests` is imported (`strategies/openrouter.py:6`) but **undeclared**. |
| **Payoffs: sender truly state-independent +10 on BUY; receiver −15/+5/+20.** | `core/payoffs.py:16-27`. 3 states, 3 signals, **2 actions** (BUY/PASS) — `core/types.py:21-25`. |
| **Prior (μ₀=0.3/0.4/0.3) and payoffs are hardcoded as module literals in 5 strategy files.** | `_PRIOR`/`_PAYOFFS`/`_RECEIVER_PAYOFF` duplicated in `bayesian.py:15`, `game_theoretic.py:14-15`, `legacy_heuristic.py:20`, `llm_hybrid.py:13-14`, `regret.py:17-25`, `bandit.py:17-25`. Not threaded from config. |
| **No bias/misalignment knob, no commitment dial, no verifiability flag, no persona conditioning.** | `core/config.py:10-46` `GameConfig` has 6 fields only; none of these exist. `grep` confirms zero source hits. Revelation is always full and unconditional (`game.py:97-103`). |
| **"Informativeness" is a Frobenius-distance-from-identity heuristic**, wired deep into results/storage/viz. | `core/strategies/base.py:51-71`; consumed in `game.py`, `results.py`, `database.py`, `csv.py`, `html.py`, `ascii.py`. Not a Blackwell/decision-theoretic quantity. |
| **`GameResults` has no `benchmarks` field, no empirical-channel estimate.** | `core/results.py:21-33`. Only `perfect_info_benchmark` (a trivial +5/+20 receiver sum) and per-round ground-truth `garbling_info`. |
| **Latent bug: `experiment` command ignores `receiver_strategy`.** | `cli/commands/experiment.py:228` calls `agent_factory.create_receiver(model=config.llm_model)` with **no `strategy_name`** → batch runs always use default `"heuristic"`. |
| **Receiver math is genuinely sophisticated** (9 strategies; exact-arithmetic + Hypothesis property tests). | `test_strategy_math.py` (32 tests): Dirichlet conjugacy, Hart–Mas-Colell regret matching with verified regret bounds, Hedge with optimal η, SW-UCB, EXP3.S. ~214 test functions across 19 files. |

---

## 2. The program as a dependency graph (and why ordering matters)

The proposals describe four projects but they are **not** four independent builds. The HTML deck's own dependency DAG (`fig0b`) and Proposal 08's §7 make the structure explicit:

```
08  Theory baseline (solver + metrics)            ← foundation; everything leans on this
      │  provides the benchmark every experiment measures against
      ├──────────────┬──────────────────┬───────────────────┐
      ▼              ▼                  ▼                   ▼
09  Reputation→χ   10 Sender learning   11 Verifiability×ρ
    (LRS curve)       (SenderStrategy)     (FLP w/ LLMs)
      │                  │                  │
      └── commitment/credibility dial (defined in 09, reused by 11)
                         └── SenderStrategy interface (defined in 10, reused by 09 & 11)
```

Two **shared mechanism primitives** are each *defined* in one proposal and *consumed* by others:

1. **The commitment/credibility dial** (announce π_C → honored w.p. χ else sender deviates → receiver sees message only) — *defined in 09 §3, reused by 11*.
2. **The `SenderStrategy` interface** (a sender-side mirror of the receiver zoo) — *defined in 10 §3, reused by 09 and 11*.

**The critical sequencing insight:** the proposals were written as if 08's solver already exists (every experiment in 09–11 measures against `V_qcav`, `V_cav`, `v*_χ`). It does not. **08 must be built and validated first, full stop** — otherwise 09–11 cannot make any benchmark-anchored claim, which is their entire contribution. Within 08, the only external dependency is `scipy`.

The recommended build order, with rationale:

1. **08** (solver + metrics) — unblocks everything; pure additive computation; cannot break existing runs.
2. **10's interface half** (`SenderStrategy` ABC + the cheapest learner) — unblocks 09 and 11, and is itself a clean capability win.
3. **09's mechanism half** (the credibility dial, mechanical `χ` only) — Phase-A validation only; produces a publishable "simulator reproduces the LRS curve" result *using the solver from step 1*.
4. **10's full sender zoo + convergence science** — the two-sided learning study.
5. **09's Phase B** (endogenous χ_eff) + **11** (FLP replication) — the flagship science; both depend on all of the above.

This ordering also matches the **venue strategy** the proposals imply: 08 + 10-interface + 09-Phase-A together constitute a *methods/tools artifact* that can be released early, de-risking the flagship empirical papers.

---

## 3. Proposal 08 — The Theory Baseline (the foundation; build first)

### 3.1 What the proposal asks for
A side-effect-free, additive `theory/` package computing, on the belief simplex: the babbling value, the commitment value $V_{cav}=\operatorname{cav}\hat v(\mu_0)$, the cheap-talk value $V_{qcav}=\operatorname{qcav}v(\mu_0)$, the weak-institution curve $v^*_\chi$, the CS partition $N(b)$, plus an information-metric module (mutual information, a Blackwell-garbling test, a posterior MPS check). Delivered with a `gg solve` CLI and an **oracle** that reproduces three closed-form results already computed during the literature review (KG 0.60; LRS $(\tfrac32, 2\chi, 1)$; CS $N(b)$ 7→5→3→2→1).

### 3.2 What the code actually has (and the gaps)
- **Math substrate is clean and ready to interface with:** the garbling matrix is a validated `(3,3)` row-stochastic `np.ndarray` (`core/strategies/base.py:9-35`); the payoff table is a dataclass (`payoffs.py`); `DirichletBayesianStrategy._alpha` is directly injectable for solver-driven beliefs (already exercised by Hypothesis tests). `hypothesis` is already a dev dep.
- **Everything theoretical is net-new:** no `cav`/`qcav`, no Blackwell test, no MI/entropy, no scipy. The only "information" quantity is the Frobenius heuristic (`base.py:51-71`), and it is wired into 6 files.
- **Portable reference implementation exists:** the JS in `research_proposals.html:472-475` already computes `concaveEnv` (upper convex hull sweep) and `quasiEnv` (running-max-from-left/right min), and `lrsV` (`html:475`). These port to Python almost line-for-line — meaning 08 has a working numeric reference *before* writing any code.
- **Prior/payoff parameterization is a prerequisite that doesn't exist:** the solver must accept arbitrary `(μ₀, payoffs)`, but μ₀ and payoffs are hardcoded across 5 strategy modules. **Sub-step 0 of 08 is a refactor**, not new theory.

### 3.3 Concrete build plan for 08

| Step | Deliverable | Key risk / note |
|---|---|---|
| 0a | **Add `scipy` to deps; declare `requests`.** | Trivial but currently absent. |
| 0b | **Consolidate prior/payoffs into config.** Replace the 5 module-level `_PRIOR`/`_PAYOFFS` literals with a single config-sourced source (inject into strategies). | Touches `bayesian.py`, `game_theoretic.py`, `legacy_heuristic.py`, `llm_hybrid.py`, `regret.py`, `bandit.py`. Must not regress the exact-arithmetic tests in `test_strategy_math.py`. |
| 0c | **Fix the `experiment.py` strategy-wiring bug** (`:228` add `strategy_name=config.receiver_strategy`). | Any sweep over receivers (09–11 need this) is broken without it. |
| 1 | `theory/envelopes.py` — `cav` / `qcav` on the 1-D simplex first, then the 2-D simplex for |Ω|=3 (lower convex hull of the hypograph). Port `concaveEnv`/`quasiEnv` from the HTML JS. | 2-D concavification at |Ω|=3 needs care at the boundary (Proposal 08 §9). Validate on the 2-D collapse (merge two states) against the 1-D oracle. |
| 2 | `theory/weak_inst.py` — $v^*_\chi$ by gridding γ over the simplex; `β,k` pinned by Bayes-plausibility; feasibility constraint $(1{-}k)\gamma(\theta)\ge(1{-}\chi)\mu_0(\theta)$. | Resolution/speed tradeoff (§9); refine around the active constraint. |
| 3 | `theory/metrics.py` — `I(θ;s)` (empirical MI from `L̂`), **Blackwell garbling test** as an LP (`L' = L Kᵀ`, state-independent `K`, feasible via `scipy.optimize.linprog`), posterior MPS check. | This is the principled replacement for the Frobenius heuristic. |
| 4 | `theory/estimate.py` — estimate empirical channel `L̂` from a `GameResults`/history with credible intervals. | Noisy at 20–50 rounds (§9): report posteriors, never claim an ordering the credible region doesn't support. |
| 5 | `theory/benchmarks.py` — assemble the bundle (floor, ceiling, babbling, cav, qcav, v*_χ) for any config. | — |
| 6 | `GameResults` gains `benchmarks` + `realized` fields; `gg solve` CLI; auto-attach in `gg run`/`experiment`. | Additive; default game untouched. |
| 7 | `tests/core/theory/` — **the four oracle cases** to ≤1e-3: KG prosecutor–judge = 0.60; LRS $(\tfrac32,2\chi,1)$ incl. the χ=2/3 discontinuity; CS $N(b)$ collapse; `V_qcav ≤ V_cav` pointwise. | *These were already computed numerically for the lit-review figures*, so the reference exists from day one — the proposal's strongest trust argument. |

**Decision worth making explicit before coding 08:** the receiver has **2 actions (BUY/PASS)**, not 3. The proposals' prose occasionally gestures at a richer action set, but the code, payoffs, and tests are all binary. The sender indirect value $\hat v(\mu)=\max_{a\in BR(\mu)}u_S(a,\mu)$ over a binary action is a step function of the posterior — which actually *simplifies* the concavification (it's a step envelope), but it constrains how dramatic the cav-vs-qcav gap can be. **Recommendation: build the solver to the actual 2-action game, and if 09–11 need a richer value function for a visible cav/qcav gap, add a configurable action/payoff set as a deliberate sub-step rather than assuming it.** This is the single most important reconciliation between the proposals and the code.

### 3.4 Why 08 is the right first investment
- It is **additive and side-effect-free** (the proposal's own framing): a new package, no game-loop or agent changes, so it cannot break the 214 existing tests.
- It **converts the gym from a simulator into an instrument** — the prerequisite for every benchmark-anchored claim in 09–11, and a reusable "Theoretical baseline" methods section for any downstream paper.
- It has a **ready oracle**, which is rare; the trust story is concrete from commit one.

---

## 4. Proposals 09 / 10 / 11 — the science layer (built on 08)

These three share so much infrastructure that planning them in isolation wastes effort. I cover the **mechanism primitives** they each define, the **science** they each pursue, and the **gaps** the audit surfaced.

### 4.1 Shared primitive A — the `SenderStrategy` interface (defined in 10, reused by 09 & 11)

**What it is:** a sender-side mirror of the existing `ReceiverStrategy` ABC (`core/agents/strategies/__init__.py:10-72`), so senders are swappable via config/CLI exactly like receivers. Proposal 10 §3 specifies the interface:

```python
class SenderStrategy(ABC):
    def choose_garbling(self, true_quality, round_num, total_rounds) -> Garbling: ...
    def update(self, true_quality, signal, action, sender_payoff, receiver_payoff) -> None: ...
    def reset(self) -> None: ...
    def get_diagnostics(self) -> dict: ...
```

**Gap vs. code — and a reconciliation needed.** The current sender (`core/agents/sender.py`) returns a **string name** into a fixed registry of 6 matrices; it never holds a matrix. The proposal's `choose_garbling()` returns a `Garbling` (a matrix / σ / message policy). These are different abstraction levels. **Two concrete decisions:**

1. Does `SenderStrategy.choose_garbling()` return a *matrix object* (enabling continuous/learned garblings, which 10's bandit/hedge senders need) or a *name* (preserving the current discrete zoo)? **Recommendation: return a matrix object** (`GarblingStrategy` or a lightweight equivalent), and treat the existing 6 matrices as one concrete sender-strategy's action set. This matches the proposal's intent and unblocks 10's continuous-σ bandit sender.
2. The sender currently **lives at the agent layer and inherits from `LLMAgent`**; the receiver separates *agent* (`ReceiverAgent`) from *strategy* (`ReceiverStrategy`). To mirror the receiver, `SenderAgent` should be split into an agent shell + a `SenderStrategy` registry. **This is a refactor, but a small and well-patterned one** — the receiver side is the template (`agents/registry.py`, `strategies/registry.py`).

**Sender learners to implement** (Proposal 10 §3 table): `sender-regret`, `sender-hedge`, `sender-bandit`, `sender-best-response` (the sharpest convergence signal — best-response to the *empirically estimated* receiver `P(BUY|signal)`), `sender-llm`. All four non-LLM learners have **direct receiver-side analogues already implemented and tested** (`regret.py`, `bandit.py`), so the math is proven in this codebase — the work is *porting* the online-learning rules to the sender's action space, not deriving them.

### 4.2 Shared primitive B — the commitment/credibility dial (defined in 09, reused by 11)

**What it is** (Proposal 09 §3): each round the sender *announces* a garbling rule π_C (observable to the receiver); with probability χ the realized signal is drawn from π_C, with probability 1−χ the sender draws from a deviation π_R(·|θ) *after* seeing θ; the receiver sees the message only. χ=1 ⇒ Bayesian persuasion (cav); χ=0 ⇒ transparent-motive cheap talk (qcav) — exactly the LRS technology.

**Gap vs. code:** the game loop (`core/game.py:54-123`) is a single simultaneous step — sample state → sender picks garbling → signal → receiver decides → payoffs → receiver learns. There is **no announce phase, no enforcement bit, no deviation step, no notion of credibility**. `grep` for `announce|deviate|credibility` returns only prompt prose.

**Build:** a new `core/mechanisms/credibility.py` with `CredibilityChannel(chi)` (mechanical) and `ReputationChannel` (χ_eff emerges), and `play_round()` gains announce/enforce/reveal steps **gated behind a channel flag so the default game is untouched** (Proposal 09 §8). The dial reuses the solver from 08: the LRS-optimal *deviation* strategy for Phase A comes straight from `theory/weak_inst.py`.

### 4.3 Proposal 10 — Two-Sided Learning (the convergence science)

- **Question:** under which sender×receiver pairings does the garbling rate converge / cycle / collapse — and where do LLM senders sit vs. the *learned* optimum?
- **Why the code can't answer it yet:** the sender never learns (§1.3). This is, per the proposal, "the single biggest capability gap in the framework."
- **Extra primitive 10 introduces:** an optional **misalignment knob `b`** (state-dependent sender payoff `u_S = base + b·1[BUY]·f(θ)`) so "garbling rate" acquires a Crawford–Sobel partition meaning and learners can be asked to *discover* `N(b)`. This knob does **not** exist in `GameConfig` and the receiver-payoff-vs-sender-payoff split would need to be generalized. **Recommendation: build `b` as part of 10, validated against the CS `N(b)` oracle from 08 step 7.**
- **Convergence diagnostics** the proposal specifies — trailing-window variance, distance to `{V_qcav, V_cav, CS partition}`, limit-cycle autocorrelation detection — are net-new analysis code consuming 08's benchmarks.

### 4.4 Proposal 09 — Reputation as Endogenous Credibility (the flagship)

- **Question:** does emergent reputation among adaptive/LLM agents manufacture an *effective credibility* χ_eff that lifts the sender from qcav toward cav — and does the lift show the LRS discontinuity?
- **Two phases.** Phase A (mechanical χ, no learning) is a *validation* that the simulator reproduces `v*_χ` — directly enabled by 08 + the dial. Phase B (remove the enforced bit; receiver keeps a trust state; recover χ_eff three ways: value-fit, behavioral honesty rate, revealed trust) is the science.
- **New receiver-side need:** a **trust state** added to the receiver zoo (a belief about announcement reliability). The receiver zoo is already extensible via `ReceiverStrategyRegistry.register` — a `TrustReceiver` plugs in with no game-loop change.
- **The identifiability risk (§9) is real and worth pre-registering:** the three χ_eff estimators may diverge. The proposal wisely nominates **value-fit as primary** to avoid post-hoc selection. **Recommendation: encode the primary estimator in the analysis config before any Phase-B run.**

### 4.5 Proposal 11 — Verifiability × Commitment (the FLP replication)

- **Question:** do LLM agents reproduce FLP's opposite comparative statics (informativeness rises in ρ under unverifiability, falls under verifiability), and are they commitment-blind — in which *direction*, given their over-honesty?
- **Two arms:** game-theoretic agents (validate the theory slopes against the solver) and LLM agents (the new science; recover ρ_perceived). Reuses the dial (09) and the sender zoo (10).
- **The one genuinely new primitive:** a **verifiability mask** — a channel-level constraint forbidding false state-specific claims, built on Milgrom's hard-evidence logic. New `core/mechanisms/verifiability.py`; per-state admissible-message mask. **The proposal is emphatic (§9) that this must be enforced mechanically, not via prompt** — an LLM told "don't lie" is not the same as a message space that forbids the lie. This is a correctness-critical design constraint, not a style preference.
- **External comparability caveat:** matching FLP's human data is approximate (different payoffs/UI); the proposal's mitigation (match μ₀=1/3, q=1/2; report theory-relative gaps) is sound and depends on 08 being correct.

---

## 5. Cross-cutting work the proposals under-specify (but the build needs)

These items are not "in" any single proposal but the audit shows they are prerequisites for *all* of them:

1. **Dependency hygiene.** Add `scipy`; declare the already-used `requests`. Neither is currently in `pyproject.toml`.
2. **Config consolidation.** Prior and payoffs must flow from `GameConfig`, not from literals in 5 files. Without this, *no* proposal can sweep μ₀ or payoffs — and 09–11 all need to.
3. **The `experiment.py` receiver-strategy wiring bug** (`:228`). Any batch sweep over receivers is silently broken. Cheap fix; blocks credible sweeps.
4. **The 2-vs-3 action reconciliation** (§3.3). The solver, the value function, and the cav/qcav gap all depend on resolving this. Decide deliberately; do not let it drift.
5. **Empirical-channel estimation surfaced into results.** Today the channel is estimated *inside* receiver strategies for their own decisions and never persisted. The Blackwell/MI metrics (08) and the χ_eff/ρ_perceived estimators (09/11) all need a persisted `L̂` with uncertainty. This is a `GameResults` schema change.
6. **Persona conditioning for LLM agents.** 09-H4 and 11-H4 both hinge on persona modulating LLM honesty. There is **no** persona mechanism today — prompts are fixed strings (`sender.py:35-64`, `receiver.py:32-65`). This must be built for those hypotheses to be testable. (Also: the OpenRouter caller factory exists but is **unwired** — `receiver_strategy_registry.get` instantiates LLM strategies with `llm_caller=None`, so they silently fall back to heuristics. Wiring this is a prerequisite for any LLM-arm cell.)
7. **Reproducibility/seeds.** Convergence science (10) and χ_eff/ρ_perceived recovery (09/11) demand paired seeds and distributional reporting across ≥20 reps/cell. The integration tests already verify seed-based reproducibility (`test_game_flow.py`), so the substrate exists; the experiment harness needs to use it systematically.

---

## 6. Phased roadmap (what to build, in what order, with off-ramps)

| Phase | Builds | Unlocks / Off-ramp |
|---|---|---|
| **P0 — Hygiene** | scipy + requests in deps; config consolidation (prior/payoffs); fix `experiment.py` bug. | Makes the gym *parameterizable*. No science yet. |
| **P1 — 08 (foundation)** | `theory/` package: cav/qcav/v*_χ/CS, metrics (MI, Blackwell LP, MPS), empirical-channel estimate, `benchmarks`+`realized` on `GameResults`, `gg solve`, 4 oracle tests. | **Off-ramp A: a methods/tools artifact** ("the garbling gym as an instrument") is publishable here. Every downstream claim now has a benchmark. |
| **P2 — Sender learner (10-interface)** | `SenderStrategy` ABC + registry; `SenderAgent` split; ≥1 learner (best-response-to-empirical-receiver — sharpest signal). | Unblocks 09 & 11. The gym becomes genuinely two-sided. |
| **P3 — Credibility dial (09-Phase A)** | `CredibilityChannel`; announce/enforce/reveal in `play_round` (flag-gated); validate realized value traces `v*_χ` using 08's solver. | **Off-ramp B: "simulator reproduces the weak-institutions curve"** — a short, self-contained validation result. |
| **P4 — Two-sided dynamics (10-full)** | Full sender zoo (regret/hedge/bandit/LLM) + bias knob `b`; convergence/cycle/collapse diagnostics vs. 08 benchmarks. | **Off-ramp C: a learning-dynamics paper** (convergence map + persistent honesty gap). Independent of reputation. |
| **P5 — The flagships (09-Phase B + 11)** | `ReputationChannel` + trust receiver + χ_eff estimators (09); verifiability mask + ρ_perceived + FLP 2×2 with LLMs (11). Persona conditioning + OpenRouter wiring needed here. | The flagship empirical papers. Both depend on P1–P4. |

**Sequencing rationale:** P0→P1 is non-negotiable (no benchmark, no benchmarked science). P2 before P3 because the dial's deviation strategy and 10's senders are reused everywhere. P3 (Phase A) is deliberately split from P5 (Phase B) so the *harness* is validated against theory before the *science* is run — this is the proposal's own Phase A/B logic, and it de-risks the expensive LLM cells.

---

## 7. Risks the proposals name (and where the audit sharpens them)

| Risk (from proposals) | Audit note |
|---|---|
| **Simplex concavification at |Ω|=3** (08 §9). | Real; mitigate by validating the 2-D solver on the 2-D-collapse (merge states) against the 1-D oracle. |
| **Empirical-channel noise at 20–50 rounds** (08 §9). | Real and currently unaddressed — `L̂` isn't even persisted today. Must report credible intervals; never claim a Blackwell ordering the CI doesn't support. |
| **χ_eff identifiability** (09 §9) — three estimators may diverge. | Pre-register value-fit as primary (encode in config). Divergence is itself a finding (diagnoses non-strategic honesty). |
| **Equilibrium multiplicity** in repeated games (09 §9). | Measure *realized selection* across seeds, report distributions, not point claims. |
| **Faithful operationalization of verifiability** (11 §9) — must be mechanical, not prompt-level. | Correctness-critical. The mask is a hard channel constraint. |
| **H3 (11) is high-risk/high-reward** — LLMs may replicate human under-communication rather than the predicted over-communication. | The proposal is right that the design is *informative either way*; preserve this framing. |
| **LLM compute cost** at N=100 (09, 10, 11). | Establish curves with cheap learning agents; spot-check LLMs at chosen N. |

**One risk the proposals *under*-weight, surfaced by the audit:** the **2-action receiver** materially constrains how expressive the value function — and hence the cav-vs-qcav gap that 09–11 live inside — can be. If early P1 prototyping shows the gap is too small to measure against at the default payoff/prior, the remedy is a configurable action/payoff set (§3.3), not abandoning the proposals. Flag this at P1, not P5.

---

## 8. Bottom line

The four proposals form a **coherent, well-anchored program** — the literature review is solid, the dependency graph is explicit, and the empirical hooks (especially the "LLMs are too honest" gap measured against a computed optimum) are genuinely sharp and publishable. The proposals also self-aware identify most of their own gaps.

What the audit adds is the **ground truth that the gym is currently a receiver-learning testbed, not the persuasion instrument the proposals presume** — and a precise accounting of what must be true before each proposal's claims can hold:

- **Nothing** can make a benchmark-anchored claim until **08** (the solver) exists and passes its four oracles. Build it first.
- **No sweep over receivers** is trustworthy until the `experiment.py` bug is fixed and prior/payoffs flow from config.
- **No sender-side science** (10), **no reputation science** (09-Phase B), and **no FLP replication** (11) is possible until the two shared primitives (`SenderStrategy`, the credibility dial) exist — and those reuse the solver.
- The **2-vs-3 action** question and the **persona/OpenRouter wiring** gap are prerequisites the proposals don't name but the code demands.

Sequenced as P0→P1→…→P5, with off-ramps at each phase, the program is buildable incrementally and produces publishable artifacts (a methods note, a validation result, a dynamics paper) well before the flagship empirical papers — which is the healthiest possible structure for a research program of this ambition.

---

### Appendix — provenance of this report

- **Source documents read in full:** `bayesian-persuasion/.ai/analyses/{01_literature_review_information_economics_katex.md, 02_visual_design_system.md, research_proposals.html}`; `garbling-sims/.ai/analyses/{08,09,10,11}_proposal_*.md`, `intro_v2_hidden-design.md`.
- **Codebase facts:** verified by direct read and grep of `/Users/pureicis/dev/garbling-sims/src/garbling_gym/` and `tests/` on 2026-07-31; load-bearing facts re-confirmed (scipy absent, `experiment.py:228` bug, prior/payoff hardcoding across 5 files, zero solver code).
- **Math/figure references:** the concave/quasiconcave-envelope and LRS-curve computations are taken from the embedded JS of `research_proposals.html` (`concaveEnv`/`quasiEnv`/`lrsV`), which is the de facto reference implementation for Proposal 08.
