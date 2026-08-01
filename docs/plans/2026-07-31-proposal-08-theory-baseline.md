# Proposal 08 — Theory Baseline & Solver: Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Add a side-effect-free `core/theory/` package that computes the Bayesian-persuasion (concave-envelope), cheap-talk (quasiconcave-envelope), weak-institution (capped-concavification), and Crawford–Sobel benchmarks for any `(prior, payoffs)`, plus an information-metrics module (mutual information, Blackwell-garbling test) — turning the gym from a simulator into a benchmark-anchored instrument.

**Architecture:** All theory code lives in a new `src/garbling_gym/core/theory/` package with no dependence on the game loop or agents (pure functions over the belief simplex). It reads the game's `(prior, payoffs)` from `GameConfig`/`PayoffStructure`. A `benchmarks` bundle is computed by the solver and attached to `GameResults`; a new `gg solve` CLI prints the bundle without running a game. The default game is never changed — everything is additive and gated.

**Tech Stack:** Python ≥3.10, numpy, **scipy** (`scipy.spatial.ConvexHull`, `scipy.optimize.linprog`), pytest + hypothesis (already a dev dep). Test runner: `uv run pytest`. Exact-arithmetic assertions in the style of `test_strategy_math.py` (`pytest.approx(..., abs=1e-10)`).

---

## Scientific specification (the source of truth)

This plan implements the research proposal at `.ai/analyses/08_proposal_theory-baseline-and-solver.md`. The proposal is the **spec**; this document is the **task breakdown**. When a task says "compute X", the formal definition of X is in the proposal §3–§5 and reproduced inline below for the executor's convenience. Do not reinterpret the math — copy it.

### The four oracle cases (success = reproduce each to ≤ 1e-3)
| Oracle | Expected | Source |
|---|---|---|
| KG prosecutor–judge | prior 0.3, threshold 0.5 ⇒ `V_cav = 0.60` | proposal §4 |
| LRS central bank | `v*_χ = 3/2` for χ≥3/4; `2χ` for 2/3≤χ<3/4; `1` for χ<2/3; discontinuity at χ=2/3 | proposal §4 |
| Crawford–Sobel | `N(b)` = largest integer with `2N(N−1)b < 1`; collapses 7→5→3→2→1 over `b ∈ {.01,.02,.05,.10,.25}` | proposal §4 |
| LR ≤ KG | `V_qcav(μ₀) ≤ V_cav(μ₀)` pointwise | proposal §4 |

### A critical reconciliation the executor MUST know
The codebase is **3 states × 3 signals × 2 actions** (`core/types.py`), and the sender indirect value over a binary action (BUY/PASS) is a **step function of the posterior**. This is what the solver builds against. Do **not** invent a 3rd action. If the cav-vs-qcav gap at the default payoff/prior turns out too small to be diagnostic (flag this in Task 13), the remedy is a configurable payoff override — *not* changing the action space.

### Reference implementation already exists
The embedded JS in `.ai/analyses/research_proposals.html` (functions `concaveEnv`, `quasiEnv`, `lrsV`, lines ~472–475) computes these numerically. Port the algorithms, not the JS syntax.

---

## Conventions used throughout

- **Run all tests with:** `uv run pytest <path> -q --no-cov` (fast path) or `uv run pytest` (full, with coverage).
- **TDD discipline:** write failing test → confirm FAIL → implement → confirm PASS → commit. One logical change per commit.
- **Exact-math style:** mirror `tests/core/agents/strategies/test_strategy_math.py` — use `pytest.approx(value, abs=1e-10)` for hand-computable cases.
- **Commit messages:** conventional commits (`feat:`, `test:`, `refactor:`, `chore:`, `docs:`). No AI-trace phrasing.
- **New package layout:**
  ```
  src/garbling_gym/core/theory/
    __init__.py
    game_spec.py     # convert (prior, payoffs) -> numpy arrays the solver uses
    envelopes.py     # cav / qcav on 1-D simplex, then 2-D (|Ω|=3)
    weak_inst.py     # LRS v*_χ via gridding
    cs_partition.py  # Crawford-Sobel N(b) and cutoffs
    metrics.py       # I(θ;s), Blackwell garbling LP test, posterior MPS check
    estimate.py      # empirical channel L̂ from GameResults history
    benchmarks.py    # assemble the benchmark bundle for a GameConfig
  ```

---

## Task 0: Hygiene & prerequisites (P0 — do these first; they unblock everything)

> **Why first:** The solver needs `scipy` (absent), needs prior/payoffs to flow from config (currently hardcoded in 6 strategy files), and downstream proposals need the `experiment.py` strategy bug fixed. These are not optional polish.

### Task 0.1: Add scipy + requests to dependencies

**Files:**
- Modify: `pyproject.toml:12-24` (dependencies), `pyproject.toml:30-35` and `:44-51` (dev groups)

**Step 1: Add runtime deps.** In `pyproject.toml`, add to the `dependencies = [...]` list:
```toml
    "scipy>=1.10.0",
    "requests>=2.31.0",
```
(`requests` is already imported by `core/agents/strategies/openrouter.py:6` but undeclared — fix the latent bug now.)

**Step 2: Sync the env and verify scipy imports.**
Run: `uv sync && uv run python -c "import scipy; from scipy.spatial import ConvexHull; from scipy.optimize import linprog; print('scipy', scipy.__version__)"`
Expected: prints `scipy <version>` with no error.

**Step 3: Confirm the suite still passes.**
Run: `uv run pytest -q --no-cov 2>&1 | tail -5`
Expected: same pass count as before (no regressions from the dep change).

**Step 4: Commit.**
```bash
git add pyproject.toml uv.lock
git commit -m "chore: add scipy and requests to dependencies"
```

---

### Task 0.2: Make prior and payoffs flow from GameConfig (consolidation refactor)

**Goal:** Replace the module-level `_PRIOR` / `_PAYOFFS` / `_RECEIVER_PAYOFF` literals in the receiver strategies with values sourced from config, so the solver (and sweeps) can vary them. **This is the refactor the proposal's "arbitrary priors and payoff matrices" assumes.**

**Files (6 strategy modules + their tests):**
- Modify: `src/garbling_gym/core/agents/strategies/bayesian.py:15-16` (`_PRIOR`, `_PAYOFFS`)
- Modify: `src/garbling_gym/core/agents/strategies/game_theoretic.py:14-15`
- Modify: `src/garbling_gym/core/agents/strategies/legacy_heuristic.py:20`
- Modify: `src/garbling_gym/core/agents/strategies/llm_hybrid.py:13-14`
- Modify: `src/garbling_gym/core/agents/strategies/regret.py:17-25` (`_RECEIVER_PAYOFF`)
- Modify: `src/garbling_gym/core/agents/strategies/bandit.py:17-25` (`_RECEIVER_PAYOFF`)
- Modify: `src/garbling_gym/core/agents/strategies/__init__.py:10-72` (extend `ReceiverStrategy` ABC)
- Modify: `src/garbling_gym/core/agents/strategies/registry.py:42-47` (pass config through)
- Modify: `src/garbling_gym/core/agents/registry.py` (`create_receiver` to pass config)

**Approach (minimal, backward-compatible):**
1. Extend the `ReceiverStrategy` ABC with an optional config hook that defaults to current behavior:
   ```python
   # in ReceiverStrategy (core/agents/strategies/__init__.py)
   def configure(self, prior: Dict[str, float], receiver_payoffs: Dict[Tuple[str,str], float]) -> None:
       """Inject game parameters. Default: no-op (subclasses keep module defaults)."""
       pass
   ```
2. In each of the 6 strategies, replace the module-level literals with instance attributes initialized to the **same defaults** (`{"LOW":0.3,"MEDIUM":0.4,"HIGH":0.3}` and the −15/5/20 table), and override `configure()` to store them. Internal references (`_PRIOR[q]` → `self._prior[q]`) change at the use sites only.
3. `ReceiverStrategyRegistry.get()` signature unchanged — but add `get(name, prior=None, payoffs=None)` that constructs then calls `.configure(...)` when args provided.
4. `AgentFactory.create_receiver` passes `config.prior` and the receiver-payoff table derived from `config.payoffs` through to the registry.

**Step 1: Write a regression test FIRST** (the existing exact-math tests must still pass — they are the safety net). Add `tests/core/agents/strategies/test_config_injection.py`:
```python
def test_default_prior_unchanged_after_refactor():
    s = DirichletBayesianStrategy()
    # drive one update and check the EXACT same numbers as test_strategy_math.py
    s.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10, 20)
    pm = s._posterior_mean()
    assert pm["HIGH"]["GOOD"] == pytest.approx(2/4, abs=1e-12)

def test_configure_overrides_prior_and_payoffs():
    s = DirichletBayesianStrategy()
    s.configure(prior={"LOW":0.5,"MEDIUM":0.0,"HIGH":0.5},
                receiver_payoffs={("BUY","LOW"):-99.0,("BUY","MEDIUM"):0.0,
                                  ("BUY","HIGH"):99.0,("PASS","LOW"):0.0,
                                  ("PASS","MEDIUM"):0.0,("PASS","HIGH"):0.0})
    # with a HIGH-only prior, GOOD signal → posterior concentrates on HIGH
    s.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10, 99)
    assert s.choose_action(Signal.GOOD, 1, 1) == Action.BUY
```

**Step 2:** Run the new test → confirm FAIL (configure doesn't exist).

**Step 3:** Implement the ABC hook + per-strategy `configure` overrides + registry/factory threading. Keep all module-level literals as the defaults so nothing changes unless `configure` is called.

**Step 4:** Run the **entire** strategy test suite to prove no regression:
Run: `uv run pytest tests/core/agents/strategies/ -q --no-cov`
Expected: all previously-passing tests still pass (the exact-arithmetic suite is the proof).

**Step 5:** Run the new injection test → PASS.

**Step 6: Commit.**
```bash
git add -A
git commit -m "refactor: source prior/payoffs from config into receiver strategies

Replaces 6 module-level _PRIOR/_PAYOFFS literals with config-injected
attributes (defaults unchanged). Adds ReceiverStrategy.configure() hook.
Prerequisite for the theory solver (Proposal 08) sweeping priors/payoffs."
```

---

### Task 0.3: Fix the `experiment.py` receiver-strategy wiring bug

**Files:**
- Modify: `src/garbling_gym/cli/commands/experiment.py:224-234` (`_run_single_experiment`)

**Step 1: Write a failing test.** Add `tests/cli/test_experiment_strategy.py`:
```python
def test_experiment_uses_configured_receiver_strategy(monkeypatch, tmp_path):
    # Write a 1-experiment YAML that sets receiver_strategy: bayesian
    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text("experiments:\n  - name: t\n    rounds: 5\n    receiver_strategy: bayesian\n")
    # (The reader must map receiver_strategy through _create_game_config; see Step 3)
    ...
    # Assert: the Game actually constructed a DirichletBayesianStrategy (spy on the factory)
```
(Use `monkeypatch` on `agent_factory.create_receiver` to capture the `strategy_name` kwarg.)

**Step 2:** Run → FAIL (currently `create_receiver` is called with no `strategy_name`).

**Step 3: Fix.** In `_run_single_experiment` (`experiment.py:228`), change:
```python
    receiver = agent_factory.create_receiver(model=config.llm_model)
```
to:
```python
    receiver = agent_factory.create_receiver(
        model=config.llm_model,
        strategy_name=config.receiver_strategy,
    )
```
Also extend `_create_game_config` (`experiment.py:205-221`) to read `receiver_strategy` from the YAML into `config_kwargs`.

**Step 4:** Run the test → PASS. Run `uv run pytest tests/cli/ -q --no-cov` → all green.

**Step 5: Commit.**
```bash
git add -A
git commit -m "fix: experiment command now honors receiver_strategy from config

Was: create_receiver(model=...) with no strategy_name -> always default
heuristic. Now threads config.receiver_strategy through (bug at experiment.py:228)."
```

---

## Task 1: `game_spec.py` — the solver's view of a game

**Goal:** A single pure function converting `GameConfig` → numpy arrays the solver operates on. Everything downstream depends on this.

**Files:**
- Create: `src/garbling_gym/core/theory/__init__.py`
- Create: `src/garbling_gym/core/theory/game_spec.py`
- Test: `tests/core/theory/__init__.py`
- Test: `tests/core/theory/test_game_spec.py`

**Step 1: Write the failing test.**
```python
import numpy as np
from garbling_gym.core.config import GameConfig
from garbling_gym.core.theory.game_spec import game_spec

def test_game_spec_default_config():
    spec = game_spec(GameConfig())
    # prior over 3 states, sums to 1
    assert spec.mu0.shape == (3,)
    assert spec.mu0.sum() == pytest.approx(1.0, abs=1e-12)
    assert spec.mu0.tolist() == [0.3, 0.4, 0.3]
    # receiver payoff for the BUY action, indexed by state
    assert spec.receiver_buy.shape == (3,)
    assert spec.receiver_buy.tolist() == [-15.0, 5.0, 20.0]
    # sender payoff is state-independent: +10 on BUY
    assert spec.sender_buy.shape == (3,)
    assert np.all(spec.sender_buy == 10.0)

def test_game_spec_buy_threshold():
    # receiver buys iff E[receiver_buy | posterior] > 0
    spec = game_spec(GameConfig())
    assert spec.buy_threshold == 0.0
```

**Step 2:** Run → FAIL (`ModuleNotFoundError`).

**Step 3: Implement.** Define a `GameSpec` dataclass with `mu0: np.ndarray`, `receiver_buy: np.ndarray`, `sender_buy: np.ndarray`, `buy_threshold: float = 0.0`, plus `n_states = 3`. `game_spec(config)` reads `config.prior` (ordered LOW/MEDIUM/HIGH) and `config.payoffs`. Add an `indirect_sender_value(mu)` helper returning `+sender_buy_at_buying_states` when the receiver buys at posterior `mu` else `0` — i.e. `v_S(mu) = sum(mu)·10 = 10 if E_R[BUY|mu] > threshold else 0` (since sender payoff is flat +10). Document this clearly.

**Step 4:** Run → PASS.

**Step 5: Commit** — `feat(theory): add GameSpec converter from GameConfig`.

---

## Task 2: `envelopes.py` — concave & quasiconcave envelopes on the 1-D simplex (|Ω|=2)

**Goal:** The 1-D solver first (binary-state collapse), validated against the KG prosecutor–judge oracle. This is the lower-risk half; the 2-D (|Ω|=3) extension is Task 3.

**Files:**
- Create: `src/garbling_gym/core/theory/envelopes.py`
- Test: `tests/core/theory/test_envelopes_1d.py`

**The algorithms** (ported from `research_proposals.html` `concaveEnv` / `quasiEnv`):
- **`cav(xs, vs)`** — the **least concave majorant**: upper convex hull of the points `{(x_i, v_i)}` swept left-to-right (monotone chain, pop while the cross product indicates the new point makes the hull non-concave). Return the hull interpolated back onto `xs`.
- **`qcav(xs, vs)`** — `min(running_max_from_left, running_max_from_right)` at each `x`. (Equivalently the quasiconcave envelope on a line.)

**Step 1: Write failing tests** (unit tests on simple shapes + the KG oracle):
```python
import numpy as np
from garbling_gym.core.theory.envelopes import cav, qcav

def test_cav_of_concave_function_is_itself():
    xs = np.linspace(0, 1, 101)
    vs = -(xs - 0.5)**2          # concave
    assert np.allclose(cav(xs, vs), vs, atol=1e-12)

def test_cav_concavifies_a_bump():
    xs = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
    vs = np.array([0.0, 0.9, 1.0, 0.9, 0.0])   # already concave-ish; cav lifts endpoints
    out = cav(xs, vs)
    assert np.all(out >= vs - 1e-12)           # majorant
    # the midpoint of any two hull points lies on/above — concavity check
    for i in range(len(xs)):
        for j in range(i+2, len(xs)):
            mid = (out[i] + out[j]) / 2
            assert mid <= out[i] + 1e-9 or True  # (use a real concavity property)

def test_qcav_is_quasiconcave_majorant():
    xs = np.linspace(0,1,101)
    vs = np.where(xs < 0.5, 0.2, 0.8)  # step up
    out = qcav(xs, vs)
    assert out[0] <= out[50]            # non-decreasing up to the mode
```

**Step 2:** Run → FAIL.

**Step 3: Implement** `cav` (monotone-chain upper hull) and `qcav` (prefix-max ∩ suffix-max) as pure numpy functions. Keep them 1-D for this task.

**Step 4:** Run → PASS.

**Step 5: Commit** — `feat(theory): 1-D concave and quasiconcave envelopes`.

---

## Task 3: KG prosecutor–judge oracle (validates the cav solver)

> This is **Oracle 1 of 4**. It must pass before any downstream task.

**Files:**
- Test: `tests/core/theory/test_oracles.py` (create; will accumulate all 4 oracles)

**Step 1: Write the failing oracle test.** The prosecutor–judge: binary state, prior μ₀=0.3 (probability of "guilty"/HIGH), receiver convicts (BUY) iff posterior > 0.5, sender wants conviction (payoff 1 on BUY else 0). Expected `V_cav = 0.60`.
```python
from garbling_gym.core.theory.envelopes import cav

def test_oracle_kg_prosecutor_judge():
    # binary state: P(guilty) = 0.3 ; sender value = 1 if posterior>0.5 else 0
    xs = np.linspace(0, 1, 2001)
    vs = (xs > 0.5).astype(float)
    mu0 = 0.3
    idx = np.argmin(np.abs(xs - mu0))
    V_cav = cav(xs, vs)[idx]
    assert V_cav == pytest.approx(0.60, abs=1e-3)
```

**Step 2:** Run → confirm whether it PASSES already (the Task-2 `cav` may already satisfy this) or FAILS.

**Step 3:** If failing, debug `cav` against this oracle specifically (it's the reference). Do not proceed until green.

**Step 4:** Commit — `test(theory): KG prosecutor-judge oracle (V_cav=0.60)`.

---

## Task 4: `envelopes.py` — extend to the 2-D simplex (|Ω|=3)

**Goal:** The gym is 3-state. Concavification over a triangle is a lower-convex-hull-of-the-hypograph problem.

**Files:**
- Modify: `src/garbling_gym/core/theory/envelopes.py` (add `cav_simplex`, `qcav_simplex`)
- Test: `tests/core/theory/test_envelopes_2d.py`

**Step 1: Write failing tests.** The key property tests (use hypothesis, already a dep):
```python
from hypothesis import given, strategies as st
from garbling_gym.core.theory.envelopes import cav_simplex, qcav_simplex

@given(...)
def test_cav_simplex_majorant_and_bayes_plausible(...):
    # cav_simplex(mu_grid, v_grid)(mu0) >= v(mu0) pointwise
    # and cav is concave (midpoint test on the triangulation)
    ...

def test_cav_simplex_collapses_to_1d_when_one_state_zero():
    # degenerate prior (one state prob 0) must match the 1-D cav exactly
    ...
```

**Step 2:** Run → FAIL.

**Step 3: Implement** using `scipy.spatial.ConvexHull` on the lifted points `{(μ_x, μ_y, v)}` over a triangular grid; the concave envelope is the lower hull of the hypograph's upper surface. Validate on the boundary-collapse case against the 1-D `cav` (this is the proposal's §9 mitigation). `qcav_simplex` via superlevel-set convex hulls: `qcav v(μ) = sup{ s : μ ∈ conv{ μ' : v(μ') ≥ s } }`.

**Step 4:** Run → PASS (including the boundary-collapse agreement with Task 2's 1-D `cav`).

**Step 5:** Commit — `feat(theory): 2-D simplex concave/quasiconcave envelopes`.

---

## Task 5: `weak_inst.py` — the LRS curve v\*_χ

> Builds Oracle 2 of 4. **Depends on Tasks 1–4.**

**Files:**
- Create: `src/garbling_gym/core/theory/weak_inst.py`
- Test: `tests/core/theory/test_weak_inst.py` + add to `test_oracles.py`

**Step 1: Write the failing LRS central-bank oracle.**
```python
def test_oracle_lrs_central_bank():
    # v*_chi = 3/2 for chi>=3/4 ; 2*chi for 2/3<=chi<3/4 ; 1 for chi<2/3
    for chi, expected in [(0.5, 1.0), (0.66, 1.0), (0.70, 1.40), (0.80, 1.50), (1.0, 1.50)]:
        v = weak_inst_curve(... central-bank GameSpec ..., chi)
        assert v == pytest.approx(expected, abs=1e-3)
```
(Construct the central-bank `GameSpec` from the proposal §4 / lit-review §6 — the values $(\tfrac32, 2\chi, 1)$ come from a specific payoff structure; encode that structure as a test fixture.)

**Step 2:** Run → FAIL.

**Step 3: Implement** `weak_inst_curve(spec, chi)` by gridding γ over the simplex; for each γ, β and k are pinned by Bayes plausibility (`k·β + (1−k)·γ = μ₀`); take the max of `k·cav(v^∧γ)(β) + (1−k)·qcav(v)(γ)` subject to `(1−k)·γ(θ) ≥ (1−χ)·μ₀(θ) ∀θ`. Refine grid near the active constraint.

**Step 4:** Run → PASS.

**Step 5:** Commit — `feat(theory): LRS weak-institution curve v*_chi (+ central-bank oracle)`.

---

## Task 6: `cs_partition.py` — Crawford–Sobel N(b)

> Builds Oracle 3 of 4.

**Files:**
- Create: `src/garbling_gym/core/theory/cs_partition.py`
- Test: add to `tests/core/theory/test_oracles.py`

**Step 1: Write the failing oracle.**
```python
from garbling_gym.core.theory.cs_partition import max_partition_count

def test_oracle_crawford_sobel_Nb():
    # N(b) = largest N with 2N(N-1)b < 1
    for b, expected_N in [(0.01,7),(0.02,5),(0.05,3),(0.10,2),(0.25,1)]:
        assert max_partition_count(b) == expected_N
```

**Step 2:** Run → FAIL.

**Step 3: Implement.** `max_partition_count(b)`: largest integer N≥1 with `2*N*(N-1)*b < 1`. Also implement `cs_cutoffs(b, N)` returning the partition boundaries via `a_{i+1} = 2·a_i − a_{i−1} + 4·b`. Pure arithmetic.

**Step 4:** Run → PASS.

**Step 5:** Commit — `feat(theory): Crawford-Sobel partition N(b) and cutoffs (+ oracle)`.

---

## Task 7: `metrics.py` — mutual information + Blackwell garbling test

**Files:**
- Create: `src/garbling_gym/core/theory/metrics.py`
- Test: `tests/core/theory/test_metrics.py`

**Step 1: Write failing tests.**
```python
from garbling_gym.core.theory.metrics import mutual_information, is_garbling_of

def test_mi_identity_channel_maximal():
    L = np.eye(3)              # full revelation
    mu0 = np.array([1/3]*3)
    # I(theta;s) = H(theta) when s = theta deterministically
    assert mutual_information(L, mu0) == pytest.approx(np.log(3), abs=1e-10)  # nats

def test_mi_uniform_channel_zero():
    L = np.ones((3,3))/3       # pure noise
    mu0 = np.array([1/3]*3)
    assert mutual_information(L, mu0) == pytest.approx(0.0, abs=1e-12)

def test_is_garbling_of_identity_dominates_noise():
    L_better = np.eye(3)
    L_worse = np.ones((3,3))/3
    assert is_garbling_of(L_worse, L_better) is True    # noise is a garbling of identity
    assert is_garbling_of(L_better, L_worse) is False
```

**Step 2:** Run → FAIL.

**Step 3: Implement.**
- `mutual_information(L, mu0)`: `I(θ;s) = Σ_θ μ₀(θ) Σ_s L[s|θ] log( L[s|θ] / Σ_θ' μ₀(θ') L[s|θ'] )` (nats). Handle 0·log0 = 0.
- `is_garbling_of(L_prime, L)`: returns True iff ∃ row-stochastic K (shape 3×3, state-independent) with `L_prime = L @ K`. Solve as an **LP** via `scipy.optimize.linprog`: variables = 9 entries of K; equality constraints `L @ K = L_prime` (one per (s,θ)); inequality `K ≥ 0`; row-sum `Σ_s K[s|·] = 1`. Feasible ⇒ True.

**Step 4:** Run → PASS.

**Step 5:** Commit — `feat(theory): mutual information and Blackwell garbling LP test`.

---

## Task 8: `estimate.py` — empirical channel L̂ from a run

**Goal:** Estimate `L̂(s|θ)` from `GameResults.history` with a credible interval. Required by the metrics module and by Proposals 09/11's χ_eff / ρ_perceived estimators.

**Files:**
- Create: `src/garbling_gym/core/theory/estimate.py`
- Test: `tests/core/theory/test_estimate.py`

**Step 1: Write failing tests.**
```python
def test_estimate_channel_recovers_identity_from_identity_run():
    # Build a fake history where signal==quality every round
    history = [...]
    L_hat, ci = estimate_channel(history, credible_mass=0.9)
    assert np.allclose(L_hat, np.eye(3), atol=1e-9)
    # CI width should be ~0 for a long deterministic run
    assert ci.width.max() < 0.1

def test_estimate_channel_uniform_for_random_history():
    ...
```

**Step 2:** Run → FAIL.

**Step 3: Implement** `estimate_channel(history, credible_mass)`: Laplace-smoothed counts `L̂[s|θ] = (n_{θ,s} + 1) / (n_θ + 3)`; CI from the Dirichlet posterior (per-row concentration = counts+1). Return `(L_hat, ChannelCI)`.

**Step 4:** Run → PASS.

**Step 5:** Commit — `feat(theory): empirical channel estimation with Dirichlet credible intervals`.

---

## Task 9: `benchmarks.py` — assemble the bundle

**Goal:** One function returning the whole benchmark ladder for a `GameConfig`.

**Files:**
- Create: `src/garbling_gym/core/theory/benchmarks.py`
- Test: `tests/core/theory/test_benchmarks.py`

**Step 1: Write failing tests.**
```python
def test_benchmark_bundle_has_all_fields():
    b = compute_benchmarks(GameConfig())
    for key in ["babbling","cav","qcav","weak_inst_curve","floor","ceiling"]:
        assert key in b
    # Oracle 4: LR <= KG pointwise
    assert b["qcav"] <= b["cav"] + 1e-9

def test_benchmark_bundle_default_game_qcav_le_cav():
    # the gym's native world is qcav; cav must weakly exceed it
    ...
```

**Step 2:** Run → FAIL.

**Step 3: Implement** `compute_benchmarks(config, chi_grid=np.linspace(0,1,21))` returning a dict: `babbling` (= `v̂(μ₀)`), `cav` (Task 4), `qcav` (Task 4), `weak_inst_curve` (dict χ→value from Task 5), `floor`/`ceiling` (re-derived from config payoffs — replace the hardcoded +5/+20 in `game.py:199-205` conceptually; do not change `game.py` yet).

**Step 4:** Run → PASS (this is where **Oracle 4** "LR ≤ KG" is enforced).

**Step 5:** Commit — `feat(theory): benchmark bundle assembler (+ LR<=KG oracle)`.

---

## Task 10: Attach benchmarks to `GameResults`

**Goal:** Persist the bundle + the realized empirical channel so every run is self-describing.

**Files:**
- Modify: `src/garbling_gym/core/results.py:21-33` (add fields)
- Modify: `src/garbling_gym/core/game.py:170-221` (`_compute_summary` calls the bundle)
- Modify: `src/garbling_gym/core/storage.py:241-256` (`_results_to_dict` serializes them)
- Test: `tests/core/test_results.py` (create or extend)

**Step 1: Write failing test** that a `GameResults` from a real short game has non-`None` `benchmarks` and a `realized` channel estimate.

**Step 2:** Run → FAIL.

**Step 3:** Add to `GameResults`:
```python
    benchmarks: Optional[Dict[str, Any]] = None    # the solver bundle
    realized: Optional[Dict[str, Any]] = None      # {channel: L_hat, mi: float, ci: ...}
```
In `Game._compute_summary`, call `compute_benchmarks(self.config)` and `estimate_channel(history)` and attach. (Guard with try/except so a solver error never breaks a run — log a warning and leave them `None`.)

**Step 4:** Update `storage._results_to_dict` to include the two new keys (they're plain dicts/arrays → JSON-serializable via `.tolist()`).

**Step 5:** Run the full suite (`uv run pytest -q --no-cov`) → all green, including existing integration tests.

**Step 6:** Commit — `feat(theory): attach benchmarks + realized channel to GameResults`.

---

## Task 11: `gg solve` CLI

**Goal:** Print the benchmark table for a config without running a game (proposal §6).

**Files:**
- Create: `src/garbling_gym/cli/commands/solve.py`
- Modify: `src/garbling_gym/cli/__main__.py:43-49` (register the command)
- Test: `tests/cli/test_solve.py`

**Step 1: Write failing test** using Click's `CliRunner`:
```python
from click.testing import CliRunner
from garbling_gym.cli.__main__ import main
def test_gg_solve_default():
    res = CliRunner().invoke(main, ["solve"])
    assert res.exit_code == 0
    assert "cav" in res.output and "qcav" in res.output
```

**Step 2:** Run → FAIL (no `solve` command).

**Step 3: Implement** `solve` accepting `--prior`, `--payoffs` (optional overrides), `--chi-grid`, and the shared `--config`. Prints a `rich.table.Table` of `{babbling, qcav, cav, (value of commitment = cav − qcav)}` and the χ-grid curve.

**Step 4:** Run → PASS.

**Step 5:** Commit — `feat(cli): gg solve command to print benchmark bundle`.

---

## Task 12: Auto-attach benchmarks in `gg run` and `gg experiment`

**Files:**
- Modify: `src/garbling_gym/cli/commands/run.py` (nudge: results already carry benchmarks from Task 10; just surface them in the quick-summary print at `run.py:141-146`)
- Verify: `experiment.py` path inherits via `GameResults` (no change needed beyond Task 10)

**Step 1: Test** that `gg run` output mentions the benchmark values (extend the Task 11 CliRunner test or add a new one).

**Step 2–4:** Implement/verify/pass.

**Step 5:** Commit — `feat(cli): surface benchmark values in gg run summary`.

---

## Task 13: Validation pass + the gap diagnostic

**Goal:** Confirm all four oracles pass together, measure the cav-vs-qcav gap at the default config, and **decide** whether the gap is diagnostic enough for Proposals 09–11.

**Files:**
- Create: `docs/notes/theory_baseline_validation.md` (a short findings note)

**Step 1:** Run the full theory suite:
Run: `uv run pytest tests/core/theory/ -v --no-cov`
Expected: all four oracles green.

**Step 2:** Run `uv run gg solve` on the default config and on a few priors; record `cav`, `qcav`, and `cav − qcav` (the value of commitment).

**Step 3:** **Decision point — write it down in the note.** If `cav − qcav` at the default config is < ~0.05 (sender payoff units), the gap is too small to measure reputation/commitment effects against. In that case the remedy is a configurable payoff override (e.g., a payoff set where the receiver's BUY threshold sits inside the prior's support so the sender can profitably split beliefs) — implement it as an **additional** `GameConfig` payoff preset, *not* by changing the action space. Document which preset 09–11 should use.

**Step 4:** Commit the note — `docs: theory baseline validation + gap diagnostic`.

---

## Task 14: Methods note (the proposal's §8 deliverable)

**Files:**
- Create: `.ai/analyses/08b_theory_baseline_methods_note.md`

Write the reusable "Theoretical baseline" methods section (proposal §8): what each benchmark is, the formal objects (cite the proposal), the four oracles with their values, and the gap diagnostic from Task 13. This is the text any paper from Proposals 09–11 pastes in as its "Theoretical baseline" subsection.

**Commit** — `docs: theory baseline methods note (Proposal 08 deliverable)`.

---

## Definition of Done

- [ ] `core/theory/` package exists with `game_spec`, `envelopes` (1-D + 2-D), `weak_inst`, `cs_partition`, `metrics`, `estimate`, `benchmarks`.
- [ ] All **four oracles** pass to ≤ 1e-3 (`tests/core/theory/test_oracles.py`).
- [ ] `GameResults` carries non-null `benchmarks` and `realized` on every run; serialized to `results.json`.
- [ ] `gg solve` prints the benchmark table; `gg run` surfaces benchmarks in its summary.
- [ ] The full pre-existing suite (214 tests) still passes — **no regressions** (proof: Tasks 0.2/0.3 ran the strategy suite; Task 10 ran the integration suite).
- [ ] scipy + requests declared in `pyproject.toml`; prior/payoffs source from config; `experiment.py` strategy bug fixed.
- [ ] Task 13's gap diagnostic is documented; the default-or-preset payoff for 09–11 is named.
- [ ] Methods note (Task 14) exists.

## Risks (from the proposal §9, with executor guidance)

| Risk | Mitigation in this plan |
|---|---|
| 2-D simplex concavification boundary errors | Task 4 validates against the 1-D collapse (boundary where one state has prob 0) |
| LRS gridding resolution vs. speed | Task 5 refines the grid near the active constraint; the central-bank oracle bounds error |
| Empirical channel noise at 20–50 rounds | Task 8 returns a Dirichlet credible interval; never claim an ordering the CI doesn't support |
| cav-vs-qcav gap too small to be diagnostic (audit finding, not in proposal) | Task 13 measures it explicitly and names the payoff preset remedy |

## Non-goals (explicitly out of scope for 08)

- The commitment/credibility dial (Proposal 09).
- The `SenderStrategy` interface and sender learners (Proposal 10).
- The verifiability mask (Proposal 11).
- Any change to the game loop, the default game, or the receiver zoo's behavior (Task 0.2 preserves defaults exactly).
- Replacing the Frobenius `informativeness_score` in the existing viz/storage — the new metrics live alongside; deprecation is a separate decision.
