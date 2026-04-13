# Receiver Learning Modes: Implementation Task Plan

## Overview

Implement a pluggable receiver learning system that supports six distinct learning modes, unified under a common `ReceiverStrategy` interface. Each mode represents a different approach to the receiver's core problem: given a signal and game history, infer the sender's garbling behavior and choose BUY or PASS.

## Research Basis

Based on parallel deep research across five domains (see conversation 2026-04-13):
- **Bayesian learning**: Dirichlet-multinomial, BOCPD, Thompson Sampling, hierarchical HMM
- **Online/adversarial learning**: MWU/Hedge, EXP3, Regret Matching, CFR, FTRL
- **RL and bandits**: Contextual bandits, per-signal non-stationary bandits, level-k reasoning
- **LLM in-context learning**: ICL as implicit Bayes, CoT prompting, hybrid LLM+formal
- **Information economics**: Cheap talk equilibria, repeated game dynamics, mechanism design

## Key Game-Theoretic Constraints

These findings from the information economics research constrain the design:

- **Receiver floor**: 3.5/round (buy unconditionally, ignore signals) under default prior (0.3, 0.4, 0.3)
- **Receiver ceiling**: 8/round (full information, optimal play)
- **Only babbling equilibrium** exists in one-shot cheap talk (sender's state-independent payoff kills informative equilibria)
- **Receiver can only beat 3.5 by exploiting non-equilibrium sender behavior**
- **Deep RL is not viable** at 20-50 rounds (DQN, PPO, DRON, LOLA all need 10^3+ episodes)

## Design Principles

1. All modes implement the same interface — swappable via config/CLI flag
2. Each mode is self-contained with its own state management
3. Modes work across all signal types (discrete, continuous, natural language) where applicable
4. TDD: write failing tests first for each mode
5. Smallest reasonable changes per task — commit frequently

---

## Phase 0: Interface Abstraction

### Task 0.1: Define ReceiverStrategy Interface

**Goal**: Abstract base class that all learning modes implement.

**File**: `src/garbling_gym/core/agents/strategies/__init__.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from ..types import Action, Signal, AssetQuality

class ReceiverStrategy(ABC):
    """Pluggable learning strategy for the receiver agent."""

    @abstractmethod
    def choose_action(self, signal: Any, round_num: int, total_rounds: int) -> Action:
        """Decide BUY or PASS given the current signal."""
        ...

    @abstractmethod
    def update(self, signal: Any, action: Action, true_quality: AssetQuality,
               sender_payoff: float, receiver_payoff: float) -> None:
        """Learn from a completed round (called after quality is revealed)."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset state for a new game."""
        ...

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return internal state for logging/visualization. Optional."""
        return {}
```

**Tasks**:
- [ ] Create `src/garbling_gym/core/agents/strategies/` package
- [ ] Define `ReceiverStrategy` ABC in `__init__.py`
- [ ] Define `StrategyDiagnostics` typed dict for common diagnostic fields
- [ ] Write `tests/core/agents/strategies/test_interface.py` — verify ABC enforcement

**Acceptance**: ABC is importable, concrete subclass without all methods raises TypeError.

---

### Task 0.2: Refactor ReceiverAgent to Use Strategy

**Goal**: The existing `ReceiverAgent` delegates to a `ReceiverStrategy` for learning and decision-making. The current heuristic Bayesian logic becomes the first strategy implementation.

**Tasks**:
- [ ] Extract the heuristic logic from `ReceiverAgent._fallback_response()` into a new `LegacyHeuristicStrategy(ReceiverStrategy)`
- [ ] Add `strategy: ReceiverStrategy` parameter to `ReceiverAgent.__init__`
- [ ] `ReceiverAgent.make_decision()` calls `self.strategy.choose_action()` in heuristic mode
- [ ] After each round, game loop calls `strategy.update()` with revealed quality
- [ ] Existing tests must pass unchanged (behavioral equivalence)
- [ ] Update `Game.play_round()` to call `receiver.strategy.update()` after payoffs

**Files modified**:
- `src/garbling_gym/core/agents/receiver.py`
- `src/garbling_gym/core/game.py`

**Acceptance**: All existing tests pass. Behavior is identical to before.

---

### Task 0.3: Strategy Registry and CLI Integration

**Goal**: Register learning modes and select via config/CLI.

**Tasks**:
- [ ] Create `src/garbling_gym/core/agents/strategies/registry.py` with `ReceiverStrategyRegistry`
- [ ] Register `LegacyHeuristicStrategy` as `"heuristic"` (default)
- [ ] Add `receiver_strategy: str` field to `GameConfig` (default: `"heuristic"`)
- [ ] Add `--receiver-strategy` CLI flag to `gg run`
- [ ] Wire config → registry → ReceiverAgent in the game setup path
- [ ] Write test: `gg run --receiver-strategy heuristic` produces same results as before

**Acceptance**: CLI flag works, default behavior unchanged.

---

## Phase 1: Bayesian Learning Mode

### Task 1.1: Dirichlet-Multinomial Strategy

**Goal**: Proper Bayesian inference over the sender's garbling matrix using Dirichlet conjugate priors.

**File**: `src/garbling_gym/core/agents/strategies/bayesian.py`

**Algorithm**:
- Maintain 3 Dirichlet distributions (one per quality row), parameterized by alpha vectors
- Prior: `alpha[q][s] = 1.0` for all (q, s) — uniform Dirichlet (Laplace)
- After observing `(quality=q, signal=s)`: `alpha[q][s] += 1`
- Posterior mean estimate: `P_hat(s|q) = alpha[q][s] / sum(alpha[q])`
- Decision: Bayes' rule → posterior P(q|s) → E[BUY] → BUY if E[BUY] > 0

**Config parameters**:
- `prior_strength: float = 1.0` — Dirichlet concentration (higher = stronger prior)
- `forgetting_factor: float = 1.0` — multiply all alphas by this before each update (1.0 = no forgetting, 0.95 = exponential decay)

**Tasks**:
- [ ] Write failing test: `test_bayesian_updates_dirichlet_correctly`
- [ ] Write failing test: `test_bayesian_buys_on_reliable_good_signal`
- [ ] Write failing test: `test_bayesian_passes_on_unreliable_good_signal`
- [ ] Write failing test: `test_bayesian_forgetting_discounts_old_data`
- [ ] Implement `DirichletBayesianStrategy(ReceiverStrategy)`
- [ ] Run tests to confirm pass
- [ ] Register as `"bayesian"` in strategy registry
- [ ] Integration test: `gg run --receiver-strategy bayesian` completes without error

**Diagnostics** (from `get_diagnostics()`):
- Current alpha matrix (3x3)
- Posterior mean garbling estimate (3x3)
- Per-signal posterior P(quality|signal)
- Effective sample size per quality row

**Acceptance**: All tests pass, strategy is selectable via CLI.

---

### Task 1.2: Bayesian Online Change Point Detection (BOCPD)

**Goal**: Extend Dirichlet Bayesian strategy with automatic detection of sender strategy shifts.

**File**: `src/garbling_gym/core/agents/strategies/bayesian.py` (extend or subclass)

**Algorithm** (Adams & MacKay, 2007):
- Maintain a distribution over **run lengths** r (rounds since last change point)
- For each active run length, maintain a separate Dirichlet sufficient statistic
- Hazard function: `H(r) = 1/lambda` (constant, configurable expected run length)
- Each round: compute growth probabilities, change point probability, normalize
- Predictions are a mixture over run lengths
- Prune run lengths with negligible posterior mass (< 1e-4)

**Config parameters**:
- `expected_run_length: int = 20` — lambda for constant hazard function
- `prune_threshold: float = 1e-4` — drop run lengths below this posterior mass
- `prior_strength: float = 1.0` — Dirichlet concentration for each regime

**Tasks**:
- [ ] Write failing test: `test_bocpd_detects_strategy_shift`
  - Run 10 rounds with one garbling matrix, then 10 with a different one
  - After shift, BOCPD should assign high mass to short run lengths
- [ ] Write failing test: `test_bocpd_stable_sender_long_run_length`
  - 20 rounds of consistent behavior → run length posterior concentrates on r~20
- [ ] Write failing test: `test_bocpd_predictions_match_current_regime`
  - After detecting a shift, predictions should use recent data, not old
- [ ] Implement `BOCPDStrategy(ReceiverStrategy)`
- [ ] Run tests to confirm pass
- [ ] Register as `"bayesian-bocpd"` in strategy registry

**Diagnostics**:
- Run length posterior distribution
- Most probable run length
- Change point probability for current round
- Per-regime Dirichlet parameters

**Acceptance**: All tests pass, BOCPD correctly identifies strategy shifts in synthetic data.

---

### Task 1.3: Thompson Sampling Strategy

**Goal**: Bayesian exploration via posterior sampling rather than posterior mean.

**File**: `src/garbling_gym/core/agents/strategies/bayesian.py`

**Algorithm**:
- Same Dirichlet state as Task 1.1
- At decision time: sample a garbling matrix from the Dirichlet posterior (one row per quality)
- Compute posterior P(q|s) using the sampled matrix (not the mean)
- Compute E[BUY] and decide

**Config parameters**:
- Same as Dirichlet Bayesian plus:
- `sample_count: int = 1` — number of samples to average (1 = pure Thompson, >1 = approximates posterior mean)

**Tasks**:
- [ ] Write failing test: `test_thompson_samples_from_posterior`
  - With high uncertainty (few observations), decisions should vary across calls
- [ ] Write failing test: `test_thompson_converges_to_bayesian_with_data`
  - After many observations, Thompson and Dirichlet Bayesian should agree
- [ ] Implement `ThompsonSamplingStrategy(ReceiverStrategy)`
- [ ] Register as `"thompson"` in strategy registry

**Acceptance**: Tests pass, Thompson shows higher variance early, convergence late.

---

## Phase 2: Online/Adversarial Learning Mode

### Task 2.1: Regret Matching Strategy

**Goal**: No-regret learning that converges to correlated equilibrium. Assumes nothing about the sender.

**File**: `src/garbling_gym/core/agents/strategies/regret.py`

**Algorithm** (Hart & Mas-Colell, 2000):
- For each signal s, track cumulative regret `R[s][a]` for each action a
- After round t with signal s_t, quality q_t, action a_t:
  - For each action a: `R[s_t][a] += u_R(a, q_t) - u_R(a_t, q_t)`
- Strategy: `pi(a|s) = max(R[s][a], 0) / sum(max(R[s][a'], 0))` (uniform if all regrets <= 0)

**Config parameters**:
- None required (parameter-free algorithm)

**Tasks**:
- [ ] Write failing test: `test_regret_matching_uniform_at_start`
  - With no history, all regrets are 0, strategy is uniform
- [ ] Write failing test: `test_regret_matching_learns_to_pass_on_bad_signal`
  - After several BAD+LOW rounds, regret for PASS on BAD should dominate
- [ ] Write failing test: `test_regret_matching_counterfactual_updates`
  - Even actions not taken accumulate regret correctly
- [ ] Implement `RegretMatchingStrategy(ReceiverStrategy)`
- [ ] Register as `"regret-matching"` in strategy registry

**Diagnostics**:
- Cumulative regret matrix (3 signals x 2 actions)
- Current mixed strategy per signal
- Average strategy (converges to correlated equilibrium)

**Acceptance**: Tests pass, uniform start, learns from counterfactuals.

---

### Task 2.2: Multiplicative Weights (Hedge) Strategy

**Goal**: Worst-case optimal no-regret learning with O(sqrt(T ln K)) regret bound.

**File**: `src/garbling_gym/core/agents/strategies/regret.py`

**Algorithm** (Freund & Schapire, 1997):
- For each signal s, maintain weights `w[s][BUY]`, `w[s][PASS]`
- Initialize: `w[s][a] = 1` for all s, a
- Learning rate: `eta = sqrt(8 * ln(2) / total_rounds)` (optimal for known horizon)
- After round t: observe quality q_t, compute loss for each action (normalized to [0,1])
  - `loss(BUY) = (u_max - u_R(BUY, q_t)) / (u_max - u_min)` where u_max=20, u_min=-15
  - `loss(PASS) = (u_max - 0) / (u_max - u_min)`
- Update: `w[s_t][a] *= (1 - eta)^loss(a)` for both actions (full information)
- Strategy: `pi(a|s) = w[s][a] / sum(w[s])`

**Config parameters**:
- `learning_rate: float | None = None` — if None, compute optimal eta from total_rounds

**Tasks**:
- [ ] Write failing test: `test_hedge_regret_bound`
  - Over T rounds with adversarial losses, cumulative regret <= sqrt(T * ln(2) / 2)
- [ ] Write failing test: `test_hedge_adapts_to_signal_quality_correlation`
- [ ] Implement `HedgeStrategy(ReceiverStrategy)`
- [ ] Register as `"hedge"` in strategy registry

**Acceptance**: Tests pass, regret bound holds empirically.

---

## Phase 3: Game-Theoretic Learning Mode

### Task 3.1: Level-k Reasoning Strategy

**Goal**: Exploit known payoff structure via iterated best-response reasoning. Most data-efficient mode (works with zero history).

**File**: `src/garbling_gym/core/agents/strategies/game_theoretic.py`

**Algorithm** (Camerer, Ho, Chong, 2004):
- **Level 0 sender**: Garbles uniformly (complete noise matrix)
- **Level 1 receiver**: Best-responds to level-0 sender
  - P(q|s) using uniform garbling → posterior = prior → E[BUY] = 3.5 > 0 → always BUY
- **Level 2 sender**: Best-responds to level-1 receiver (who always buys)
  - Sender is indifferent (gets +10 on BUY regardless), may garble arbitrarily
- **Level 2 receiver**: Best-responds to estimated sender behavior
  - Uses Bayesian estimation from history to infer sender's actual level/strategy
  - Computes best response to the empirical garbling distribution

The strategy:
1. Start with level-1 assumption (BUY always, since E[BUY|prior] = 3.5 > 0)
2. After each round, update empirical estimate of sender's garbling matrix
3. Compute best response to empirical matrix
4. Weight between prior level-k prediction and empirical estimate based on confidence

**Config parameters**:
- `max_level: int = 2` — depth of strategic reasoning
- `empirical_weight_growth: float = 0.1` — per-round increase in weight on empirical vs. level-k prior

**Tasks**:
- [ ] Write failing test: `test_levelk_buys_unconditionally_at_start`
  - Level-1 reasoning: E[BUY|prior] > 0, so BUY
- [ ] Write failing test: `test_levelk_adapts_to_empirical_garbling`
  - After observing deceptive sender, should shift to empirical best-response
- [ ] Write failing test: `test_levelk_exploits_known_payoff_structure`
  - Given sender payoff is +10 on any BUY, the strategy should predict sender wants max BUY
- [ ] Implement `LevelKStrategy(ReceiverStrategy)`
- [ ] Register as `"level-k"` in strategy registry

**Diagnostics**:
- Current effective level
- Empirical garbling matrix estimate
- Weight on empirical vs. prior
- Predicted sender strategy

**Acceptance**: Tests pass, starts at level-1, transitions to empirical best-response.

---

## Phase 4: Bandit Learning Mode

### Task 4.1: Per-Signal Non-Stationary Bandit Strategy

**Goal**: Treat each signal as an independent 2-armed bandit problem. Handles sender adaptation natively.

**File**: `src/garbling_gym/core/agents/strategies/bandit.py`

**Algorithm** (Sliding-Window UCB variant):
- For each signal s, maintain a sliding window of the last W rounds where signal=s
- Compute mean reward of BUY and PASS within the window
- UCB exploration bonus: `bonus = c * sqrt(ln(N) / N_a)` where N = window count, N_a = action count
- Select action with highest `mean + bonus`
- Note: PASS reward is always 0 (known), so only BUY reward needs estimation

**Config parameters**:
- `window_size: int = 10` — sliding window per signal
- `exploration_constant: float = 1.0` — UCB exploration parameter c
- `variant: str = "sw-ucb"` — `"sw-ucb"` or `"exp3s"` for adversarial variant

**EXP3.S variant** (adversarial):
- Maintain weights per action per signal
- Importance-weighted reward estimates
- Mixing parameter for forced exploration: `gamma = sqrt(2 * ln(2) / total_rounds)`
- Update: `w[s][a] *= exp(gamma * r_hat / K)` with mixing

**Tasks**:
- [ ] Write failing test: `test_bandit_explores_early`
  - First few rounds should have substantial randomness
- [ ] Write failing test: `test_bandit_learns_signal_quality_mapping`
  - After enough rounds with GOOD→HIGH, should reliably BUY on GOOD
- [ ] Write failing test: `test_bandit_adapts_to_sender_shift`
  - Sliding window forgets old data after sender changes strategy
- [ ] Write failing test: `test_exp3s_handles_adversarial_sender`
  - EXP3.S variant should have bounded regret against worst-case sender
- [ ] Implement `BanditStrategy(ReceiverStrategy)` with both variants
- [ ] Register as `"bandit"` in strategy registry

**Diagnostics**:
- Per-signal action counts and mean rewards (within window)
- UCB values per signal per action
- Current mixed strategy

**Acceptance**: Tests pass, both SW-UCB and EXP3.S variants work.

---

## Phase 5: LLM Learning Modes

### Task 5.1: Hybrid LLM + Bayesian Strategy

**Goal**: Formal Bayesian computation in code, LLM qualitative reasoning for the final decision. Best of both worlds.

**File**: `src/garbling_gym/core/agents/strategies/llm_hybrid.py`

**Algorithm**:
1. Maintain Dirichlet state (same as Task 1.1)
2. Compute posterior P(q|s) and E[BUY] in code
3. Format a prompt with:
   - Pre-computed Bayesian analysis (posterior, expected values)
   - Summary statistics (deception rate, avg payoff by signal)
   - Last 10 rounds of raw history (expanded from current 5)
   - Qualitative questions (has sender shifted? trust calibration?)
4. Call LLM with structured output: `ActionChoice` with `action`, `reasoning`
5. Return the LLM's decision

**Config parameters**:
- `history_window: int = 10` — rounds of raw history in prompt
- `model: str = "gpt-4o-mini"` — LLM model identifier
- `fallback_strategy: str = "bayesian"` — strategy to use if LLM call fails

**Tasks**:
- [ ] Write failing test: `test_hybrid_includes_bayesian_stats_in_prompt`
  - Mock the LLM call, verify the prompt contains computed posteriors and E[BUY]
- [ ] Write failing test: `test_hybrid_falls_back_on_api_failure`
  - When LLM unavailable, should use the fallback strategy
- [ ] Write failing test: `test_hybrid_updates_bayesian_state_regardless`
  - The Dirichlet state should update even when LLM makes the decision
- [ ] Implement `HybridLLMStrategy(ReceiverStrategy)`
- [ ] Register as `"hybrid-llm"` in strategy registry

**Diagnostics**:
- All Bayesian diagnostics (Task 1.1)
- LLM reasoning text from most recent decision
- LLM-override count (how often LLM disagrees with Bayesian recommendation)

**Acceptance**: Tests pass, prompt includes quantitative analysis, graceful fallback.

---

### Task 5.2: Pure LLM Strategy (Improved)

**Goal**: In-context learning with expanded history, CoT prompting, and structured assessment output.

**File**: `src/garbling_gym/core/agents/strategies/llm_pure.py`

**Improvements over current implementation**:
- Expand history window from 5 to configurable (default 15)
- Add pre-computed summary statistics to prompt
- Use CoT prompting: ask the LLM to reason through strategy identification, posterior estimation, and expected value before deciding
- Structured output includes estimated quality and confidence, not just BUY/PASS

**Output model**:
```python
class ReceiverAssessment(BaseModel):
    estimated_quality: Literal["LOW", "MEDIUM", "HIGH"]
    confidence: float  # 0 to 1
    reasoning: str
    action: Literal["BUY", "PASS"]
```

**Config parameters**:
- `history_window: int = 15`
- `model: str = "gpt-4o-mini"`
- `use_cot: bool = True` — whether to prompt for chain-of-thought reasoning
- `fallback_strategy: str = "heuristic"` — fallback when API unavailable

**Tasks**:
- [ ] Write failing test: `test_pure_llm_returns_structured_assessment`
  - Output includes estimated_quality, confidence, reasoning, action
- [ ] Write failing test: `test_pure_llm_expanded_history_window`
  - Prompt includes up to 15 rounds of history
- [ ] Write failing test: `test_pure_llm_cot_prompt_structure`
  - When use_cot=True, prompt includes explicit reasoning steps
- [ ] Write failing test: `test_pure_llm_falls_back_gracefully`
- [ ] Implement `PureLLMStrategy(ReceiverStrategy)`
- [ ] Register as `"pure-llm"` in strategy registry

**Diagnostics**:
- Most recent LLM assessment (estimated quality, confidence, reasoning)
- Running accuracy of LLM quality estimates vs. true quality
- Average confidence by true quality (calibration check)

**Acceptance**: Tests pass, structured output works, CoT prompting produces explicit reasoning steps.

---

## Phase 6: Integration and Comparison

### Task 6.1: Strategy Comparison Framework

**Goal**: Run the same game with different receiver strategies and compare outcomes.

**Tasks**:
- [ ] Add `--receiver-strategy` to `gg experiment` YAML schema:
  ```yaml
  experiments:
    - name: bayesian_vs_hedge
      rounds: 50
      receiver_strategies: [bayesian, hedge, regret-matching]
  ```
- [ ] For each strategy, run a separate game with the same random seed
- [ ] Add comparison metrics to experiment output:
  - Per-strategy: receiver total, buy rate, regret vs. perfect info, regret vs. minimax (3.5/round)
  - Cross-strategy: rank by receiver payoff, statistical significance (paired t-test across rounds)
- [ ] Add strategy comparison to `gg compare` output

**Acceptance**: Can run multi-strategy experiments and compare results side-by-side.

---

### Task 6.2: Strategy Diagnostics in Visualization

**Goal**: Surface each strategy's internal state in visualizations and dashboards.

**Tasks**:
- [ ] Add `diagnostics` field to `RoundResult` (stores `strategy.get_diagnostics()` per round)
- [ ] Update `gg visualize` to show strategy-specific panels:
  - Bayesian: posterior heatmap, change point indicators
  - Regret: cumulative regret bars per signal
  - Bandit: UCB values over time
  - LLM: reasoning text, confidence calibration
- [ ] Update HTML export to include diagnostics
- [ ] Update Marimo dashboard with strategy-specific widgets

**Acceptance**: Each strategy's internal state is visible in visualizations.

---

### Task 6.3: Signal Type Compatibility Matrix

**Goal**: Ensure each strategy works (or gracefully errors) across signal types.

| Strategy | Discrete (Tier 0) | Continuous (Tier 1) | Structured (Tier 2) | Natural Language (Tier 3) |
|----------|-------------------|--------------------|--------------------|--------------------------|
| `heuristic` | Yes | No | No | No |
| `bayesian` | Yes | Needs Gaussian variant | Needs per-dimension | No |
| `bayesian-bocpd` | Yes | Needs Gaussian variant | Needs per-dimension | No |
| `thompson` | Yes | Needs Gaussian variant | Needs per-dimension | No |
| `regret-matching` | Yes | Needs discretization | Needs flattening | No |
| `hedge` | Yes | Needs discretization | Needs flattening | No |
| `level-k` | Yes | Yes (analytical) | Partial | No |
| `bandit` | Yes | Yes (LinUCB) | Partial | No |
| `hybrid-llm` | Yes | Yes | Yes | Yes |
| `pure-llm` | Yes | Yes | Yes | Yes |

**Tasks**:
- [ ] For each strategy, implement signal type detection in `choose_action()`
- [ ] Raise clear error if strategy doesn't support the signal type
- [ ] Add `supported_signal_types: list[str]` class attribute to each strategy
- [ ] Validate strategy/signal compatibility at game setup time
- [ ] Write test matrix covering all (strategy, signal_type) pairs

**Acceptance**: Invalid combinations fail fast with clear error messages. Valid combinations work.

---

## Phase 7: Documentation and Benchmarks

### Task 7.1: Benchmark Suite

**Goal**: Standard benchmark comparing all strategies under controlled conditions.

**Tasks**:
- [ ] Create `experiments/yaml/strategy_benchmark.yaml`:
  - 6 scenarios: honest sender, deceptive sender, adaptive sender, shifting sender, random sender, adversarial sender
  - Each scenario runs all strategies with same seed
  - 50 rounds per game, 10 repetitions per (scenario, strategy) pair
- [ ] Create `experiments/python/benchmark_analysis.py`:
  - Aggregate results across repetitions
  - Compute mean payoff, std, regret vs. floor (3.5) and ceiling (8)
  - Rank strategies per scenario
- [ ] Add `gg benchmark` command that runs the full suite

**Acceptance**: Benchmark runs, produces per-strategy per-scenario results, identifies best strategy for each sender type.

---

### Task 7.2: Update Knowledge Docs

**Goal**: Update `.ai/knowledges/06_experimental-setup-and-parameters.md` with the new learning modes.

**Tasks**:
- [ ] Add section documenting each strategy's algorithm, parameters, and theoretical properties
- [ ] Add the game-theoretic bounds (3.5 floor, 8 ceiling, babbling equilibrium)
- [ ] Document the signal type compatibility matrix
- [ ] Update README.md with strategy selection documentation

**Acceptance**: Documentation is complete and accurate.

---

## Implementation Order and Dependencies

```
Phase 0 (Interface) ──────────────────────────────────────────────┐
  0.1 Define interface                                            │
  0.2 Refactor ReceiverAgent (depends on 0.1)                     │
  0.3 Registry + CLI (depends on 0.2)                             │
                                                                  ▼
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Phase 1      │ Phase 2      │ Phase 3      │ Phase 4      │  (parallel after Phase 0)
│ Bayesian     │ Online       │ Game-theory  │ Bandit       │
│ 1.1 Dirichlet│ 2.1 Regret   │ 3.1 Level-k  │ 4.1 SW-UCB   │
│ 1.2 BOCPD    │ 2.2 Hedge    │              │     + EXP3.S │
│ 1.3 Thompson │              │              │              │
└──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
Phase 5 (LLM modes — depends on Phase 1.1 for hybrid fallback)
  5.1 Hybrid LLM + Bayesian
  5.2 Pure LLM (improved)
       │
       ▼
Phase 6 (Integration — depends on 2+ strategies existing)
  6.1 Comparison framework
  6.2 Diagnostics visualization
  6.3 Signal type compatibility
       │
       ▼
Phase 7 (Docs + Benchmarks — depends on Phase 6)
  7.1 Benchmark suite
  7.2 Update knowledge docs
```

**Estimated total**: ~30 tasks across 8 phases. Phases 1-4 are independent and can be worked in parallel after Phase 0.
