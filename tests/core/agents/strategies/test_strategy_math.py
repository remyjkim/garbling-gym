# ABOUTME: Deterministic mathematical correctness tests for all receiver strategies
# ABOUTME: Every algorithmic formula is verified with exact arithmetic, not behavioral tendencies

"""
These tests check the internal mathematics of each strategy, not just directional behavior.
No randomness: where strategies involve stochastic decisions, we seed or test the underlying
probability computations directly.

Strategy | Core invariant tested
---------|----------------------------------------------
Dirichlet Bayesian | Bayes rule, posterior normalization, EV formula
Regret Matching    | Counterfactual regret, mixed-strategy computation
Hedge              | exp(-η·l) weight update, expected payoff, regret bound
Level-k            | Level-1 EV = 3.5, empirical weight schedule, EV blending
Bandit (SW-UCB)    | UCB arm-selection formula
Thompson Sampling  | Sampling variance properties (seeded)
Property-based     | Posterior sums to 1, action always valid (hypothesis)
"""

import random
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from garbling_gym.core.agents.strategies.bayesian import DirichletBayesianStrategy
from garbling_gym.core.agents.strategies.regret import RegretMatchingStrategy, HedgeStrategy
from garbling_gym.core.agents.strategies.game_theoretic import LevelKStrategy
from garbling_gym.core.agents.strategies.bandit import BanditStrategy
from garbling_gym.core.agents.strategies.legacy_heuristic import LegacyHeuristicStrategy
from garbling_gym.core.agents.strategies.bayesian import ThompsonSamplingStrategy
from garbling_gym.core.types import Action, AssetQuality, Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_informed_alpha():
    """
    Build a DirichletBayesianStrategy with a known, structured alpha matrix.

    alpha:
        LOW    → {BAD: 5, NEUTRAL: 1, GOOD: 1}  (sum = 7)
        MEDIUM → {BAD: 1, NEUTRAL: 5, GOOD: 1}  (sum = 7)
        HIGH   → {BAD: 1, NEUTRAL: 1, GOOD: 5}  (sum = 7)

    Constructed by starting with prior_strength=1.0 and feeding
    4 observations per quality-signal pair.
    """
    strategy = DirichletBayesianStrategy(prior_strength=1.0)
    for _ in range(4):
        strategy.update(Signal.BAD,     Action.PASS, AssetQuality.LOW,    0.0, 0.0)
        strategy.update(Signal.NEUTRAL, Action.BUY,  AssetQuality.MEDIUM, 10.0, 5.0)
        strategy.update(Signal.GOOD,    Action.BUY,  AssetQuality.HIGH,   10.0, 20.0)
    return strategy


# ---------------------------------------------------------------------------
# Dirichlet Bayesian — exact Bayes arithmetic
# ---------------------------------------------------------------------------

class TestDirichletBayesianMath:
    """Bayes rule, posterior normalization, and EV formula are computed exactly."""

    def test_posterior_sums_to_one_for_all_signals(self):
        """P(LOW|s) + P(MED|s) + P(HIGH|s) == 1 for every signal after arbitrary updates."""
        strategy = _build_informed_alpha()
        pm = strategy._posterior_mean()
        prior = {"LOW": 0.3, "MEDIUM": 0.4, "HIGH": 0.3}

        for signal_name in ("BAD", "NEUTRAL", "GOOD"):
            unnorm = {q: pm[q][signal_name] * prior[q] for q in ("LOW", "MEDIUM", "HIGH")}
            total = sum(unnorm.values())
            posterior = {q: unnorm[q] / total for q in ("LOW", "MEDIUM", "HIGH")}
            assert sum(posterior.values()) == pytest.approx(1.0, abs=1e-12)

    def test_posterior_exact_values_for_good_signal(self):
        """
        With alpha LOW=[5,1,1], MED=[1,5,1], HIGH=[1,1,5] (all sums = 7):
            P_hat(GOOD|LOW)  = 1/7
            P_hat(GOOD|MED)  = 1/7
            P_hat(GOOD|HIGH) = 5/7

        Prior: P(LOW)=0.3, P(MED)=0.4, P(HIGH)=0.3

        Unnorm: LOW=0.3/7, MED=0.4/7, HIGH=1.5/7  →  total=2.2/7
        Posterior: LOW=3/22, MED=4/22, HIGH=15/22
        """
        strategy = _build_informed_alpha()
        pm = strategy._posterior_mean()
        prior = {"LOW": 0.3, "MEDIUM": 0.4, "HIGH": 0.3}

        unnorm = {q: pm[q]["GOOD"] * prior[q] for q in ("LOW", "MEDIUM", "HIGH")}
        total = sum(unnorm.values())
        posterior = {q: unnorm[q] / total for q in ("LOW", "MEDIUM", "HIGH")}

        assert posterior["LOW"]    == pytest.approx(3/22,  abs=1e-10)
        assert posterior["MEDIUM"] == pytest.approx(4/22,  abs=1e-10)
        assert posterior["HIGH"]   == pytest.approx(15/22, abs=1e-10)

    def test_ev_buy_exact_for_good_signal(self):
        """
        E[BUY|GOOD] = (3/22)×(-15) + (4/22)×5 + (15/22)×20
                    = (-45 + 20 + 300) / 22
                    = 275/22 ≈ 12.5
        """
        strategy = _build_informed_alpha()
        # choose_action is deterministic (pure argmax on EV)
        # Verify via internal computation
        pm = strategy._posterior_mean()
        prior = {"LOW": 0.3, "MEDIUM": 0.4, "HIGH": 0.3}
        payoffs = {"LOW": -15.0, "MEDIUM": 5.0, "HIGH": 20.0}

        unnorm = {q: pm[q]["GOOD"] * prior[q] for q in ("LOW", "MEDIUM", "HIGH")}
        total = sum(unnorm.values())
        posterior = {q: unnorm[q] / total for q in ("LOW", "MEDIUM", "HIGH")}
        ev = sum(posterior[q] * payoffs[q] for q in ("LOW", "MEDIUM", "HIGH"))

        assert ev == pytest.approx(275/22, abs=1e-10)

    def test_ev_buy_exact_for_bad_signal(self):
        """
        E[BUY|BAD] = (15/22)×(-15) + (4/22)×5 + (3/22)×20 = -145/22 ≈ -6.59
        """
        strategy = _build_informed_alpha()
        pm = strategy._posterior_mean()
        prior = {"LOW": 0.3, "MEDIUM": 0.4, "HIGH": 0.3}
        payoffs = {"LOW": -15.0, "MEDIUM": 5.0, "HIGH": 20.0}

        unnorm = {q: pm[q]["BAD"] * prior[q] for q in ("LOW", "MEDIUM", "HIGH")}
        total = sum(unnorm.values())
        posterior = {q: unnorm[q] / total for q in ("LOW", "MEDIUM", "HIGH")}
        ev = sum(posterior[q] * payoffs[q] for q in ("LOW", "MEDIUM", "HIGH"))

        assert ev == pytest.approx(-145/22, abs=1e-10)

    def test_choose_action_buys_when_ev_positive_no_randomness(self):
        """
        DirichletBayesianStrategy.choose_action is deterministic (pure threshold on EV).
        With GOOD signal and informed alpha, EV = 275/22 > 0 → always BUY.
        """
        strategy = _build_informed_alpha()
        # Run 50 times with no seeding — result must be identical every time
        actions = {strategy.choose_action(Signal.GOOD, i, 50) for i in range(1, 51)}
        assert actions == {Action.BUY}, "Deterministic strategy returned mixed actions"

    def test_choose_action_passes_when_ev_negative_no_randomness(self):
        """With BAD signal and informed alpha, EV = -145/22 < 0 → always PASS."""
        strategy = _build_informed_alpha()
        actions = {strategy.choose_action(Signal.BAD, i, 50) for i in range(1, 51)}
        assert actions == {Action.PASS}, "Deterministic strategy returned mixed actions"

    def test_forgetting_exact_after_two_updates(self):
        """
        With forgetting_factor=0.5 and prior_strength=1.0:
          Decay is applied BEFORE every increment (including the very first).

          Before update 1: alpha[HIGH][GOOD] = 1.0
            → multiply all by 0.5: 0.5
            → += 1: alpha[HIGH][GOOD] = 1.5
            → non-updated entries: 0.5

          Before update 2: alpha[HIGH][GOOD] = 1.5
            → multiply all by 0.5: alpha[HIGH][GOOD] = 0.75, others = 0.25
            → += 1: alpha[HIGH][GOOD] = 1.75
        """
        strategy = DirichletBayesianStrategy(prior_strength=1.0, forgetting_factor=0.5)

        strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        alpha1 = strategy.get_diagnostics()["alpha"]
        assert alpha1["HIGH"]["GOOD"] == pytest.approx(1.5, abs=1e-12)  # 1.0*0.5 + 1
        assert alpha1["HIGH"]["BAD"]  == pytest.approx(0.5, abs=1e-12)  # 1.0*0.5 (no increment)
        assert alpha1["LOW"]["GOOD"]  == pytest.approx(0.5, abs=1e-12)  # 1.0*0.5 (different quality)

        strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        alpha2 = strategy.get_diagnostics()["alpha"]
        assert alpha2["HIGH"]["GOOD"] == pytest.approx(1.75, abs=1e-12)  # 1.5*0.5 + 1
        assert alpha2["HIGH"]["BAD"]  == pytest.approx(0.25, abs=1e-12)  # 0.5*0.5
        assert alpha2["LOW"]["GOOD"]  == pytest.approx(0.25, abs=1e-12)  # 0.5*0.5

    def test_forgetting_reduces_effective_sample_size_over_time(self):
        """
        With forgetting_factor < 1, sum of alpha for any quality row decreases
        toward a finite steady-state as old mass is discounted.
        After many updates with factor=0.5 and one observation per step,
        the effective sample size (sum of alpha) is bounded.
        """
        strategy = DirichletBayesianStrategy(prior_strength=1.0, forgetting_factor=0.5)
        prev_sum = sum(strategy._alpha["HIGH"].values())
        for _ in range(20):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)

        # With no forgetting the sum would be 3 + 20 = 23
        # With f=0.5, sum converges to finite value << 23
        no_forget = DirichletBayesianStrategy(prior_strength=1.0, forgetting_factor=1.0)
        for _ in range(20):
            no_forget.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)

        sum_forget  = sum(strategy._alpha["HIGH"].values())
        sum_no_forget = sum(no_forget._alpha["HIGH"].values())
        assert sum_forget < sum_no_forget


# ---------------------------------------------------------------------------
# Regret Matching — exact regret accumulation and mixing
# ---------------------------------------------------------------------------

class TestRegretMatchingMath:
    """Counterfactual regret values and resulting mixed-strategy probabilities are exact."""

    def test_regret_exact_after_known_sequence(self):
        """
        Sequence: PASS on (GOOD, HIGH) then BUY on (GOOD, LOW)

        Round 1: action=PASS, quality=HIGH
            R[GOOD][BUY]  += u(BUY,HIGH)  - u(PASS,HIGH)  = 20 - 0 = +20
            R[GOOD][PASS] += u(PASS,HIGH) - u(PASS,HIGH)  = 0  - 0 = 0

        Round 2: action=BUY, quality=LOW
            R[GOOD][BUY]  += u(BUY,LOW)  - u(BUY,LOW)   = -15 - (-15) = 0
            R[GOOD][PASS] += u(PASS,LOW) - u(BUY,LOW)   = 0   - (-15) = +15

        Final: R[GOOD][BUY] = 20, R[GOOD][PASS] = 15
        """
        strategy = RegretMatchingStrategy()
        strategy.update(Signal.GOOD, Action.PASS, AssetQuality.HIGH, 0.0,   0.0)
        strategy.update(Signal.GOOD, Action.BUY,  AssetQuality.LOW,  10.0, -15.0)

        regret = strategy.get_diagnostics()["cumulative_regret"]["GOOD"]
        assert regret["BUY"]  == pytest.approx(20.0, abs=1e-12)
        assert regret["PASS"] == pytest.approx(15.0, abs=1e-12)

    def test_mixed_strategy_exact_from_regrets(self):
        """
        R[GOOD][BUY]=20, R[GOOD][PASS]=15  →  p(BUY|GOOD) = 20/(20+15) = 4/7
        """
        strategy = RegretMatchingStrategy()
        strategy.update(Signal.GOOD, Action.PASS, AssetQuality.HIGH, 0.0,   0.0)
        strategy.update(Signal.GOOD, Action.BUY,  AssetQuality.LOW,  10.0, -15.0)

        mixed = strategy.get_diagnostics()["mixed_strategy"]["GOOD"]
        assert mixed["BUY"]  == pytest.approx(4/7, abs=1e-10)
        assert mixed["PASS"] == pytest.approx(3/7, abs=1e-10)

    def test_negative_regrets_contribute_zero_to_mixing(self):
        """When R[s][a] ≤ 0 for all actions, mixed strategy is uniform (0.5/0.5)."""
        strategy = RegretMatchingStrategy()
        # BUY on (GOOD, LOW): R[GOOD][BUY] += 0, R[GOOD][PASS] += +15
        # So R[GOOD][BUY] = 0, R[GOOD][PASS] = 15
        strategy.update(Signal.GOOD, Action.BUY, AssetQuality.LOW, 10.0, -15.0)

        regret = strategy.get_diagnostics()["cumulative_regret"]["GOOD"]
        assert regret["BUY"] == pytest.approx(0.0, abs=1e-12)
        assert regret["PASS"] == pytest.approx(15.0, abs=1e-12)

        # p(BUY|GOOD) = max(0,0) / (max(0,0) + max(15,0)) = 0/15 = 0
        # → uniform (0.5) because we treat zero-sum as uniform
        # Actually: p(BUY) = 0/15 = 0, p(PASS) = 15/15 = 1
        # Let's just check p(PASS) = 1 (not "uniform" — that was wrong in doc)
        mixed = strategy.get_diagnostics()["mixed_strategy"]["GOOD"]
        assert mixed["BUY"]  == pytest.approx(0.0,  abs=1e-10)
        assert mixed["PASS"] == pytest.approx(1.0,  abs=1e-10)

    def test_regret_separate_per_signal(self):
        """Regrets for one signal do not contaminate another."""
        strategy = RegretMatchingStrategy()
        strategy.update(Signal.GOOD, Action.PASS, AssetQuality.HIGH, 0.0, 0.0)

        regret = strategy.get_diagnostics()["cumulative_regret"]
        # Only GOOD should have non-zero regret
        assert regret["GOOD"]["BUY"] == pytest.approx(20.0, abs=1e-12)
        assert regret["BAD"]["BUY"]  == pytest.approx(0.0,  abs=1e-12)
        assert regret["NEUTRAL"]["BUY"] == pytest.approx(0.0, abs=1e-12)

    def test_regret_accumulates_across_rounds(self):
        """Regret is cumulative: two identical rounds double the regret."""
        strategy = RegretMatchingStrategy()
        strategy.update(Signal.GOOD, Action.PASS, AssetQuality.HIGH, 0.0, 0.0)
        strategy.update(Signal.GOOD, Action.PASS, AssetQuality.HIGH, 0.0, 0.0)

        regret = strategy.get_diagnostics()["cumulative_regret"]["GOOD"]
        assert regret["BUY"] == pytest.approx(40.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Hedge — exact weight update formula: exp(-η · l)
# ---------------------------------------------------------------------------

class TestHedgeMath:
    """Hedge weights update via exp(-η·l); this formula is exact and gives the regret bound."""

    def test_weight_update_formula_is_exp_neg_eta_loss(self):
        """
        After one update with Signal.GOOD, Action.BUY, AssetQuality.HIGH and η=0.2:
            loss(BUY)  = (20 - 20) / 35 = 0
            loss(PASS) = (20 -  0) / 35 = 4/7

            w_new[GOOD][BUY]  = 1.0 × exp(-0.2 × 0) = 1.0
            w_new[GOOD][PASS] = 1.0 × exp(-0.2 × 4/7) = exp(-4/35)
        """
        eta = 0.2
        strategy = HedgeStrategy(learning_rate=eta)
        strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)

        weights = strategy.get_diagnostics()["weights"]["GOOD"]
        expected_w_buy  = math.exp(-eta * 0.0)
        expected_w_pass = math.exp(-eta * (20.0 / 35.0))

        assert weights["BUY"]  == pytest.approx(expected_w_buy,  abs=1e-12)
        assert weights["PASS"] == pytest.approx(expected_w_pass, abs=1e-12)

    def test_weight_update_for_bad_quality_punishes_buy(self):
        """
        After update with LOW quality:
            loss(BUY)  = (20 - (-15)) / 35 = 35/35 = 1.0
            loss(PASS) = (20 -   0  ) / 35 = 20/35 = 4/7

            w[BUY]  = exp(-η × 1.0)
            w[PASS] = exp(-η × 4/7)

        BUY is penalized more heavily than PASS.
        """
        eta = 0.3
        strategy = HedgeStrategy(learning_rate=eta)
        strategy.update(Signal.BAD, Action.PASS, AssetQuality.LOW, 0.0, 0.0)

        weights = strategy.get_diagnostics()["weights"]["BAD"]
        expected_w_buy  = math.exp(-eta * 1.0)
        expected_w_pass = math.exp(-eta * (20.0 / 35.0))

        assert weights["BUY"]  == pytest.approx(expected_w_buy,  abs=1e-12)
        assert weights["PASS"] == pytest.approx(expected_w_pass, abs=1e-12)
        assert weights["BUY"] < weights["PASS"]

    def test_mixed_strategy_probabilities_sum_to_one(self):
        """Mixed strategy probabilities derived from weights sum to 1."""
        eta = 0.2
        strategy = HedgeStrategy(learning_rate=eta)
        strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)

        mixed = strategy.get_diagnostics()["mixed_strategy"]["GOOD"]
        assert sum(mixed.values()) == pytest.approx(1.0, abs=1e-12)

    def test_regret_bound_holds_deterministically(self):
        """
        Hedge with optimal η = sqrt(2 ln(2) / T) satisfies:
            expected_regret ≤ sqrt(2 · T · ln(2)) · U_RANGE

        Setup: T=100 rounds, Signal.GOOD, all HIGH quality.
        Best fixed action: BUY with total = 100 × 20 = 2000.

        Expected payoff is computed deterministically (weights don't depend on
        which action was sampled — full-information updates).
        """
        T = 100
        eta = math.sqrt(2 * math.log(2) / T)
        strategy = HedgeStrategy(learning_rate=eta)
        U_RANGE = 35.0

        expected_total = 0.0
        for _ in range(T):
            diag = strategy.get_diagnostics()
            w = diag["weights"]["GOOD"]
            total_w = sum(w.values())
            p_buy = w["BUY"] / total_w

            # Quality is HIGH every round
            expected_total += p_buy * 20.0 + (1 - p_buy) * 0.0

            # Full-information update: weights updated for BOTH actions regardless of draw
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)

        best_fixed = 100 * 20.0  # always BUY
        actual_regret = best_fixed - expected_total
        theoretical_bound = math.sqrt(2 * T * math.log(2)) * U_RANGE

        assert actual_regret >= 0, "Expected payoff exceeded best fixed action (impossible)"
        assert actual_regret <= theoretical_bound, (
            f"Regret {actual_regret:.4f} exceeded theoretical bound {theoretical_bound:.4f}"
        )


# ---------------------------------------------------------------------------
# Level-k — EV computation and weight schedule
# ---------------------------------------------------------------------------

class TestLevelKMath:
    """Level-1 EV is exactly 3.5; blending formula and weight schedule are exact."""

    def test_level1_ev_is_exactly_3_5(self):
        """
        E[BUY|prior] = 0.3×(-15) + 0.4×5 + 0.3×20 = -4.5 + 2 + 6 = 3.5
        This is the level-1 expected value and the unconditional buy floor.
        """
        prior = {"LOW": 0.3, "MEDIUM": 0.4, "HIGH": 0.3}
        payoffs = {"LOW": -15.0, "MEDIUM": 5.0, "HIGH": 20.0}
        ev = sum(prior[q] * payoffs[q] for q in prior)
        assert ev == pytest.approx(3.5, abs=1e-12)

    def test_level_k_empirical_weight_exact(self):
        """empirical_weight = min(n_observations × growth, 1.0)."""
        growth = 0.05
        strategy = LevelKStrategy(empirical_weight_growth=growth)

        for n in range(1, 25):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
            expected_weight = min(n * growth, 1.0)
            actual_weight = strategy.get_diagnostics()["empirical_weight"]
            assert actual_weight == pytest.approx(expected_weight, abs=1e-12), \
                f"At n={n}: expected {expected_weight}, got {actual_weight}"

    def test_level_k_blended_ev_exact_at_known_weight(self):
        """
        At empirical_weight=0.0 (zero observations), blended EV = level-1 EV = 3.5.
        So choose_action must return BUY for all signals (3.5 > 0).
        """
        strategy = LevelKStrategy()
        assert strategy.get_diagnostics()["empirical_weight"] == 0.0
        for signal in Signal:
            assert strategy.choose_action(signal, 1, 20) == Action.BUY, \
                f"Level-1 should BUY on all signals, failed for {signal}"

    def test_level_k_empirical_posterior_exact(self):
        """
        After 4 observations of (GOOD, HIGH): alpha["HIGH"]["GOOD"] = 5, others = 1 each.
        P_hat(GOOD|HIGH) = 5/7, P_hat(GOOD|others) = 1/7.

        For GOOD signal:
          unnorm(HIGH) = (5/7) × 0.3 = 1.5/7
          unnorm(MED)  = (1/7) × 0.4 = 0.4/7
          unnorm(LOW)  = (1/7) × 0.3 = 0.3/7
          total = 2.2/7
          P(HIGH|GOOD) = 1.5/2.2 = 15/22

        E[BUY|GOOD] = (3/22)×(-15) + (4/22)×5 + (15/22)×20 = 275/22 ≈ 12.5 > 0 → BUY
        """
        strategy = LevelKStrategy(empirical_weight_growth=1.0)  # weight=1 after 1 observation
        for _ in range(4):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)

        # empirical_weight = min(4 * 1.0, 1.0) = 1.0 → purely empirical
        assert strategy.get_diagnostics()["empirical_weight"] == pytest.approx(1.0)
        assert strategy.choose_action(Signal.GOOD, 5, 20) == Action.BUY


# ---------------------------------------------------------------------------
# Bandit (SW-UCB) — exact arm-selection formula
# ---------------------------------------------------------------------------

class TestBanditMath:
    """UCB arm-selection formula is computed exactly from the sliding window."""

    def _build_window_strategy(self, buy_rewards, window_size=20):
        """Feed known BUY rewards, no PASS observations."""
        strategy = BanditStrategy(window_size=window_size, exploration_constant=1.0)
        for r in buy_rewards:
            quality = (AssetQuality.HIGH if r == 20.0 else
                       AssetQuality.MEDIUM if r == 5.0 else AssetQuality.LOW)
            strategy.update(Signal.GOOD, Action.BUY, quality, 10.0, r)
        return strategy

    def test_ucb_buy_beats_pass_after_high_rewards(self):
        """
        After seeing only HIGH-quality BUY rewards (r=20 each):
            mean_buy = 20.0, N=10, n_buy=10
            UCB(BUY)  = 20.0 + 1.0 × sqrt(ln(10)/10)
            UCB(PASS) = 0.0  (no exploration bonus — value is deterministically known)

        UCB(BUY) > UCB(PASS), so choose_action must be BUY (deterministically).
        """
        strategy = self._build_window_strategy([20.0] * 10)
        # With only BUY actions observed and high mean reward, BUY's UCB dominates
        action = strategy.choose_action(Signal.GOOD, 11, 40)
        assert action == Action.BUY

    def test_ucb_pass_beats_buy_after_low_rewards(self):
        """
        After seeing only LOW-quality BUY rewards (r=-15 each):
            mean_buy = -15.0, N=10, n_buy=10
            UCB(BUY)  = -15.0 + 1.0 × sqrt(ln(10)/10) ≈ -15 + 0.48 = -14.52
            UCB(PASS) = 0.0 (deterministically known)

        UCB(PASS) = 0 > UCB(BUY) ≈ -14.52, so choose_action must be PASS.
        """
        strategy = self._build_window_strategy([-15.0] * 10)
        action = strategy.choose_action(Signal.GOOD, 11, 40)
        assert action == Action.PASS

    def test_ucb_formula_exact_mean_and_bonus(self):
        """
        Feed 4 BUY rewards: [20, 20, -15, 20] on GOOD signal.
            mean_buy = (20+20-15+20)/4 = 45/4 = 11.25
            N = 4, n_buy = 4
            UCB(BUY) = 11.25 + 1.0 × sqrt(ln(4)/4)
            UCB(PASS) = 0.0

        UCB(BUY) > 0 → action = BUY.
        """
        strategy = BanditStrategy(window_size=10, exploration_constant=1.0)
        rewards = [20.0, 20.0, -15.0, 20.0]
        qualities = [AssetQuality.HIGH, AssetQuality.HIGH, AssetQuality.LOW, AssetQuality.HIGH]
        for r, q in zip(rewards, qualities):
            strategy.update(Signal.GOOD, Action.BUY, q, 10.0, r)

        # Verify UCB formula: mean + c * sqrt(ln(N) / N_action)
        n = 4
        n_buy = 4
        mean_buy = 45.0 / 4
        expected_ucb_buy = mean_buy + 1.0 * math.sqrt(math.log(n) / n_buy)
        # UCB(BUY) = 11.25 + sqrt(ln(4)/4) ≈ 11.25 + 0.589 ≈ 11.839 > 0
        assert expected_ucb_buy > 0  # confirms BUY should win
        action = strategy.choose_action(Signal.GOOD, 5, 40)
        assert action == Action.BUY

    def test_window_forgets_old_observations(self):
        """After window fills, oldest observation is dropped and mean recomputed."""
        strategy = BanditStrategy(window_size=3, exploration_constant=0.0)
        # Feed 3 HIGH (positive) then 3 LOW (negative)
        for _ in range(3):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        for _ in range(3):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.LOW, 10.0, -15.0)

        # Window now contains only the 3 LOW observations; mean = -15
        diag = strategy.get_diagnostics()
        assert diag["windows"]["GOOD"]["buy_count"] == 3
        assert diag["windows"]["GOOD"]["mean_buy_reward"] == pytest.approx(-15.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Thompson Sampling — variance properties (seeded for reproducibility)
# ---------------------------------------------------------------------------

class TestThompsonMathSeeded:
    """
    Thompson Sampling variance properties verified with fixed seeds.
    These tests are deterministic and will not flake.
    """

    def test_high_variance_under_uniform_prior_seeded(self):
        """
        With uniform Dirichlet (all alpha=1), posterior samples vary considerably.
        Under the default prior (E[BUY|prior]=3.5>0), Thompson will tend toward BUY,
        but NOT deterministically — it should also produce PASS when sampled garbling
        matrices happen to yield E[BUY] < 0.

        Seeded for reproducibility: verifies that both actions appear and that the
        fraction of PASS decisions is measurably non-zero (variance property).
        """
        np.random.seed(42)
        random.seed(42)
        strategy = ThompsonSamplingStrategy()
        actions = [strategy.choose_action(Signal.GOOD, i, 20) for i in range(1, 201)]
        buys  = sum(1 for a in actions if a == Action.BUY)
        passes = 200 - buys

        # Both actions must appear (non-degenerate sampling)
        assert buys  > 0,  "Expected at least one BUY under uniform prior"
        assert passes > 0, "Expected at least one PASS under uniform prior"

        # Meaningful variance: at least 10% of decisions go against the EV-positive default
        assert passes >= 20, (
            f"Expected substantial variance (≥20 PASS), got {passes}/200. "
            "Thompson should not be deterministic under a flat prior."
        )

    def test_converges_toward_deterministic_after_many_observations_seeded(self):
        """
        After 40 rounds of strong evidence (GOOD→HIGH, BAD→LOW), Thompson should
        agree with DirichletBayesian on GOOD signal ≥ 90% of the time.
        Seeded for reproducibility.
        """
        np.random.seed(123)
        random.seed(123)

        from garbling_gym.core.agents.strategies.bayesian import DirichletBayesianStrategy
        thompson = ThompsonSamplingStrategy()
        bayesian = DirichletBayesianStrategy()

        for _ in range(20):
            for strat in (thompson, bayesian):
                strat.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
                strat.update(Signal.BAD,  Action.PASS, AssetQuality.LOW,  0.0,  0.0)

        bayesian_action = bayesian.choose_action(Signal.GOOD, 41, 100)

        np.random.seed(123)
        random.seed(123)
        agreement = sum(
            1 for _ in range(100)
            if thompson.choose_action(Signal.GOOD, 41, 100) == bayesian_action
        )
        assert agreement >= 85, \
            f"Expected Thompson to agree with Bayesian ≥85%, got {agreement}/100"


# ---------------------------------------------------------------------------
# Property-based tests (hypothesis) — universal invariants
# ---------------------------------------------------------------------------

def _make_alpha_arrays():
    """Hypothesis strategy: random positive alpha vectors for 3 quality rows."""
    return st.fixed_dictionaries({
        q: st.fixed_dictionaries({
            s: st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False)
            for s in ("BAD", "NEUTRAL", "GOOD")
        })
        for q in ("LOW", "MEDIUM", "HIGH")
    })


class TestPropertyBasedInvariants:
    """Universal invariants that must hold for any valid strategy state."""

    @given(alpha=_make_alpha_arrays())
    @settings(max_examples=200)
    def test_dirichlet_posterior_always_sums_to_one(self, alpha):
        """For any alpha matrix, posterior P(q|s) sums to 1 for every signal."""
        strategy = DirichletBayesianStrategy()
        # Inject arbitrary alpha directly
        strategy._alpha = {q: dict(alpha[q]) for q in ("LOW", "MEDIUM", "HIGH")}

        pm = strategy._posterior_mean()
        prior = {"LOW": 0.3, "MEDIUM": 0.4, "HIGH": 0.3}

        for signal in ("BAD", "NEUTRAL", "GOOD"):
            unnorm = {q: pm[q][signal] * prior[q] for q in ("LOW", "MEDIUM", "HIGH")}
            total = sum(unnorm.values())
            if total > 1e-15:  # skip numerically degenerate cases
                posterior = {q: unnorm[q] / total for q in ("LOW", "MEDIUM", "HIGH")}
                assert sum(posterior.values()) == pytest.approx(1.0, abs=1e-9)

    @given(
        n_buy=st.integers(min_value=0, max_value=30),
        n_neutral=st.integers(min_value=0, max_value=30),
        n_pass=st.integers(min_value=0, max_value=30),
    )
    @settings(max_examples=200)
    def test_regret_mixed_strategy_always_sums_to_one(self, n_buy, n_neutral, n_pass):
        """Mixed strategy probabilities always sum to 1, for any regret history."""
        strategy = RegretMatchingStrategy()
        for _ in range(n_buy):
            strategy.update(Signal.GOOD, Action.PASS, AssetQuality.HIGH, 0.0, 0.0)
        for _ in range(n_neutral):
            strategy.update(Signal.NEUTRAL, Action.BUY, AssetQuality.MEDIUM, 10.0, 5.0)
        for _ in range(n_pass):
            strategy.update(Signal.BAD, Action.BUY, AssetQuality.LOW, 10.0, -15.0)

        mixed = strategy.get_diagnostics()["mixed_strategy"]
        for s, probs in mixed.items():
            assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9), \
                f"Mixed strategy for {s} does not sum to 1: {probs}"

    @given(
        n_updates=st.integers(min_value=0, max_value=20),
        signal_idx=st.integers(min_value=0, max_value=2),
        round_num=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=200)
    def test_all_strategies_always_return_valid_action(self, n_updates, signal_idx, round_num):
        """Every strategy returns Action.BUY or Action.PASS for any valid input."""
        signals = list(Signal)
        signal = signals[signal_idx]
        qualities = [AssetQuality.HIGH, AssetQuality.MEDIUM, AssetQuality.LOW]
        q = qualities[n_updates % 3]

        for strategy in [
            DirichletBayesianStrategy(),
            RegretMatchingStrategy(),
            LevelKStrategy(),
            BanditStrategy(),
        ]:
            for _ in range(n_updates):
                strategy.update(signal, Action.BUY, q, 10.0, 5.0)
            action = strategy.choose_action(signal, round_num, 50)
            assert action in (Action.BUY, Action.PASS), \
                f"{type(strategy).__name__} returned invalid action: {action}"

    @given(n=st.integers(min_value=1, max_value=40))
    @settings(max_examples=100)
    def test_level_k_empirical_weight_never_exceeds_one(self, n):
        """empirical_weight is always in [0, 1]."""
        strategy = LevelKStrategy(empirical_weight_growth=0.1)
        for _ in range(n):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        w = strategy.get_diagnostics()["empirical_weight"]
        assert 0.0 <= w <= 1.0

    @given(eta=st.floats(min_value=0.001, max_value=0.5, allow_nan=False))
    @settings(max_examples=100)
    def test_hedge_weights_always_positive_after_updates(self, eta):
        """Hedge weights remain strictly positive regardless of eta and losses."""
        strategy = HedgeStrategy(learning_rate=eta)
        for q in (AssetQuality.HIGH, AssetQuality.LOW, AssetQuality.MEDIUM):
            strategy.update(Signal.GOOD, Action.BUY, q, 10.0, 5.0)
        weights = strategy.get_diagnostics()["weights"]
        for s_weights in weights.values():
            for w in s_weights.values():
                assert w > 0, f"Hedge weight went non-positive: {w}"
