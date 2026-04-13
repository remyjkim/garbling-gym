# ABOUTME: Tests for DirichletBayesianStrategy
# ABOUTME: Verifies Dirichlet conjugate updates, posterior inference, and forgetting factor

import pytest
import numpy as np
from garbling_gym.core.agents.strategies import ReceiverStrategy
from garbling_gym.core.agents.strategies.bayesian import DirichletBayesianStrategy
from garbling_gym.core.agents.strategies.registry import receiver_strategy_registry
from garbling_gym.core.types import Action, AssetQuality, Signal


class TestDirichletBayesianStrategy:
    """DirichletBayesianStrategy does proper Bayesian inference over the garbling matrix."""

    def test_is_receiver_strategy(self):
        """DirichletBayesianStrategy is a ReceiverStrategy subclass."""
        strategy = DirichletBayesianStrategy()
        assert isinstance(strategy, ReceiverStrategy)

    def test_initial_alpha_is_uniform(self):
        """Before any updates, alpha[q][s] == prior_strength for all (q, s)."""
        strength = 2.0
        strategy = DirichletBayesianStrategy(prior_strength=strength)
        alpha = strategy.get_diagnostics()["alpha"]
        # 3 qualities × 3 signals
        assert len(alpha) == 3
        for row in alpha.values():
            for val in row.values():
                assert val == pytest.approx(strength)

    def test_update_increments_correct_alpha(self):
        """update() increments alpha[quality][signal] by 1."""
        strategy = DirichletBayesianStrategy(prior_strength=1.0)
        strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)

        alpha = strategy.get_diagnostics()["alpha"]
        # Only HIGH/GOOD should have increased
        assert alpha["HIGH"]["GOOD"] == pytest.approx(2.0)
        # Others unchanged
        assert alpha["HIGH"]["BAD"] == pytest.approx(1.0)
        assert alpha["LOW"]["GOOD"] == pytest.approx(1.0)

    def test_posterior_mean_estimates_garbling_probabilities(self):
        """Posterior mean P_hat(s|q) = alpha[q][s] / sum(alpha[q])."""
        strategy = DirichletBayesianStrategy(prior_strength=1.0)
        # Feed 9 observations: all HIGH → GOOD
        for _ in range(9):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)

        # alpha[HIGH] = [1, 1, 10] (BAD, NEUTRAL, GOOD)
        # sum = 12, P_hat(GOOD|HIGH) = 10/12
        diag = strategy.get_diagnostics()
        p_good_given_high = diag["posterior_mean"]["HIGH"]["GOOD"]
        assert p_good_given_high == pytest.approx(10.0 / 12.0, abs=1e-6)

    def test_buys_on_reliable_good_signal(self):
        """After learning that GOOD reliably means HIGH quality, GOOD signal → BUY."""
        strategy = DirichletBayesianStrategy(prior_strength=1.0)
        # Feed many HIGH→GOOD observations
        for _ in range(20):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        for _ in range(20):
            strategy.update(Signal.BAD, Action.PASS, AssetQuality.LOW, 0.0, 0.0)

        action = strategy.choose_action(Signal.GOOD, round_num=21, total_rounds=40)
        assert action == Action.BUY

    def test_passes_on_unreliable_good_signal(self):
        """After learning that GOOD actually predicts LOW quality, GOOD signal → PASS."""
        strategy = DirichletBayesianStrategy(prior_strength=1.0)
        # Feed many LOW→GOOD observations (sender always garbles LOW to GOOD)
        for _ in range(30):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.LOW, 10.0, -15.0)

        action = strategy.choose_action(Signal.GOOD, round_num=31, total_rounds=50)
        assert action == Action.PASS

    def test_forgetting_factor_decays_old_observations(self):
        """forgetting_factor < 1.0 reduces weight of old observations before each update."""
        strategy_forget = DirichletBayesianStrategy(prior_strength=1.0, forgetting_factor=0.5)
        strategy_no_forget = DirichletBayesianStrategy(prior_strength=1.0, forgetting_factor=1.0)

        # Feed same observations to both
        for _ in range(10):
            strategy_forget.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
            strategy_no_forget.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)

        alpha_forget = strategy_forget.get_diagnostics()["alpha"]
        alpha_no_forget = strategy_no_forget.get_diagnostics()["alpha"]

        # Forgetting strategy should have smaller alpha sums (old mass decayed)
        sum_forget = sum(alpha_forget["HIGH"].values())
        sum_no_forget = sum(alpha_no_forget["HIGH"].values())
        assert sum_forget < sum_no_forget

    def test_reset_restores_initial_alpha(self):
        """reset() returns alpha to the uniform prior."""
        strategy = DirichletBayesianStrategy(prior_strength=2.0)
        strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        strategy.reset()

        alpha = strategy.get_diagnostics()["alpha"]
        for row in alpha.values():
            for val in row.values():
                assert val == pytest.approx(2.0)

    def test_choose_action_returns_action_enum(self):
        """choose_action always returns a valid Action."""
        strategy = DirichletBayesianStrategy()
        for signal in Signal:
            result = strategy.choose_action(signal, round_num=1, total_rounds=20)
            assert isinstance(result, Action)

    def test_registered_as_bayesian_in_registry(self):
        """DirichletBayesianStrategy is accessible as 'bayesian' in the global registry."""
        strategy = receiver_strategy_registry.get("bayesian")
        assert isinstance(strategy, DirichletBayesianStrategy)

    def test_diagnostics_include_required_keys(self):
        """get_diagnostics() includes alpha, posterior_mean, and effective_sample_size."""
        strategy = DirichletBayesianStrategy()
        diag = strategy.get_diagnostics()
        assert "alpha" in diag
        assert "posterior_mean" in diag
        assert "effective_sample_size" in diag
