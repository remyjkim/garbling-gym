# ABOUTME: Tests for ThompsonSamplingStrategy
# ABOUTME: Verifies posterior sampling behavior, variance under uncertainty, convergence

import pytest
from garbling_gym.core.agents.strategies import ReceiverStrategy
from garbling_gym.core.agents.strategies.bayesian import (
    DirichletBayesianStrategy,
    ThompsonSamplingStrategy,
)
from garbling_gym.core.agents.strategies.registry import receiver_strategy_registry
from garbling_gym.core.types import Action, AssetQuality, Signal


class TestThompsonSamplingStrategy:
    """ThompsonSamplingStrategy samples from the Dirichlet posterior rather than using the mean."""

    def test_is_receiver_strategy(self):
        """ThompsonSamplingStrategy is a ReceiverStrategy subclass."""
        strategy = ThompsonSamplingStrategy()
        assert isinstance(strategy, ReceiverStrategy)

    def test_choose_action_returns_action_enum(self):
        """choose_action always returns a valid Action."""
        strategy = ThompsonSamplingStrategy()
        for signal in Signal:
            result = strategy.choose_action(signal, round_num=1, total_rounds=20)
            assert isinstance(result, Action)

    def test_high_variance_under_uncertainty(self):
        """With few observations, Thompson should produce a mix of BUY and PASS."""
        strategy = ThompsonSamplingStrategy()
        # Only 1 observation — posterior is very diffuse
        strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)

        actions = [strategy.choose_action(Signal.GOOD, i, 20) for i in range(1, 201)]
        buys = sum(1 for a in actions if a == Action.BUY)
        passes = 200 - buys
        # Under high uncertainty should see both outcomes (not all one way)
        assert buys > 10, f"Expected some BUY actions, got {buys}/200"
        assert passes > 10, f"Expected some PASS actions, got {passes}/200"

    def test_converges_toward_bayesian_with_many_observations(self):
        """After many consistent observations, Thompson and Dirichlet should mostly agree."""
        thompson = ThompsonSamplingStrategy()
        bayesian = DirichletBayesianStrategy()

        # Feed same 30 deterministic observations to both
        for _ in range(30):
            for strat in (thompson, bayesian):
                strat.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
                strat.update(Signal.BAD, Action.PASS, AssetQuality.LOW, 0.0, 0.0)

        # With lots of data, Thompson should mostly agree with Bayesian on GOOD signal
        bayesian_action = bayesian.choose_action(Signal.GOOD, 61, 100)
        thompson_actions = [thompson.choose_action(Signal.GOOD, i, 100) for i in range(61, 161)]
        dominant = Action.BUY if bayesian_action == Action.BUY else Action.PASS
        dominant_count = sum(1 for a in thompson_actions if a == dominant)
        assert dominant_count >= 80, f"Expected Thompson to agree with Bayesian ≥80%, got {dominant_count}/100"

    def test_reset_clears_state(self):
        """reset() restores Thompson to the initial uniform Dirichlet state."""
        strategy = ThompsonSamplingStrategy()
        for _ in range(20):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        strategy.reset()
        diag = strategy.get_diagnostics()
        alpha = diag["alpha"]
        for row in alpha.values():
            for val in row.values():
                assert val == pytest.approx(1.0)

    def test_registered_as_thompson_in_registry(self):
        """ThompsonSamplingStrategy is accessible as 'thompson' in the global registry."""
        strategy = receiver_strategy_registry.get("thompson")
        assert isinstance(strategy, ThompsonSamplingStrategy)
