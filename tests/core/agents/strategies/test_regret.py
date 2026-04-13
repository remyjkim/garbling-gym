# ABOUTME: Tests for RegretMatchingStrategy and HedgeStrategy
# ABOUTME: Verifies counterfactual regret accumulation, uniform start, convergence

import random

import pytest
import numpy as np
from garbling_gym.core.agents.strategies import ReceiverStrategy
from garbling_gym.core.agents.strategies.regret import RegretMatchingStrategy, HedgeStrategy
from garbling_gym.core.agents.strategies.registry import receiver_strategy_registry
from garbling_gym.core.types import Action, AssetQuality, Signal


class TestRegretMatchingStrategy:
    """RegretMatchingStrategy is uniform at start and learns counterfactual regrets."""

    def test_is_receiver_strategy(self):
        strategy = RegretMatchingStrategy()
        assert isinstance(strategy, ReceiverStrategy)

    def test_uniform_at_start(self):
        """With no history all regrets are 0, strategy is uniform (50/50)."""
        random.seed(42)
        strategy = RegretMatchingStrategy()
        n = 1000
        buys = sum(1 for _ in range(n) if strategy.choose_action(Signal.GOOD, 1, 20) == Action.BUY)
        # Should be roughly 50/50 ± 5%
        assert 400 < buys < 600, f"Expected ~500 BUY, got {buys}"

    def test_learns_to_pass_on_bad_signal_after_bad_outcomes(self):
        """After many BAD-signal+LOW-quality rounds, regret for PASS on BAD dominates → PASS."""
        random.seed(42)
        strategy = RegretMatchingStrategy()
        # BAD signal followed by LOW quality: buying loses -15, passing gives 0
        for _ in range(40):
            strategy.update(Signal.BAD, Action.BUY, AssetQuality.LOW, 10.0, -15.0)

        passes = sum(
            1 for _ in range(200) if strategy.choose_action(Signal.BAD, 41, 60) == Action.PASS
        )
        assert passes > 150, f"Expected mostly PASS on BAD signal, got {passes}/200"

    def test_counterfactual_regret_tracks_untaken_action(self):
        """Regret accumulates for actions not taken (counterfactual updates)."""
        strategy = RegretMatchingStrategy()
        # Take PASS on a GOOD+HIGH round — regret should grow for BUY
        strategy.update(Signal.GOOD, Action.PASS, AssetQuality.HIGH, 0.0, 0.0)

        diag = strategy.get_diagnostics()
        regret_buy_on_good = diag["cumulative_regret"]["GOOD"]["BUY"]
        # BUY on HIGH would have given +20, PASS gave 0, so regret for BUY = 20
        assert regret_buy_on_good == pytest.approx(20.0)

    def test_reset_clears_regrets(self):
        """reset() zeros all cumulative regrets."""
        strategy = RegretMatchingStrategy()
        strategy.update(Signal.GOOD, Action.PASS, AssetQuality.HIGH, 0.0, 0.0)
        strategy.reset()
        diag = strategy.get_diagnostics()
        for s_regrets in diag["cumulative_regret"].values():
            for v in s_regrets.values():
                assert v == pytest.approx(0.0)

    def test_registered_as_regret_matching(self):
        strategy = receiver_strategy_registry.get("regret-matching")
        assert isinstance(strategy, RegretMatchingStrategy)

    def test_choose_action_returns_action_enum(self):
        strategy = RegretMatchingStrategy()
        for signal in Signal:
            result = strategy.choose_action(signal, 1, 20)
            assert isinstance(result, Action)


class TestHedgeStrategy:
    """HedgeStrategy uses multiplicative weights with O(sqrt(T ln K)) regret bound."""

    def test_is_receiver_strategy(self):
        strategy = HedgeStrategy()
        assert isinstance(strategy, ReceiverStrategy)

    def test_uniform_at_start(self):
        """With equal weights, strategy is roughly 50/50."""
        random.seed(42)
        strategy = HedgeStrategy()
        n = 1000
        buys = sum(1 for _ in range(n) if strategy.choose_action(Signal.GOOD, 1, 40) == Action.BUY)
        assert 400 < buys < 600, f"Expected ~500 BUY, got {buys}"

    def test_adapts_to_signal_quality_correlation(self):
        """After learning GOOD→HIGH correlation, Hedge favors BUY on GOOD signal."""
        random.seed(42)
        strategy = HedgeStrategy(learning_rate=0.1)
        for _ in range(30):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
            strategy.update(Signal.BAD, Action.PASS, AssetQuality.LOW, 0.0, 0.0)

        buys = sum(
            1 for _ in range(200) if strategy.choose_action(Signal.GOOD, 61, 100) == Action.BUY
        )
        assert buys > 120, f"Expected Hedge to prefer BUY on GOOD, got {buys}/200"

    def test_reset_clears_weights(self):
        """reset() restores all weights to 1."""
        strategy = HedgeStrategy()
        strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        strategy.reset()
        diag = strategy.get_diagnostics()
        for weights in diag["weights"].values():
            for w in weights.values():
                assert w == pytest.approx(1.0)

    def test_registered_as_hedge(self):
        strategy = receiver_strategy_registry.get("hedge")
        assert isinstance(strategy, HedgeStrategy)

    def test_choose_action_returns_action_enum(self):
        strategy = HedgeStrategy()
        for signal in Signal:
            result = strategy.choose_action(signal, 1, 40)
            assert isinstance(result, Action)
