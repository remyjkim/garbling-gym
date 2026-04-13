# ABOUTME: Tests for LegacyHeuristicStrategy - the existing receiver heuristic as a strategy
# ABOUTME: Verifies Bayesian decision-making, history accumulation, and reset behavior

import pytest
from garbling_gym.core.agents.strategies import ReceiverStrategy
from garbling_gym.core.agents.strategies.legacy_heuristic import LegacyHeuristicStrategy
from garbling_gym.core.types import Action, AssetQuality, Signal


class TestLegacyHeuristicStrategy:
    """LegacyHeuristicStrategy wraps the existing receiver heuristic as a ReceiverStrategy."""

    def test_is_receiver_strategy(self):
        """LegacyHeuristicStrategy is a ReceiverStrategy subclass."""
        strategy = LegacyHeuristicStrategy()
        assert isinstance(strategy, ReceiverStrategy)

    def test_choose_action_returns_action_enum(self):
        """choose_action returns a valid Action enum value."""
        strategy = LegacyHeuristicStrategy()
        result = strategy.choose_action(Signal.GOOD, round_num=1, total_rounds=20)
        assert isinstance(result, Action)
        assert result in (Action.BUY, Action.PASS)

    def test_choose_action_accepts_all_signals(self):
        """choose_action handles all Signal variants."""
        strategy = LegacyHeuristicStrategy()
        for signal in Signal:
            result = strategy.choose_action(signal, round_num=1, total_rounds=20)
            assert isinstance(result, Action)

    def test_update_accumulates_history(self):
        """update() stores round observations for use in future decisions."""
        strategy = LegacyHeuristicStrategy()
        assert strategy.history_length() == 0

        strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        assert strategy.history_length() == 1

        strategy.update(Signal.BAD, Action.PASS, AssetQuality.LOW, 0.0, 0.0)
        assert strategy.history_length() == 2

    def test_reset_clears_history(self):
        """reset() removes all accumulated observations."""
        strategy = LegacyHeuristicStrategy()
        strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        strategy.update(Signal.NEUTRAL, Action.BUY, AssetQuality.MEDIUM, 10.0, 5.0)
        assert strategy.history_length() == 2

        strategy.reset()
        assert strategy.history_length() == 0

    def test_get_diagnostics_includes_history_length(self):
        """get_diagnostics returns a dict with history_length key."""
        strategy = LegacyHeuristicStrategy()
        strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        diagnostics = strategy.get_diagnostics()
        assert isinstance(diagnostics, dict)
        assert "history_length" in diagnostics
        assert diagnostics["history_length"] == 1

    def test_good_signal_with_no_history_tends_to_buy(self):
        """With GOOD signal and uninformative prior, E[BUY] > 0, so BUY is favored."""
        # Run many times to check statistical tendency given stochasticity
        strategy = LegacyHeuristicStrategy()
        buys = sum(
            1 for _ in range(200)
            if strategy.choose_action(Signal.GOOD, 1, 20) == Action.BUY
        )
        # With GOOD signal and default reliability, E[BUY] >> 0, should buy most of the time
        assert buys > 120, f"Expected BUY >> 50%, got {buys}/200"

    def test_bad_signal_with_no_history_tends_to_pass(self):
        """With BAD signal and uninformative prior, E[BUY] < 0, so PASS is favored."""
        strategy = LegacyHeuristicStrategy()
        passes = sum(
            1 for _ in range(200)
            if strategy.choose_action(Signal.BAD, 1, 20) == Action.PASS
        )
        # With BAD signal and default reliability, E[BUY] << 0, should pass most of the time
        assert passes > 120, f"Expected PASS >> 50%, got {passes}/200"
