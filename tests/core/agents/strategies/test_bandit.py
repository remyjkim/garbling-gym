# ABOUTME: Tests for BanditStrategy (SW-UCB and EXP3.S variants)
# ABOUTME: Verifies exploration, learning, and non-stationarity handling

import pytest
from garbling_gym.core.agents.strategies import ReceiverStrategy
from garbling_gym.core.agents.strategies.bandit import BanditStrategy
from garbling_gym.core.agents.strategies.registry import receiver_strategy_registry
from garbling_gym.core.types import Action, AssetQuality, Signal


class TestBanditStrategySWUCB:
    """BanditStrategy (sw-ucb variant) explores early, learns signal→quality mappings."""

    def test_is_receiver_strategy(self):
        strategy = BanditStrategy()
        assert isinstance(strategy, ReceiverStrategy)

    def test_choose_action_returns_action_enum(self):
        strategy = BanditStrategy()
        for signal in Signal:
            result = strategy.choose_action(signal, 1, 20)
            assert isinstance(result, Action)

    def test_explores_early_with_no_data(self):
        """With no observations, exploration bonus is large — should produce both BUY and PASS."""
        strategy = BanditStrategy(window_size=10, exploration_constant=2.0)
        actions = [strategy.choose_action(Signal.GOOD, i, 40) for i in range(1, 201)]
        buys = sum(1 for a in actions if a == Action.BUY)
        passes = 200 - buys
        # Should explore (we allow wide range due to UCB favoring unexplored arm)
        assert buys > 0 and passes > 0, f"Expected exploration, got {buys} BUY / {passes} PASS"

    def test_learns_to_buy_after_consistent_good_high_signal(self):
        """After many GOOD→HIGH rounds, should reliably BUY on GOOD."""
        strategy = BanditStrategy(window_size=15, exploration_constant=0.5)
        for i in range(20):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)

        buys = sum(1 for _ in range(100) if strategy.choose_action(Signal.GOOD, 21, 40) == Action.BUY)
        assert buys >= 80, f"Expected mostly BUY on reliable GOOD signal, got {buys}/100"

    def test_adapts_to_sender_shift_via_window(self):
        """Sliding window forgets old data after sender shifts strategy."""
        strategy = BanditStrategy(window_size=5, exploration_constant=0.3)
        # First 15 rounds: GOOD → HIGH (good to buy)
        for i in range(15):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        # Next 10 rounds: GOOD → LOW (bad to buy)
        for i in range(10):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.LOW, 10.0, -15.0)

        # Window should now primarily reflect the LOW quality observations
        passes = sum(1 for _ in range(100) if strategy.choose_action(Signal.GOOD, 26, 40) == Action.PASS)
        assert passes >= 60, f"Expected PASS after sender shift, got {passes}/100 PASS"

    def test_reset_clears_all_windows(self):
        """reset() empties all per-signal sliding windows."""
        strategy = BanditStrategy()
        for _ in range(5):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        strategy.reset()
        diag = strategy.get_diagnostics()
        for signal_stats in diag["windows"].values():
            assert signal_stats["count"] == 0

    def test_registered_as_bandit(self):
        strategy = receiver_strategy_registry.get("bandit")
        assert isinstance(strategy, BanditStrategy)


class TestBanditStrategyEXP3S:
    """BanditStrategy (exp3s variant) handles adversarial senders."""

    def test_is_receiver_strategy(self):
        strategy = BanditStrategy(variant="exp3s")
        assert isinstance(strategy, ReceiverStrategy)

    def test_choose_action_returns_action_enum(self):
        strategy = BanditStrategy(variant="exp3s")
        for signal in Signal:
            result = strategy.choose_action(signal, 1, 40)
            assert isinstance(result, Action)

    def test_adapts_to_signal_quality_correlation(self):
        """EXP3.S variant should learn to BUY on GOOD after consistent GOOD→HIGH rounds."""
        strategy = BanditStrategy(variant="exp3s", window_size=20)
        for i in range(25):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
            strategy.update(Signal.BAD, Action.PASS, AssetQuality.LOW, 0.0, 0.0)

        buys = sum(1 for _ in range(100) if strategy.choose_action(Signal.GOOD, 51, 100) == Action.BUY)
        assert buys >= 60, f"Expected EXP3.S to prefer BUY on GOOD, got {buys}/100"

    def test_reset_clears_exp3s_state(self):
        """reset() restores EXP3.S weights to uniform."""
        strategy = BanditStrategy(variant="exp3s")
        for _ in range(5):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        strategy.reset()
        diag = strategy.get_diagnostics()
        for s_weights in diag["windows"].values():
            assert s_weights["count"] == 0
