# ABOUTME: Tests for LevelKStrategy
# ABOUTME: Verifies level-1 BUY-always start, empirical adaptation, and payoff exploitation

import pytest
from garbling_gym.core.agents.strategies import ReceiverStrategy
from garbling_gym.core.agents.strategies.game_theoretic import LevelKStrategy
from garbling_gym.core.agents.strategies.registry import receiver_strategy_registry
from garbling_gym.core.types import Action, AssetQuality, Signal


class TestLevelKStrategy:
    """LevelKStrategy starts from game-theoretic priors and adapts to observed sender behavior."""

    def test_is_receiver_strategy(self):
        strategy = LevelKStrategy()
        assert isinstance(strategy, ReceiverStrategy)

    def test_buys_unconditionally_at_start(self):
        """Level-1 reasoning: E[BUY|prior] = 3.5 > 0, so always BUY before any observations."""
        strategy = LevelKStrategy()
        for signal in Signal:
            action = strategy.choose_action(signal, round_num=1, total_rounds=20)
            assert action == Action.BUY, f"Expected BUY at level-1 for signal {signal}"

    def test_adapts_to_deceptive_sender(self):
        """After observing GOOD signals always meaning LOW quality, should switch to PASS on GOOD."""
        strategy = LevelKStrategy()
        # Feed many observations: GOOD signal always precedes LOW quality (deceptive sender)
        for i in range(30):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.LOW, 10.0, -15.0)

        # With enough evidence of deceptive sender, should stop buying on GOOD
        passes = sum(
            1 for _ in range(50) if strategy.choose_action(Signal.GOOD, 31, 50) == Action.PASS
        )
        assert passes >= 40, f"Expected level-k to mostly PASS on deceptive GOOD, got {passes}/50"

    def test_reset_returns_to_level1(self):
        """After reset(), returns to level-1 BUY-always behavior."""
        strategy = LevelKStrategy()
        for _ in range(20):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.LOW, 10.0, -15.0)
        strategy.reset()

        for signal in Signal:
            action = strategy.choose_action(signal, 1, 20)
            assert action == Action.BUY

    def test_choose_action_returns_action_enum(self):
        strategy = LevelKStrategy()
        for signal in Signal:
            result = strategy.choose_action(signal, 1, 20)
            assert isinstance(result, Action)

    def test_diagnostics_include_required_keys(self):
        """get_diagnostics() includes effective_level and empirical_weight keys."""
        strategy = LevelKStrategy()
        diag = strategy.get_diagnostics()
        assert "effective_level" in diag
        assert "empirical_weight" in diag

    def test_registered_as_level_k(self):
        strategy = receiver_strategy_registry.get("level-k")
        assert isinstance(strategy, LevelKStrategy)
