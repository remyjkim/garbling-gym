# ABOUTME: Tests for ReceiverStrategyRegistry
# ABOUTME: Verifies strategy registration, retrieval by name, and default strategy

import pytest
from garbling_gym.core.agents.strategies import ReceiverStrategy
from garbling_gym.core.agents.strategies.legacy_heuristic import LegacyHeuristicStrategy
from garbling_gym.core.agents.strategies.registry import (
    ReceiverStrategyRegistry,
    receiver_strategy_registry,
)


class TestReceiverStrategyRegistry:
    """ReceiverStrategyRegistry maps strategy names to fresh instances."""

    def test_heuristic_is_registered_by_default(self):
        """'heuristic' is registered as the default strategy."""
        registry = ReceiverStrategyRegistry()
        assert registry.is_registered("heuristic")

    def test_get_returns_legacy_heuristic_for_heuristic_key(self):
        """get('heuristic') returns a LegacyHeuristicStrategy instance."""
        registry = ReceiverStrategyRegistry()
        strategy = registry.get("heuristic")
        assert isinstance(strategy, LegacyHeuristicStrategy)

    def test_get_returns_new_instance_each_call(self):
        """Each call to get() returns a distinct instance (fresh state)."""
        registry = ReceiverStrategyRegistry()
        s1 = registry.get("heuristic")
        s2 = registry.get("heuristic")
        assert s1 is not s2

    def test_get_raises_key_error_for_unknown_strategy(self):
        """get() raises KeyError for unregistered strategy names."""
        registry = ReceiverStrategyRegistry()
        with pytest.raises(KeyError, match="nonexistent"):
            registry.get("nonexistent")

    def test_register_adds_new_strategy(self):
        """register() makes a new strategy available by name."""
        from garbling_gym.core.types import Action

        class AlwaysBuy(ReceiverStrategy):
            def choose_action(self, signal, round_num, total_rounds):
                return Action.BUY
            def update(self, signal, action, true_quality, sender_payoff, receiver_payoff):
                pass
            def reset(self):
                pass

        registry = ReceiverStrategyRegistry()
        registry.register("always_buy", AlwaysBuy)
        strategy = registry.get("always_buy")
        assert isinstance(strategy, AlwaysBuy)

    def test_register_raises_on_duplicate_name(self):
        """register() raises ValueError when name is already taken."""
        registry = ReceiverStrategyRegistry()
        with pytest.raises(ValueError, match="heuristic"):
            registry.register("heuristic", LegacyHeuristicStrategy)

    def test_list_strategies_includes_heuristic(self):
        """list_strategies() returns a dict containing 'heuristic'."""
        registry = ReceiverStrategyRegistry()
        names = registry.list_strategies()
        assert "heuristic" in names

    def test_global_registry_has_heuristic(self):
        """The global receiver_strategy_registry instance has 'heuristic' registered."""
        assert receiver_strategy_registry.is_registered("heuristic")
