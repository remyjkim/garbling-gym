# ABOUTME: Tests for the ReceiverStrategy abstract base class
# ABOUTME: Verifies interface contract enforcement and basic protocol

import pytest
from garbling_gym.core.agents.strategies import ReceiverStrategy
from garbling_gym.core.types import Action, AssetQuality, Signal


class TestReceiverStrategyInterface:
    """ReceiverStrategy enforces the abstract interface contract."""

    def test_cannot_instantiate_abstract_class(self):
        """ReceiverStrategy is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ReceiverStrategy()

    def test_concrete_missing_choose_action_is_rejected(self):
        """A concrete subclass missing choose_action cannot be instantiated."""
        class Incomplete(ReceiverStrategy):
            def update(self, signal, action, true_quality, sender_payoff, receiver_payoff):
                pass
            def reset(self):
                pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_missing_update_is_rejected(self):
        """A concrete subclass missing update cannot be instantiated."""
        class Incomplete(ReceiverStrategy):
            def choose_action(self, signal, round_num, total_rounds):
                return Action.PASS
            def reset(self):
                pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_missing_reset_is_rejected(self):
        """A concrete subclass missing reset cannot be instantiated."""
        class Incomplete(ReceiverStrategy):
            def choose_action(self, signal, round_num, total_rounds):
                return Action.PASS
            def update(self, signal, action, true_quality, sender_payoff, receiver_payoff):
                pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_concrete_subclass_can_be_instantiated(self):
        """A concrete subclass implementing all methods can be instantiated."""
        class AlwaysBuy(ReceiverStrategy):
            def choose_action(self, signal, round_num, total_rounds):
                return Action.BUY
            def update(self, signal, action, true_quality, sender_payoff, receiver_payoff):
                pass
            def reset(self):
                pass

        strategy = AlwaysBuy()
        assert isinstance(strategy, ReceiverStrategy)

    def test_choose_action_returns_action_enum(self):
        """choose_action must return an Action enum value."""
        class AlwaysBuy(ReceiverStrategy):
            def choose_action(self, signal, round_num, total_rounds):
                return Action.BUY
            def update(self, signal, action, true_quality, sender_payoff, receiver_payoff):
                pass
            def reset(self):
                pass

        strategy = AlwaysBuy()
        result = strategy.choose_action(Signal.GOOD, round_num=1, total_rounds=20)
        assert result == Action.BUY
        assert isinstance(result, Action)

    def test_get_diagnostics_has_default_implementation(self):
        """get_diagnostics has a default implementation returning an empty dict."""
        class Minimal(ReceiverStrategy):
            def choose_action(self, signal, round_num, total_rounds):
                return Action.PASS
            def update(self, signal, action, true_quality, sender_payoff, receiver_payoff):
                pass
            def reset(self):
                pass

        strategy = Minimal()
        diagnostics = strategy.get_diagnostics()
        assert isinstance(diagnostics, dict)

    def test_reset_clears_state(self):
        """reset() returns the strategy to its initial state."""
        class CountingStrategy(ReceiverStrategy):
            def __init__(self):
                self.count = 0
            def choose_action(self, signal, round_num, total_rounds):
                return Action.BUY
            def update(self, signal, action, true_quality, sender_payoff, receiver_payoff):
                self.count += 1
            def reset(self):
                self.count = 0

        strategy = CountingStrategy()
        strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        strategy.update(Signal.BAD, Action.PASS, AssetQuality.LOW, 0.0, 0.0)
        assert strategy.count == 2

        strategy.reset()
        assert strategy.count == 0
