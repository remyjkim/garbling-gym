# ABOUTME: Tests for ReceiverAgent strategy delegation
# ABOUTME: Verifies strategy injection, learn() method, and default strategy behavior

import pytest
from unittest.mock import MagicMock
from garbling_gym.core.agents.receiver import ReceiverAgent
from garbling_gym.core.agents.strategies import ReceiverStrategy
from garbling_gym.core.agents.strategies.legacy_heuristic import LegacyHeuristicStrategy
from garbling_gym.core.types import Action, AssetQuality, Signal


class TestReceiverAgentStrategyDelegation:
    """ReceiverAgent delegates decisions to a pluggable ReceiverStrategy."""

    def test_default_strategy_is_legacy_heuristic(self):
        """ReceiverAgent uses LegacyHeuristicStrategy by default."""
        agent = ReceiverAgent()
        assert isinstance(agent.strategy, LegacyHeuristicStrategy)

    def test_accepts_custom_strategy(self):
        """ReceiverAgent accepts any ReceiverStrategy subclass."""
        class AlwaysBuy(ReceiverStrategy):
            def choose_action(self, signal, round_num, total_rounds):
                return Action.BUY
            def update(self, signal, action, true_quality, sender_payoff, receiver_payoff):
                pass
            def reset(self):
                pass

        strategy = AlwaysBuy()
        agent = ReceiverAgent(strategy=strategy)
        assert agent.strategy is strategy

    def test_make_decision_delegates_to_strategy_when_no_llm(self):
        """make_decision uses strategy.choose_action() when LLM API is unavailable."""
        mock_strategy = MagicMock(spec=ReceiverStrategy)
        mock_strategy.choose_action.return_value = Action.BUY

        agent = ReceiverAgent(strategy=mock_strategy)
        agent.use_api = False

        result = agent.make_decision(Signal.GOOD, [], round_num=1, total_rounds=20)

        mock_strategy.choose_action.assert_called_once_with(Signal.GOOD, 1, 20)
        assert result == Action.BUY

    def test_learn_calls_strategy_update(self):
        """learn() delegates to strategy.update() with correct arguments."""
        mock_strategy = MagicMock(spec=ReceiverStrategy)
        agent = ReceiverAgent(strategy=mock_strategy)

        agent.learn(
            signal=Signal.GOOD,
            action=Action.BUY,
            true_quality=AssetQuality.HIGH,
            sender_payoff=10.0,
            receiver_payoff=20.0,
        )

        mock_strategy.update.assert_called_once_with(
            Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0
        )
