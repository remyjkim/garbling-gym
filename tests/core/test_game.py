# ABOUTME: Tests for Game orchestration
# ABOUTME: Verifies that game loop correctly calls receiver learning after each round

import pytest
from unittest.mock import MagicMock, patch
from garbling_gym.core.game import Game
from garbling_gym.core.config import GameConfig
from garbling_gym.core.agents.receiver import ReceiverAgent
from garbling_gym.core.agents.sender import SenderAgent
from garbling_gym.core.agents.strategies import ReceiverStrategy
from garbling_gym.core.types import Action, AssetQuality, Signal


class TestGameStrategyUpdate:
    """Game.play_round() calls receiver.learn() after payoffs are computed."""

    def test_play_round_calls_receiver_learn(self):
        """After each round, game calls receiver.learn() with round outcomes."""
        config = GameConfig(num_rounds=1, use_llm=False)
        sender = SenderAgent()
        receiver = ReceiverAgent()

        # Spy on the learn method
        original_learn = receiver.learn
        learn_calls = []

        def spy_learn(**kwargs):
            learn_calls.append(kwargs)
            return original_learn(**kwargs)

        receiver.learn = lambda **kwargs: learn_calls.append(kwargs) or None

        game = Game(config, sender=sender, receiver=receiver)
        round_result = game.play_round()

        # learn() must have been called once
        assert len(learn_calls) == 1
        call = learn_calls[0]

        # The signal, action, and payoffs should match the round result
        assert call["action"].name == round_result["action"]
        assert call["true_quality"].name == round_result["quality"]
        assert call["sender_payoff"] == round_result["sender_payoff"]
        assert call["receiver_payoff"] == round_result["receiver_payoff"]

    def test_play_game_calls_learn_each_round(self):
        """learn() is called once per round over a full game."""
        num_rounds = 5
        config = GameConfig(num_rounds=num_rounds, use_llm=False)
        sender = SenderAgent()
        receiver = ReceiverAgent()

        learn_call_count = [0]
        receiver.learn = lambda **kwargs: learn_call_count.__setitem__(0, learn_call_count[0] + 1)

        game = Game(config, sender=sender, receiver=receiver)
        game.play_game(verbose=False)

        assert learn_call_count[0] == num_rounds
