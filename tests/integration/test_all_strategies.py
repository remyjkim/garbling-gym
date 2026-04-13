# ABOUTME: Integration tests verifying all receiver strategies run complete games
# ABOUTME: Ensures every registered strategy works end-to-end with game orchestration

import pytest
from garbling_gym.core.config import GameConfig
from garbling_gym.core.game import Game
from garbling_gym.core.agents.sender import SenderAgent
from garbling_gym.core.agents.receiver import ReceiverAgent
from garbling_gym.core.agents.strategies.registry import receiver_strategy_registry


ALL_STRATEGIES = [
    "heuristic",
    "bayesian",
    "thompson",
    "regret-matching",
    "hedge",
    "level-k",
    "bandit",
    "hybrid-llm",
    "pure-llm",
]


class TestAllStrategiesRunGames:
    """Every registered strategy can complete a full game without error."""

    @pytest.mark.parametrize("strategy_name", ALL_STRATEGIES)
    def test_strategy_completes_full_game(self, strategy_name):
        """Strategy runs a 5-round game and produces valid results."""
        config = GameConfig(num_rounds=5, use_llm=False, receiver_strategy=strategy_name)
        sender = SenderAgent()
        strategy = receiver_strategy_registry.get(strategy_name)
        receiver = ReceiverAgent(strategy=strategy)

        game = Game(config, sender=sender, receiver=receiver)
        results = game.play_game(verbose=False)

        assert results.total_rounds == 5
        assert 0 <= results.buy_rate <= 1
        assert isinstance(results.receiver_total, (int, float))

    @pytest.mark.parametrize("strategy_name", ALL_STRATEGIES)
    def test_strategy_play_round_produces_diagnostics(self, strategy_name):
        """play_round() captures strategy diagnostics in round record."""
        config = GameConfig(num_rounds=1, use_llm=False)
        sender = SenderAgent()
        strategy = receiver_strategy_registry.get(strategy_name)
        receiver = ReceiverAgent(strategy=strategy)

        game = Game(config, sender=sender, receiver=receiver)
        round_result = game.play_round()

        assert "receiver_diagnostics" in round_result
        assert isinstance(round_result["receiver_diagnostics"], dict)

    def test_registry_has_all_expected_strategies(self):
        """All expected strategies are registered."""
        for name in ALL_STRATEGIES:
            assert receiver_strategy_registry.is_registered(name), f"'{name}' not registered"

    @pytest.mark.parametrize("strategy_name", ALL_STRATEGIES)
    def test_strategy_reset_between_games(self, strategy_name):
        """Strategy reset() enables clean state for consecutive games."""
        strategy = receiver_strategy_registry.get(strategy_name)

        config = GameConfig(num_rounds=5, use_llm=False)
        sender = SenderAgent()
        receiver = ReceiverAgent(strategy=strategy)

        # First game
        game1 = Game(config, sender=sender, receiver=receiver)
        game1.play_game(verbose=False)

        # Reset and second game
        strategy.reset()
        game2 = Game(config, sender=SenderAgent(), receiver=ReceiverAgent(strategy=strategy))
        results = game2.play_game(verbose=False)

        assert results.total_rounds == 5
