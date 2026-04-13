# ABOUTME: Live integration tests that call the real OpenRouter API
# ABOUTME: Skipped unless OPENROUTER_API_KEY is set; run with -m live_llm

import os

import pytest

from garbling_gym.core.agents.receiver import ReceiverAgent
from garbling_gym.core.agents.sender import SenderAgent
from garbling_gym.core.agents.strategies.llm_hybrid import HybridLLMStrategy
from garbling_gym.core.agents.strategies.llm_pure import PureLLMStrategy
from garbling_gym.core.agents.strategies.openrouter import make_openrouter_caller
from garbling_gym.core.config import GameConfig
from garbling_gym.core.game import Game
from garbling_gym.core.types import Action, AssetQuality, Signal

# Model to use for live tests — verify this ID on https://openrouter.ai/models
NEMOTRON_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"


def _get_caller():
    """Return an OpenRouter caller or skip the test if the key is absent."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")
    return make_openrouter_caller(model=NEMOTRON_MODEL, api_key=api_key)


@pytest.mark.live_llm
class TestHybridLLMStrategyLive:
    """HybridLLMStrategy makes real decisions via OpenRouter."""

    def test_returns_valid_action(self):
        """Single choose_action call returns BUY or PASS."""
        caller = _get_caller()
        strategy = HybridLLMStrategy(llm_caller=caller)
        action = strategy.choose_action(Signal.GOOD, round_num=1, total_rounds=10)
        assert isinstance(action, Action), f"Expected Action, got {action!r}"

    def test_completes_five_round_game(self):
        """HybridLLM strategy completes a 5-round game end-to-end."""
        caller = _get_caller()
        strategy = HybridLLMStrategy(llm_caller=caller)
        config = GameConfig(num_rounds=5, use_llm=True, receiver_strategy="hybrid-llm")
        game = Game(config, sender=SenderAgent(), receiver=ReceiverAgent(strategy=strategy))
        results = game.play_game(verbose=False)

        assert results.total_rounds == 5
        assert 0.0 <= results.buy_rate <= 1.0
        assert isinstance(results.receiver_total, (int, float))


@pytest.mark.live_llm
class TestPureLLMStrategyLive:
    """PureLLMStrategy makes real decisions via OpenRouter with CoT prompting."""

    def test_returns_valid_action(self):
        """Single choose_action call returns BUY or PASS."""
        caller = _get_caller()
        strategy = PureLLMStrategy(llm_caller=caller, use_cot=True)
        action = strategy.choose_action(Signal.NEUTRAL, round_num=1, total_rounds=10)
        assert isinstance(action, Action), f"Expected Action, got {action!r}"

    def test_completes_five_round_game(self):
        """PureLLM strategy completes a 5-round game end-to-end."""
        caller = _get_caller()
        strategy = PureLLMStrategy(llm_caller=caller, use_cot=True)
        config = GameConfig(num_rounds=5, use_llm=True, receiver_strategy="pure-llm")
        game = Game(config, sender=SenderAgent(), receiver=ReceiverAgent(strategy=strategy))
        results = game.play_game(verbose=False)

        assert results.total_rounds == 5
        assert 0.0 <= results.buy_rate <= 1.0
        assert isinstance(results.receiver_total, (int, float))

    def test_history_in_prompt_influences_decision(self):
        """After observing several rounds, the strategy uses history in its prompt."""
        caller = _get_caller()
        strategy = PureLLMStrategy(llm_caller=caller, use_cot=True, history_window=5)
        from garbling_gym.core.types import Signal

        # Feed 3 rounds of GOOD→HIGH observations
        for _ in range(3):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)

        assert strategy.history_length() == 3
        action = strategy.choose_action(Signal.GOOD, round_num=4, total_rounds=10)
        assert isinstance(action, Action)
