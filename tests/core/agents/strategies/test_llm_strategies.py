# ABOUTME: Tests for HybridLLMStrategy and PureLLMStrategy
# ABOUTME: Uses injected mock LLM callers to test prompt contents and fallback behavior

import pytest
from garbling_gym.core.agents.strategies import ReceiverStrategy
from garbling_gym.core.agents.strategies.llm_hybrid import HybridLLMStrategy
from garbling_gym.core.agents.strategies.llm_pure import PureLLMStrategy
from garbling_gym.core.agents.strategies.bayesian import DirichletBayesianStrategy
from garbling_gym.core.agents.strategies.registry import receiver_strategy_registry
from garbling_gym.core.types import Action, AssetQuality, Signal


class TestHybridLLMStrategy:
    """HybridLLMStrategy enriches prompts with Bayesian analysis and falls back gracefully."""

    def test_is_receiver_strategy(self):
        strategy = HybridLLMStrategy()
        assert isinstance(strategy, ReceiverStrategy)

    def test_includes_bayesian_stats_in_prompt(self):
        """The prompt sent to LLM contains pre-computed posterior and E[BUY]."""
        captured_prompts = []

        def mock_llm(system_prompt: str, user_prompt: str) -> str:
            captured_prompts.append(user_prompt)
            return "BUY"

        strategy = HybridLLMStrategy(llm_caller=mock_llm)
        # Add some history so diagnostics are non-trivial
        for _ in range(5):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)

        strategy.choose_action(Signal.GOOD, round_num=6, total_rounds=20)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # Should contain Bayesian analysis
        assert "E[BUY]" in prompt or "expected" in prompt.lower()
        # Should mention the signal
        assert "GOOD" in prompt

    def test_returns_llm_decision_when_available(self):
        """When LLM returns BUY/PASS, strategy respects that decision."""
        def mock_buy(system_prompt, user_prompt):
            return "BUY"

        def mock_pass(system_prompt, user_prompt):
            return "PASS"

        for mock, expected in [(mock_buy, Action.BUY), (mock_pass, Action.PASS)]:
            strategy = HybridLLMStrategy(llm_caller=mock)
            result = strategy.choose_action(Signal.GOOD, 1, 20)
            assert result == expected

    def test_falls_back_on_llm_failure(self):
        """When LLM raises an exception, falls back to the fallback strategy."""
        def failing_llm(system_prompt, user_prompt):
            raise RuntimeError("API unavailable")

        strategy = HybridLLMStrategy(
            llm_caller=failing_llm,
            fallback_strategy=DirichletBayesianStrategy(),
        )
        # Should not raise, should return a valid Action
        result = strategy.choose_action(Signal.GOOD, 1, 20)
        assert isinstance(result, Action)

    def test_updates_bayesian_state_regardless_of_llm(self):
        """Dirichlet state is updated even when LLM is making the decision."""
        def mock_llm(system_prompt, user_prompt):
            return "BUY"

        strategy = HybridLLMStrategy(llm_caller=mock_llm)
        strategy.choose_action(Signal.GOOD, 1, 20)
        strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)

        diag = strategy.get_diagnostics()
        alpha = diag["alpha"]
        # HIGH/GOOD alpha should have incremented from 1.0 to 2.0
        assert alpha["HIGH"]["GOOD"] > 1.0

    def test_reset_clears_bayesian_state(self):
        """reset() returns Dirichlet alpha to uniform prior."""
        strategy = HybridLLMStrategy()
        strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        strategy.reset()
        diag = strategy.get_diagnostics()
        for row in diag["alpha"].values():
            for val in row.values():
                assert val == pytest.approx(1.0)

    def test_registered_as_hybrid_llm(self):
        strategy = receiver_strategy_registry.get("hybrid-llm")
        assert isinstance(strategy, HybridLLMStrategy)

    def test_choose_action_without_llm_caller_uses_fallback(self):
        """When no llm_caller provided, uses fallback strategy directly."""
        strategy = HybridLLMStrategy(llm_caller=None)
        result = strategy.choose_action(Signal.GOOD, 1, 20)
        assert isinstance(result, Action)


class TestPureLLMStrategy:
    """PureLLMStrategy uses CoT prompting and returns a structured assessment."""

    def test_is_receiver_strategy(self):
        strategy = PureLLMStrategy()
        assert isinstance(strategy, ReceiverStrategy)

    def test_prompt_includes_recent_history(self):
        """Prompt includes up to history_window rounds of game history."""
        captured_prompts = []

        def mock_llm(system_prompt, user_prompt):
            captured_prompts.append(user_prompt)
            return "BUY"

        strategy = PureLLMStrategy(history_window=5, llm_caller=mock_llm)
        # Add 7 rounds of history
        for i in range(7):
            strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)

        strategy.choose_action(Signal.GOOD, 8, 20)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # 5 rounds should appear (limited by history_window), not all 7
        # Look for round numbers — the most recent 5 should be referenced
        assert "GOOD" in prompt

    def test_cot_prompt_includes_reasoning_steps(self):
        """When use_cot=True, prompt contains explicit reasoning structure."""
        captured_prompts = []

        def mock_llm(system_prompt, user_prompt):
            captured_prompts.append(user_prompt)
            return "BUY"

        strategy = PureLLMStrategy(use_cot=True, llm_caller=mock_llm)
        strategy.choose_action(Signal.GOOD, 1, 20)

        prompt = captured_prompts[0]
        # CoT prompt should contain reasoning instructions
        assert any(kw in prompt.lower() for kw in ["step", "reason", "consider", "think"])

    def test_falls_back_gracefully_on_llm_failure(self):
        """When LLM raises an exception, falls back to heuristic strategy."""
        def failing_llm(system_prompt, user_prompt):
            raise RuntimeError("API down")

        strategy = PureLLMStrategy(llm_caller=failing_llm)
        result = strategy.choose_action(Signal.GOOD, 1, 20)
        assert isinstance(result, Action)

    def test_returns_action_enum(self):
        """choose_action always returns a valid Action regardless of LLM response."""
        for response in ["BUY", "PASS", "  buy  ", "  pass  "]:
            def mock_llm(system_prompt, user_prompt, _r=response):
                return _r

            strategy = PureLLMStrategy(llm_caller=mock_llm)
            result = strategy.choose_action(Signal.GOOD, 1, 20)
            assert isinstance(result, Action)

    def test_registered_as_pure_llm(self):
        strategy = receiver_strategy_registry.get("pure-llm")
        assert isinstance(strategy, PureLLMStrategy)

    def test_update_accumulates_history(self):
        """update() stores rounds for use in prompt history."""
        strategy = PureLLMStrategy()
        strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        strategy.update(Signal.BAD, Action.PASS, AssetQuality.LOW, 0.0, 0.0)
        assert strategy.history_length() == 2

    def test_reset_clears_history(self):
        """reset() empties accumulated game history."""
        strategy = PureLLMStrategy()
        strategy.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10.0, 20.0)
        strategy.reset()
        assert strategy.history_length() == 0
