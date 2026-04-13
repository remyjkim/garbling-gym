# ABOUTME: Pure LLM receiver strategy with chain-of-thought prompting
# ABOUTME: Uses in-context learning with expanded history and CoT reasoning structure

from typing import Any, Callable, Dict, List, Optional, Tuple

from .legacy_heuristic import LegacyHeuristicStrategy
from . import ReceiverStrategy
from ...types import Action, AssetQuality, Signal


LLMCaller = Callable[[str, str], str]

_SYSTEM_PROMPT_BASE = """You are the RECEIVER in an information economics game.
You observe garbled signals about asset quality and must decide: BUY or PASS.

Payoffs:
- BUY + LOW quality: -15
- BUY + MEDIUM quality: +5
- BUY + HIGH quality: +20
- PASS: 0

Prior: P(LOW)=30%, P(MEDIUM)=40%, P(HIGH)=30%"""

_COT_INSTRUCTION = """
Step through your reasoning:
1. Consider the observed signal and what it implies about quality
2. Review how the sender has behaved in the past
3. Estimate your posterior P(quality|signal)
4. Compute your expected value of buying vs. passing
5. State your decision: BUY or PASS"""


class PureLLMStrategy(ReceiverStrategy):
    """
    In-context learning receiver strategy using chain-of-thought prompting.

    Maintains a rolling history of observations and includes the most recent
    history_window rounds in each prompt.  When use_cot=True, prompts the
    LLM to reason through the decision before answering.

    Args:
        llm_caller: Callable(system_prompt, user_prompt) -> response_text.
        fallback_strategy: Used when llm_caller is None or raises.
        history_window: Number of recent rounds included in the prompt.
        use_cot: If True, add explicit chain-of-thought reasoning instructions.
    """

    def __init__(
        self,
        llm_caller: Optional[LLMCaller] = None,
        fallback_strategy: Optional[ReceiverStrategy] = None,
        history_window: int = 15,
        use_cot: bool = True,
    ) -> None:
        self._llm_caller = llm_caller
        self._fallback = fallback_strategy or LegacyHeuristicStrategy()
        self._window = history_window
        self._use_cot = use_cot
        self._history: List[Tuple[str, str, str, float]] = []  # signal, action, quality, receiver_payoff

    def choose_action(self, signal: Any, round_num: int, total_rounds: int) -> Action:
        if self._llm_caller is None:
            return self._fallback.choose_action(signal, round_num, total_rounds)

        signal_name = signal.name if isinstance(signal, Signal) else str(signal)
        system_prompt = _SYSTEM_PROMPT_BASE + (_COT_INSTRUCTION if self._use_cot else "")
        user_prompt = self._build_prompt(signal_name, round_num, total_rounds)
        try:
            response = self._llm_caller(system_prompt, user_prompt)
            return self._parse_response(response)
        except Exception:
            return self._fallback.choose_action(signal, round_num, total_rounds)

    def update(
        self,
        signal: Any,
        action: Action,
        true_quality: AssetQuality,
        sender_payoff: float,
        receiver_payoff: float,
    ) -> None:
        signal_name = signal.name if isinstance(signal, Signal) else str(signal)
        action_name = action.name
        quality_name = true_quality.name if isinstance(true_quality, AssetQuality) else str(true_quality)
        self._history.append((signal_name, action_name, quality_name, receiver_payoff))
        self._fallback.update(signal, action, true_quality, sender_payoff, receiver_payoff)

    def reset(self) -> None:
        self._history = []
        self._fallback.reset()

    def history_length(self) -> int:
        return len(self._history)

    def get_diagnostics(self) -> Dict[str, Any]:
        return {"history_length": len(self._history), "history_window": self._window}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, signal_name: str, round_num: int, total_rounds: int) -> str:
        recent = self._history[-self._window:]
        if recent:
            history_lines = "\n".join(
                f"  Round {i + round_num - len(recent)}: "
                f"Signal={s}, You chose {a}, Quality revealed={q}, Payoff={p:.0f}"
                for i, (s, a, q, p) in enumerate(recent, 1)
            )
        else:
            history_lines = "  (no history yet)"

        return f"""Round {round_num}/{total_rounds}
Observed signal: {signal_name}

Recent history (last {len(recent)} rounds, window={self._window}):
{history_lines}

Consider what the signal tells you about quality given the sender's past behavior.
Decision (BUY or PASS):"""

    def _parse_response(self, response: str) -> Action:
        upper = response.strip().upper()
        if "BUY" in upper:
            return Action.BUY
        return Action.PASS
