# ABOUTME: Hybrid LLM + Bayesian receiver strategy
# ABOUTME: Computes Bayesian analysis in code, uses LLM for qualitative final decision

from typing import Any, Callable, Dict, List, Optional, Tuple

from .bayesian import DirichletBayesianStrategy
from . import ReceiverStrategy
from ...types import Action, AssetQuality, Signal


_QUALITIES = ("LOW", "MEDIUM", "HIGH")
_SIGNALS = ("BAD", "NEUTRAL", "GOOD")
_PRIOR = {"LOW": 0.3, "MEDIUM": 0.4, "HIGH": 0.3}
_PAYOFFS = {"LOW": -15.0, "MEDIUM": 5.0, "HIGH": 20.0}

_SYSTEM_PROMPT = """You are the RECEIVER in an information economics game.
You will receive a quantitative Bayesian analysis computed from the game history.
Use this analysis along with qualitative reasoning to decide: BUY or PASS.

Respond with exactly one word: BUY or PASS."""


LLMCaller = Callable[[str, str], str]


class HybridLLMStrategy(ReceiverStrategy):
    """
    Combines formal Bayesian inference with LLM qualitative reasoning.

    Maintains a DirichletBayesian state internally for rigorous posterior
    computation.  At decision time, builds a prompt enriched with the
    computed analysis and calls the LLM for a final verdict.

    Args:
        llm_caller: Callable(system_prompt, user_prompt) -> response_text.
            If None, falls back to the fallback_strategy.
        fallback_strategy: ReceiverStrategy used when llm_caller is absent
            or raises an exception.
        history_window: Number of recent rounds to include in the prompt.
        prior_strength: Dirichlet prior strength for the internal Bayesian state.
    """

    def __init__(
        self,
        llm_caller: Optional[LLMCaller] = None,
        fallback_strategy: Optional[ReceiverStrategy] = None,
        history_window: int = 10,
        prior_strength: float = 1.0,
    ) -> None:
        self._llm_caller = llm_caller
        self._fallback = fallback_strategy or DirichletBayesianStrategy(prior_strength=prior_strength)
        self._window = history_window
        self._bayesian = DirichletBayesianStrategy(prior_strength=prior_strength)
        # Raw history for prompt construction
        self._history: List[Tuple[str, str, str, float]] = []  # signal, action, quality, receiver_payoff

    def choose_action(self, signal: Any, round_num: int, total_rounds: int) -> Action:
        if self._llm_caller is None:
            return self._fallback.choose_action(signal, round_num, total_rounds)

        signal_name = signal.name if isinstance(signal, Signal) else str(signal)
        prompt = self._build_prompt(signal_name, round_num, total_rounds)
        try:
            response = self._llm_caller(_SYSTEM_PROMPT, prompt)
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
        self._bayesian.update(signal, action, true_quality, sender_payoff, receiver_payoff)
        self._fallback.update(signal, action, true_quality, sender_payoff, receiver_payoff)
        self._history.append((signal_name, action_name, quality_name, receiver_payoff))

    def reset(self) -> None:
        self._bayesian.reset()
        self._fallback.reset()
        self._history = []

    def get_diagnostics(self) -> Dict[str, Any]:
        diag = self._bayesian.get_diagnostics()
        return diag

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, signal_name: str, round_num: int, total_rounds: int) -> str:
        # Compute Bayesian analysis
        pm = self._bayesian._posterior_mean()
        unnorm = {q: pm[q].get(signal_name, 1e-9) * _PRIOR[q] for q in _QUALITIES}
        total = sum(unnorm.values()) or 1.0
        posterior = {q: unnorm[q] / total for q in _QUALITIES}
        ev_buy = sum(posterior[q] * _PAYOFFS[q] for q in _QUALITIES)

        history_lines = ""
        recent = self._history[-self._window:]
        if recent:
            history_lines = "\n".join(
                f"  Signal={s}, Action={a}, Quality={q}, Receiver payoff={p:.0f}"
                for s, a, q, p in recent
            )
        else:
            history_lines = "  (no history yet)"

        posterior_lines = "  " + ", ".join(
            f"P({q}|{signal_name})={posterior[q]:.2f}" for q in _QUALITIES
        )

        return f"""Round {round_num}/{total_rounds}
Current signal: {signal_name}

Bayesian posterior:
{posterior_lines}

E[BUY] = {ev_buy:.2f}  (positive = Bayesian recommends BUY)

Recent history (last {len(recent)} rounds):
{history_lines}

Consider: has the sender shifted strategy recently? Is the signal reliable?
Your decision (BUY or PASS):"""

    def _parse_response(self, response: str) -> Action:
        upper = response.strip().upper()
        if "BUY" in upper:
            return Action.BUY
        return Action.PASS
