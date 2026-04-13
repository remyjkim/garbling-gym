# ABOUTME: Legacy Bayesian heuristic receiver strategy extracted from ReceiverAgent
# ABOUTME: Implements empirical signal reliability estimation with Bayesian posterior updating

import random
from typing import Any, Dict, List, Tuple

import numpy as np

from . import ReceiverStrategy
from ...types import Action, AssetQuality, Signal


# Default P(signal | quality) before any observations
_DEFAULT_RELIABILITY: Dict[str, Dict[str, float]] = {
    "GOOD":    {"LOW": 0.2, "MEDIUM": 0.4, "HIGH": 0.8},
    "NEUTRAL": {"LOW": 0.3, "MEDIUM": 0.4, "HIGH": 0.15},
    "BAD":     {"LOW": 0.5, "MEDIUM": 0.2, "HIGH": 0.05},
}

_PRIOR = {"LOW": 0.3, "MEDIUM": 0.4, "HIGH": 0.3}


class LegacyHeuristicStrategy(ReceiverStrategy):
    """
    Bayesian heuristic decision strategy for the receiver.

    Maintains a history of observations and uses empirical frequency counts
    (with Laplace smoothing) to estimate P(signal | quality), then applies
    Bayes' rule to compute a posterior over quality and takes the action with
    positive expected value.
    """

    def __init__(self) -> None:
        # Each entry: (signal_name, quality_name, receiver_payoff)
        self._history: List[Tuple[str, str, float]] = []

    def choose_action(self, signal: Any, round_num: int, total_rounds: int) -> Action:
        signal_name = signal.name if isinstance(signal, Signal) else str(signal)

        reliability = self._estimate_reliability(signal_name)
        posterior = self._compute_posterior(signal_name, reliability)

        expected_buy = (
            posterior["LOW"] * (-15)
            + posterior["MEDIUM"] * 5
            + posterior["HIGH"] * 20
        )

        # Risk aversion based on recent payoffs
        if self._history:
            recent_payoffs = [p for _, _, p in self._history[-10:]]
            avg_payoff = float(np.mean(recent_payoffs))
            if avg_payoff < -5:
                expected_buy -= 3
            elif avg_payoff > 5:
                expected_buy += 1

        # Sigmoid-based stochastic decision (bounded rationality)
        prob_buy = 1.0 / (1.0 + np.exp(-(expected_buy - 1)))
        return Action.BUY if random.random() < prob_buy else Action.PASS

    def update(
        self,
        signal: Any,
        action: Action,
        true_quality: AssetQuality,
        sender_payoff: float,
        receiver_payoff: float,
    ) -> None:
        signal_name = signal.name if isinstance(signal, Signal) else str(signal)
        quality_name = true_quality.name if isinstance(true_quality, AssetQuality) else str(true_quality)
        self._history.append((signal_name, quality_name, receiver_payoff))

    def reset(self) -> None:
        self._history = []

    def history_length(self) -> int:
        return len(self._history)

    def get_diagnostics(self) -> Dict[str, Any]:
        return {"history_length": len(self._history)}

    def _estimate_reliability(self, signal_name: str) -> Dict[str, float]:
        """Estimate P(signal_name | quality) from stored history with Laplace smoothing."""
        if len(self._history) < 3:
            return _DEFAULT_RELIABILITY.get(signal_name, _DEFAULT_RELIABILITY["NEUTRAL"])

        counts = {q: 0 for q in ("LOW", "MEDIUM", "HIGH")}
        totals = {q: 0 for q in ("LOW", "MEDIUM", "HIGH")}

        for sig, qual, _ in self._history:
            if qual in totals:
                totals[qual] += 1
                if sig == signal_name:
                    counts[qual] += 1

        estimated: Dict[str, float] = {}
        default = _DEFAULT_RELIABILITY.get(signal_name, _DEFAULT_RELIABILITY["NEUTRAL"])
        for qual in ("LOW", "MEDIUM", "HIGH"):
            if totals[qual] > 0:
                estimated[qual] = (counts[qual] + 1) / (totals[qual] + 3)
            else:
                estimated[qual] = default.get(qual, 0.33)

        return estimated

    def _compute_posterior(
        self, signal_name: str, likelihood: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute P(quality | signal) via Bayes' rule."""
        unnormalized = {
            q: likelihood.get(q, 0.33) * _PRIOR[q] for q in ("LOW", "MEDIUM", "HIGH")
        }
        total = sum(unnormalized.values())
        if total == 0:
            return dict(_PRIOR)
        return {q: v / total for q, v in unnormalized.items()}
