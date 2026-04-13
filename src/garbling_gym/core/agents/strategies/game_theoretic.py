# ABOUTME: Game-theoretic receiver strategy via level-k iterated best response
# ABOUTME: Starts from game-theoretic priors, transitions to empirical best response

from typing import Any, Dict, List, Tuple

import numpy as np

from . import ReceiverStrategy
from ...types import Action, AssetQuality, Signal


_QUALITIES = ("LOW", "MEDIUM", "HIGH")
_SIGNALS = ("BAD", "NEUTRAL", "GOOD")
_PRIOR = {"LOW": 0.3, "MEDIUM": 0.4, "HIGH": 0.3}
_PAYOFFS = {"LOW": -15.0, "MEDIUM": 5.0, "HIGH": 20.0}


class LevelKStrategy(ReceiverStrategy):
    """
    Level-k iterated best-response reasoning.

    At level 1, treats the sender as noise (uniform garbling), which under the
    default prior yields E[BUY] = 3.5 > 0 for every signal → always BUY.

    As observations accumulate, transitions toward a best-response to the
    empirically estimated garbling matrix.  The weight placed on the empirical
    estimate grows with the number of observations.

    Args:
        max_level: Maximum level of strategic reasoning (currently only 1 and 2 matter).
        empirical_weight_growth: Additional weight on empirical estimate per observation.
    """

    def __init__(
        self,
        max_level: int = 2,
        empirical_weight_growth: float = 0.05,
    ) -> None:
        self._max_level = max_level
        self._growth = empirical_weight_growth
        self._history: List[Tuple[str, str]] = []  # (signal_name, quality_name)
        self.reset()

    def choose_action(self, signal: Any, round_num: int, total_rounds: int) -> Action:
        signal_name = signal.name if isinstance(signal, Signal) else str(signal)

        # Empirical weight: grows with observations, capped at 1
        empirical_weight = min(len(self._history) * self._growth, 1.0)
        level1_weight = 1.0 - empirical_weight

        # Level-1: uniform garbling → posterior = prior → E[BUY] = 3.5 > 0 always
        level1_ev = sum(_PRIOR[q] * _PAYOFFS[q] for q in _QUALITIES)  # = 3.5

        # Empirical best response
        empirical_ev = self._compute_empirical_ev(signal_name)

        blended_ev = level1_weight * level1_ev + empirical_weight * empirical_ev
        return Action.BUY if blended_ev > 0 else Action.PASS

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
        self._history.append((signal_name, quality_name))

    def reset(self) -> None:
        self._history = []

    def get_diagnostics(self) -> Dict[str, Any]:
        empirical_weight = min(len(self._history) * self._growth, 1.0)
        return {
            "effective_level": 1 if empirical_weight < 0.5 else 2,
            "empirical_weight": empirical_weight,
            "observation_count": len(self._history),
        }

    def _compute_empirical_ev(self, signal_name: str) -> float:
        """
        Compute E[BUY | signal] using the empirical garbling estimate.

        Uses Laplace-smoothed frequency counts to estimate P(signal|quality),
        then applies Bayes' rule.
        """
        if not self._history:
            return sum(_PRIOR[q] * _PAYOFFS[q] for q in _QUALITIES)

        # Count co-occurrences with Laplace smoothing
        counts: Dict[str, Dict[str, int]] = {
            q: {s: 1 for s in _SIGNALS} for q in _QUALITIES
        }
        quality_totals: Dict[str, int] = {q: len(_SIGNALS) for q in _QUALITIES}

        for sig, qual in self._history:
            counts[qual][sig] += 1
            quality_totals[qual] += 1

        # P(signal | quality)
        p_sig_given_qual = {
            q: counts[q][signal_name] / quality_totals[q]
            for q in _QUALITIES
        }

        # Bayes
        unnorm = {q: p_sig_given_qual[q] * _PRIOR[q] for q in _QUALITIES}
        total = sum(unnorm.values()) or 1.0
        posterior = {q: unnorm[q] / total for q in _QUALITIES}

        return sum(posterior[q] * _PAYOFFS[q] for q in _QUALITIES)
