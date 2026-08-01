# ABOUTME: No-regret receiver strategies: Regret Matching and Multiplicative Weights (Hedge)
# ABOUTME: Both converge to optimal play without assumptions on sender's strategy

import random
from typing import Any, Dict, Optional

import numpy as np

from . import ReceiverStrategy
from ...types import Action, AssetQuality, Signal


_QUALITIES = ("LOW", "MEDIUM", "HIGH")
_SIGNALS = ("BAD", "NEUTRAL", "GOOD")
_ACTIONS = ("BUY", "PASS")

# Receiver payoff u_R(action, quality) — default; overridable via configure().
_DEFAULT_RECEIVER_PAYOFF = {
    ("BUY", "LOW"):    -15.0,
    ("BUY", "MEDIUM"):   5.0,
    ("BUY", "HIGH"):    20.0,
    ("PASS", "LOW"):     0.0,
    ("PASS", "MEDIUM"):  0.0,
    ("PASS", "HIGH"):    0.0,
}


class RegretMatchingStrategy(ReceiverStrategy):
    """
    No-regret learning via Regret Matching (Hart & Mas-Colell, 2000).

    For each signal, tracks cumulative counterfactual regret for each action.
    Mixed strategy proportional to positive regrets; uniform when all regrets <= 0.
    Converges to a correlated equilibrium in the long run.
    """

    def __init__(self) -> None:
        self._regret: Dict[str, Dict[str, float]] = {}
        self._payoffs: Dict[Any, float] = dict(_DEFAULT_RECEIVER_PAYOFF)
        self.reset()

    def configure(self, prior, receiver_payoffs) -> None:
        """Inject the receiver (action, quality) payoff table from GameConfig."""
        self._payoffs = dict(receiver_payoffs)

    def choose_action(self, signal: Any, round_num: int, total_rounds: int) -> Action:
        signal_name = signal.name if isinstance(signal, Signal) else str(signal)
        pos = {a: max(self._regret[signal_name][a], 0.0) for a in _ACTIONS}
        total = sum(pos.values())
        if total <= 0:
            # Uniform when all regrets non-positive
            p_buy = 0.5
        else:
            p_buy = pos["BUY"] / total
        return Action.BUY if random.random() < p_buy else Action.PASS

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

        actual_u = self._payoffs[(action_name, quality_name)]
        # Counterfactual regret: what would each action have earned vs. what we got
        for a in _ACTIONS:
            counterfactual_u = self._payoffs[(a, quality_name)]
            self._regret[signal_name][a] += counterfactual_u - actual_u

    def reset(self) -> None:
        self._regret = {s: {a: 0.0 for a in _ACTIONS} for s in _SIGNALS}

    def get_diagnostics(self) -> Dict[str, Any]:
        mixed = {}
        for s in _SIGNALS:
            pos = {a: max(self._regret[s][a], 0.0) for a in _ACTIONS}
            total = sum(pos.values())
            mixed[s] = {a: pos[a] / total if total > 0 else 0.5 for a in _ACTIONS}
        return {
            "cumulative_regret": {s: dict(self._regret[s]) for s in _SIGNALS},
            "mixed_strategy": mixed,
        }


class HedgeStrategy(ReceiverStrategy):
    """
    Multiplicative Weights Update (Hedge / Freund & Schapire, 1997).

    Maintains per-signal weights for BUY and PASS.  Updates use the full-information
    loss (both actions' losses are observed after quality is revealed).
    Provides O(sqrt(T ln K)) worst-case regret bound.

    Args:
        learning_rate: Step size eta. If None, uses optimal eta = sqrt(8 ln(2) / T)
            computed from total_rounds at decision time.
    """

    _U_MAX = 20.0
    _U_MIN = -15.0
    _U_RANGE = _U_MAX - _U_MIN  # 35.0

    def __init__(self, learning_rate: Optional[float] = None) -> None:
        self._lr = learning_rate
        self._weights: Dict[str, Dict[str, float]] = {}
        self._payoffs: Dict[Any, float] = dict(_DEFAULT_RECEIVER_PAYOFF)
        self._round = 0
        self.reset()

    def configure(self, prior, receiver_payoffs) -> None:
        """Inject the receiver (action, quality) payoff table from GameConfig."""
        self._payoffs = dict(receiver_payoffs)

    def choose_action(self, signal: Any, round_num: int, total_rounds: int) -> Action:
        signal_name = signal.name if isinstance(signal, Signal) else str(signal)
        w_buy = self._weights[signal_name]["BUY"]
        w_pass = self._weights[signal_name]["PASS"]
        total = w_buy + w_pass
        p_buy = w_buy / total if total > 0 else 0.5
        return Action.BUY if random.random() < p_buy else Action.PASS

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

        self._round += 1
        # Compute eta: either fixed or optimal for current round count
        eta = self._lr if self._lr is not None else np.sqrt(8.0 * np.log(2) / max(self._round, 1))

        # Full-information losses (normalized to [0, 1])
        for a in _ACTIONS:
            u = self._payoffs[(a, quality_name)]
            loss = (self._U_MAX - u) / self._U_RANGE
            self._weights[signal_name][a] *= np.exp(-eta * loss)

    def reset(self) -> None:
        self._weights = {s: {a: 1.0 for a in _ACTIONS} for s in _SIGNALS}
        self._round = 0

    def get_diagnostics(self) -> Dict[str, Any]:
        mixed = {}
        for s in _SIGNALS:
            total = sum(self._weights[s].values())
            mixed[s] = {a: self._weights[s][a] / total if total > 0 else 0.5 for a in _ACTIONS}
        return {
            "weights": {s: dict(self._weights[s]) for s in _SIGNALS},
            "mixed_strategy": mixed,
        }
