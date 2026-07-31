# ABOUTME: Non-stationary bandit receiver strategies: SW-UCB and EXP3.S
# ABOUTME: Treats each signal as an independent 2-armed bandit with sliding window

import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from . import ReceiverStrategy
from ...types import Action, AssetQuality, Signal


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
_U_MAX = 20.0
_U_MIN = -15.0
_U_RANGE = _U_MAX - _U_MIN  # 35.0


class BanditStrategy(ReceiverStrategy):
    """
    Per-signal sliding-window bandit strategy.

    Each signal is treated as an independent 2-armed bandit (BUY / PASS).
    Supports two variants:

    - ``"sw-ucb"``: Sliding-Window UCB. Maintains a fixed-size window of recent
      rounds per signal and selects the action with the highest UCB score.

    - ``"exp3s"``: EXP3.S (adversarial). Maintains importance-weighted estimates
      with forced exploration mixing.  Suitable when the sender may shift strategy
      adversarially.

    Args:
        window_size: Maximum number of recent observations per signal to keep.
        exploration_constant: UCB exploration coefficient c (SW-UCB only).
        variant: ``"sw-ucb"`` or ``"exp3s"``.
    """

    def __init__(
        self,
        window_size: int = 10,
        exploration_constant: float = 1.0,
        variant: str = "sw-ucb",
    ) -> None:
        self._window_size = window_size
        self._c = exploration_constant
        self._variant = variant
        self._payoffs: Dict[Any, float] = dict(_DEFAULT_RECEIVER_PAYOFF)
        # SW-UCB state: per signal, deque of (action_name, reward) tuples
        self._windows: Dict[str, Deque[Tuple[str, float]]] = {}
        # EXP3.S state: per signal, weights per action
        self._weights: Dict[str, Dict[str, float]] = {}
        self.reset()

    def configure(self, prior, receiver_payoffs) -> None:
        """Inject the receiver (action, quality) payoff table from GameConfig."""
        self._payoffs = dict(receiver_payoffs)

    def choose_action(self, signal: Any, round_num: int, total_rounds: int) -> Action:
        signal_name = signal.name if isinstance(signal, Signal) else str(signal)
        if self._variant == "sw-ucb":
            return self._choose_sw_ucb(signal_name)
        return self._choose_exp3s(signal_name, total_rounds)

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

        reward = self._payoffs[(action_name, quality_name)]

        if self._variant == "sw-ucb":
            win = self._windows[signal_name]
            win.append((action_name, reward))
            if len(win) > self._window_size:
                win.popleft()
        else:
            self._update_exp3s(signal_name, action_name, reward)

    def reset(self) -> None:
        self._windows = {s: deque() for s in _SIGNALS}
        self._weights = {s: {a: 1.0 for a in _ACTIONS} for s in _SIGNALS}

    def get_diagnostics(self) -> Dict[str, Any]:
        windows: Dict[str, Any] = {}
        for s in _SIGNALS:
            win = list(self._windows[s])
            count = len(win)
            windows[s] = {"count": count}
            if self._variant == "sw-ucb" and count > 0:
                buy_rewards = [r for a, r in win if a == "BUY"]
                windows[s]["mean_buy_reward"] = float(np.mean(buy_rewards)) if buy_rewards else None
                windows[s]["buy_count"] = len(buy_rewards)
            elif self._variant == "exp3s":
                total = sum(self._weights[s].values())
                windows[s]["weights"] = {
                    a: self._weights[s][a] / total for a in _ACTIONS
                }
        return {"windows": windows, "variant": self._variant}

    # ------------------------------------------------------------------
    # SW-UCB
    # ------------------------------------------------------------------

    def _choose_sw_ucb(self, signal_name: str) -> Action:
        win = list(self._windows[signal_name])
        n = len(win)

        if n == 0:
            # No data: UCB infinitely favors unexplored arms → random choice
            return random.choice([Action.BUY, Action.PASS])

        buy_rewards = [r for a, r in win if a == "BUY"]
        pass_rewards = [r for a, r in win if a == "PASS"]
        n_buy = len(buy_rewards)
        n_pass = len(pass_rewards)

        # PASS reward is always 0 (known deterministically) — no exploration needed
        mean_pass = 0.0
        ucb_pass = mean_pass  # no exploration bonus: value is exactly known

        mean_buy = float(np.mean(buy_rewards)) if n_buy > 0 else 0.0
        ucb_buy = mean_buy + (self._c * np.sqrt(np.log(n) / n_buy) if n_buy > 0 else float("inf"))

        # Normalise to [0,1] for comparison
        norm_buy = (ucb_buy - _U_MIN) / _U_RANGE
        norm_pass = (ucb_pass - _U_MIN) / _U_RANGE

        if abs(norm_buy - norm_pass) < 1e-9:
            return random.choice([Action.BUY, Action.PASS])
        return Action.BUY if norm_buy > norm_pass else Action.PASS

    # ------------------------------------------------------------------
    # EXP3.S
    # ------------------------------------------------------------------

    def _choose_exp3s(self, signal_name: str, total_rounds: int) -> Action:
        gamma = np.sqrt(2.0 * np.log(2) / max(total_rounds, 1))
        w = self._weights[signal_name]
        total_w = sum(w.values())
        # Mixed strategy: (1 - gamma) * w/total_w + gamma/K
        k = len(_ACTIONS)
        probs = {a: (1.0 - gamma) * w[a] / total_w + gamma / k for a in _ACTIONS}
        r = random.random()
        return Action.BUY if r < probs["BUY"] else Action.PASS

    def _update_exp3s(self, signal_name: str, action_name: str, reward: float) -> None:
        # Importance-weighted reward estimate (only for the action taken)
        # Approximate mixing probability for importance weighting
        total_w = sum(self._weights[signal_name].values()) or 1.0
        p_action = self._weights[signal_name][action_name] / total_w

        # Normalise reward to [0, 1]
        norm_reward = (reward - _U_MIN) / _U_RANGE
        reward_hat = norm_reward / max(p_action, 1e-9)

        # EXP3 update: only for the taken action
        k = len(_ACTIONS)
        self._weights[signal_name][action_name] *= np.exp(reward_hat / k)

        # Slide window tracking (reuse deque for obs count)
        win = self._windows[signal_name]
        win.append((action_name, reward))
        if len(win) > self._window_size:
            win.popleft()
