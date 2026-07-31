# ABOUTME: Convert a GameConfig into the solver's numpy view of the game
# ABOUTME: Pure data extraction — no dependence on the game loop or agents.

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import GameConfig
from ..types import Action, AssetQuality


# Canonical state ordering used throughout the theory package: LOW, MEDIUM, HIGH.
# This matches AssetQuality.value (0, 1, 2) and every strategy module's _QUALITIES tuple.
_STATE_ORDER: Sequence[AssetQuality] = (
    AssetQuality.LOW,
    AssetQuality.MEDIUM,
    AssetQuality.HIGH,
)


@dataclass(frozen=True)
class GameSpec:
    """The solver's view of a game: everything the benchmarks need, as numpy.

    Convention: arrays are indexed by state in the order LOW, MEDIUM, HIGH.

    Attributes:
        mu0: prior over states, shape (n_states,), sums to 1.
        receiver_buy: receiver payoff for the BUY action, by state, shape (n_states,).
        sender_buy: sender payoff for the BUY action, by state, shape (n_states,).
            In the default gym this is state-independent (+10), but it is kept
            per-state so a state-dependent (Crawford-Sobel bias) variant can
            override it without changing this interface.
        buy_threshold: the receiver buys iff E[receiver_buy | posterior] > buy_threshold.
            The default receiver decision rule is a strict-greater-than-zero test
            (see DirichletBayesianStrategy.choose_action), so the threshold is 0.0.
        n_states: number of states (3 for the gym).
    """

    mu0: np.ndarray
    receiver_buy: np.ndarray
    sender_buy: np.ndarray
    buy_threshold: float = 0.0
    n_states: int = 3

    def receiver_buys_at(self, mu: np.ndarray) -> bool:
        """Does the receiver BUY at posterior ``mu``?

        Implements the receiver's threshold rule:
        ``E[receiver_buy | mu] = mu · receiver_buy > buy_threshold``.
        The strict inequality mirrors ``DirichletBayesianStrategy`` (BUY iff
        ``expected_buy > 0``).
        """
        return float(mu @ self.receiver_buy) > self.buy_threshold

    def indirect_sender_value(self, mu: np.ndarray) -> float:
        """The sender's indirect value at posterior ``mu``: v_S(mu).

        With a binary receiver action, the sender's payoff depends only on
        whether the receiver buys:

            v_S(mu) = mu · sender_buy   if the receiver buys at mu
                    = 0                 otherwise

        (``sender_buy`` is already the per-state payoff for BUY; dotting with
        ``mu`` gives its expectation at the posterior. PASS yields 0 in every
        state, so the no-buy branch is identically 0.)
        """
        if self.receiver_buys_at(mu):
            return float(mu @ self.sender_buy)
        return 0.0


def game_spec(config: GameConfig) -> GameSpec:
    """Build a :class:`GameSpec` from a :class:`GameConfig`.

    Extracts the prior, the receiver's BUY payoffs, and the sender's BUY
    payoffs in the canonical LOW/MEDIUM/HIGH order.
    """
    mu0 = np.array([config.prior[q] for q in _STATE_ORDER], dtype=float)
    receiver_buy = np.empty(3, dtype=float)
    sender_buy = np.empty(3, dtype=float)
    for i, q in enumerate(_STATE_ORDER):
        s_pay, r_pay = config.payoffs.get_payoffs(Action.BUY, q)
        sender_buy[i] = s_pay
        receiver_buy[i] = r_pay
    return GameSpec(mu0=mu0, receiver_buy=receiver_buy, sender_buy=sender_buy)
