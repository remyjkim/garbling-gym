# ABOUTME: Pluggable receiver learning strategy system
# ABOUTME: Defines the ReceiverStrategy interface all learning modes implement

from abc import ABC, abstractmethod
from typing import Any, Dict

from ...types import Action, AssetQuality, Signal


class ReceiverStrategy(ABC):
    """
    Pluggable learning strategy for the receiver agent.

    Each subclass implements a distinct approach to the receiver's core
    problem: given a signal and accumulated game history, decide BUY or PASS.

    The strategy is stateful — it accumulates observations across rounds
    via update() and uses them in subsequent choose_action() calls. Call
    reset() to return to the initial state for a new game.
    """

    @abstractmethod
    def choose_action(self, signal: Any, round_num: int, total_rounds: int) -> Action:
        """
        Decide BUY or PASS given the current signal.

        Args:
            signal: The observed signal (type depends on the channel tier:
                    Signal enum for discrete, float for continuous, str for NL)
            round_num: Current round number (1-indexed)
            total_rounds: Total rounds in the game

        Returns:
            Action.BUY or Action.PASS
        """
        ...

    @abstractmethod
    def update(
        self,
        signal: Any,
        action: Action,
        true_quality: AssetQuality,
        sender_payoff: float,
        receiver_payoff: float,
    ) -> None:
        """
        Learn from a completed round after true quality is revealed.

        Called once per round, after payoffs are computed.

        Args:
            signal: The signal that was observed this round
            action: The action that was taken (BUY or PASS)
            true_quality: The true asset quality revealed post-hoc
            sender_payoff: Sender's payoff this round
            receiver_payoff: Receiver's payoff this round
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset all accumulated state for a new game."""
        ...

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return internal state for logging and visualization.

        Override in subclasses to expose strategy-specific diagnostics.
        """
        return {}
