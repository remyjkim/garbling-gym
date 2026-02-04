# ABOUTME: Payoff structures for the garbling game
# ABOUTME: Defines rewards for different action-quality combinations

from dataclasses import dataclass, field
from typing import Dict, Tuple
from .types import Action, AssetQuality


@dataclass
class PayoffStructure:
    """
    Defines payoffs for (action, quality) pairs.
    payoffs[action][quality] = (sender_payoff, receiver_payoff)
    """
    payoffs: Dict[Action, Dict[AssetQuality, Tuple[float, float]]] = field(
        default_factory=lambda: {
            Action.BUY: {
                AssetQuality.LOW: (10, -15),     # Sender gains, receiver loses
                AssetQuality.MEDIUM: (10, 5),    # Both gain
                AssetQuality.HIGH: (10, 20),     # Both gain significantly
            },
            Action.PASS: {
                AssetQuality.LOW: (0, 0),
                AssetQuality.MEDIUM: (0, 0),
                AssetQuality.HIGH: (0, 0),
            }
        }
    )

    def get_payoffs(self, action: Action, quality: AssetQuality) -> Tuple[float, float]:
        """Get (sender_payoff, receiver_payoff) for given action and quality"""
        return self.payoffs[action][quality]
