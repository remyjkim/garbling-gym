# ABOUTME: Configuration dataclass for game setup
# ABOUTME: Encapsulates all parameters needed to run a game

from dataclasses import dataclass, field
from typing import Dict
from .types import AssetQuality
from .payoffs import PayoffStructure


@dataclass
class GameConfig:
    """Configuration for the garbling economics game"""

    # Prior distribution over asset quality P(quality)
    prior: Dict[AssetQuality, float] = field(default_factory=lambda: {
        AssetQuality.LOW: 0.3,
        AssetQuality.MEDIUM: 0.4,
        AssetQuality.HIGH: 0.3,
    })

    # Payoff structure
    payoffs: PayoffStructure = field(default_factory=PayoffStructure)

    # Number of rounds to play
    num_rounds: int = 20

    # Use LLM agents (requires API key)
    use_llm: bool = False

    # LLM model to use if use_llm is True
    llm_model: str = "gpt-4o-mini"

    def __post_init__(self):
        """Validate configuration"""
        # Ensure prior is a valid probability distribution
        prior_sum = sum(self.prior.values())
        assert abs(prior_sum - 1.0) < 1e-6, f"Prior must sum to 1.0, got {prior_sum}"

        # Ensure all probabilities are non-negative
        assert all(p >= 0 for p in self.prior.values()), "Prior probabilities must be non-negative"

        # Ensure num_rounds is positive
        assert self.num_rounds > 0, "Number of rounds must be positive"
