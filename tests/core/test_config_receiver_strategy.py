# ABOUTME: Tests for receiver_strategy field in GameConfig
# ABOUTME: Verifies the default value and that the field is accepted

import pytest
from garbling_gym.core.config import GameConfig


class TestGameConfigReceiverStrategy:
    """GameConfig includes a receiver_strategy field."""

    def test_default_receiver_strategy_is_heuristic(self):
        """receiver_strategy defaults to 'heuristic'."""
        config = GameConfig()
        assert config.receiver_strategy == "heuristic"

    def test_receiver_strategy_can_be_set(self):
        """receiver_strategy can be set to any string."""
        config = GameConfig(receiver_strategy="bayesian")
        assert config.receiver_strategy == "bayesian"
