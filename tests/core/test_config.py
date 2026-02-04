"""
Unit tests for game configuration.

Tests configuration validation and defaults.
"""

import pytest
from garbling_gym.core.config import GameConfig
from garbling_gym.core.types import AssetQuality
from garbling_gym.core.payoffs import PayoffStructure


class TestGameConfig:
    """Test GameConfig validation"""

    def test_default_config_valid(self):
        """Test that default config is valid"""
        config = GameConfig()

        assert config.num_rounds == 20
        assert config.use_llm is False
        assert config.llm_model == "gpt-4o-mini"
        assert isinstance(config.payoffs, PayoffStructure)

    def test_valid_prior_distribution(self):
        """Test creating config with valid prior"""
        config = GameConfig(
            prior={
                AssetQuality.LOW: 0.2,
                AssetQuality.MEDIUM: 0.5,
                AssetQuality.HIGH: 0.3,
            }
        )

        assert sum(config.prior.values()) == pytest.approx(1.0)

    def test_invalid_prior_not_normalized(self):
        """Test that prior must sum to 1"""
        with pytest.raises(AssertionError, match="sum to 1"):
            GameConfig(
                prior={
                    AssetQuality.LOW: 0.2,
                    AssetQuality.MEDIUM: 0.5,
                    AssetQuality.HIGH: 0.2,  # Sums to 0.9
                }
            )

    def test_invalid_prior_negative(self):
        """Test that prior probabilities must be non-negative"""
        with pytest.raises(AssertionError, match="non-negative"):
            GameConfig(
                prior={
                    AssetQuality.LOW: -0.1,
                    AssetQuality.MEDIUM: 0.6,
                    AssetQuality.HIGH: 0.5,
                }
            )

    def test_invalid_num_rounds_zero(self):
        """Test that num_rounds must be positive"""
        with pytest.raises(AssertionError, match="positive"):
            GameConfig(num_rounds=0)

    def test_invalid_num_rounds_negative(self):
        """Test that num_rounds cannot be negative"""
        with pytest.raises(AssertionError, match="positive"):
            GameConfig(num_rounds=-5)

    def test_custom_rounds(self):
        """Test setting custom number of rounds"""
        config = GameConfig(num_rounds=100)
        assert config.num_rounds == 100

    def test_llm_configuration(self):
        """Test LLM configuration options"""
        config = GameConfig(use_llm=True, llm_model="gpt-4o")

        assert config.use_llm is True
        assert config.llm_model == "gpt-4o"
