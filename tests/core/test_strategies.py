"""
Unit tests for garbling strategies.

Tests strategy validation, signal generation, and informativeness scoring.
"""

import pytest
import numpy as np
from garbling_gym.core.strategies import GarblingStrategy, BUILTIN_STRATEGIES, strategy_registry
from garbling_gym.core.types import AssetQuality, Signal


class TestGarblingStrategy:
    """Test the base GarblingStrategy class"""

    def test_valid_strategy_creation(self):
        """Test creating a valid garbling strategy"""
        matrix = np.eye(3)  # Identity matrix (perfect information)
        strategy = GarblingStrategy(matrix=matrix, name="Perfect")

        assert strategy.matrix.shape == (3, 3)
        assert np.allclose(strategy.matrix.sum(axis=1), 1.0)

    def test_invalid_strategy_non_stochastic(self):
        """Test that non-stochastic matrices are rejected"""
        # Rows don't sum to 1
        matrix = np.array([
            [0.5, 0.3, 0.1],  # Sum = 0.9
            [0.3, 0.4, 0.3],
            [0.2, 0.3, 0.5],
        ])

        with pytest.raises(ValueError, match="Rows must sum to 1"):
            GarblingStrategy(matrix=matrix)

    def test_invalid_strategy_negative_probabilities(self):
        """Test that negative probabilities are rejected"""
        matrix = np.array([
            [-0.2, 0.7, 0.5],  # Contains negative
            [0.3, 0.4, 0.3],
            [0.2, 0.3, 0.5],
        ])

        with pytest.raises(ValueError, match="non-negative"):
            GarblingStrategy(matrix=matrix)

    def test_invalid_strategy_wrong_shape(self):
        """Test that wrong-shaped matrices are rejected"""
        matrix = np.array([[0.5, 0.5], [0.5, 0.5]])  # 2x2 instead of 3x3

        with pytest.raises(ValueError, match="3x3"):
            GarblingStrategy(matrix=matrix)

    def test_signal_generation(self):
        """Test that signals are generated according to probabilities"""
        # Full revelation strategy
        strategy = GarblingStrategy(matrix=np.eye(3), name="Perfect")

        # For full revelation, LOW quality should always give BAD signal
        signals = [strategy.get_signal(AssetQuality.LOW) for _ in range(10)]
        assert all(s == Signal.BAD for s in signals)

        # MEDIUM should give NEUTRAL
        signals = [strategy.get_signal(AssetQuality.MEDIUM) for _ in range(10)]
        assert all(s == Signal.NEUTRAL for s in signals)

        # HIGH should give GOOD
        signals = [strategy.get_signal(AssetQuality.HIGH) for _ in range(10)]
        assert all(s == Signal.GOOD for s in signals)

    def test_informativeness_score_perfect(self):
        """Test informativeness score for perfect information"""
        strategy = GarblingStrategy(matrix=np.eye(3), name="Perfect")
        score = strategy.informativeness_score()

        # Perfect information should score 1.0
        assert np.isclose(score, 1.0, atol=0.01)

    def test_informativeness_score_noise(self):
        """Test informativeness score for pure noise"""
        strategy = GarblingStrategy(matrix=np.ones((3, 3)) / 3, name="Noise")
        score = strategy.informativeness_score()

        # Pure noise should score 0.0
        assert np.isclose(score, 0.0, atol=0.01)

    def test_informativeness_score_partial(self):
        """Test informativeness score for partially informative strategy"""
        strategy = GarblingStrategy(
            matrix=np.array([
                [0.8, 0.15, 0.05],  # Mostly accurate
                [0.15, 0.7, 0.15],
                [0.05, 0.15, 0.8],
            ]),
            name="Slight Noise"
        )
        score = strategy.informativeness_score()

        # Should be between 0 and 1, closer to 1
        assert 0.0 < score < 1.0
        assert score > 0.5


class TestBuiltinStrategies:
    """Test all built-in strategies"""

    def test_all_builtin_strategies_valid(self):
        """Test that all built-in strategies are valid"""
        for name, strategy in BUILTIN_STRATEGIES.items():
            # Should not raise
            assert strategy.matrix.shape == (3, 3)
            assert np.allclose(strategy.matrix.sum(axis=1), 1.0)
            assert np.all(strategy.matrix >= 0)

    def test_full_revelation_properties(self):
        """Test full revelation strategy properties"""
        strategy = BUILTIN_STRATEGIES["full_revelation"]

        # Should be identity matrix
        assert np.allclose(strategy.matrix, np.eye(3))

        # Should have informativeness of 1.0
        assert np.isclose(strategy.informativeness_score(), 1.0, atol=0.01)

    def test_complete_noise_properties(self):
        """Test complete noise strategy properties"""
        strategy = BUILTIN_STRATEGIES["complete_noise"]

        # Should be uniform matrix
        assert np.allclose(strategy.matrix, np.ones((3, 3)) / 3)

        # Should have informativeness of 0.0
        assert np.isclose(strategy.informativeness_score(), 0.0, atol=0.01)

    def test_pooling_strategies_pool_correctly(self):
        """Test that pooling strategies create indistinguishable signals"""
        # Pool low with medium
        strategy = BUILTIN_STRATEGIES["pool_low_medium"]

        # LOW and MEDIUM should have identical signal distributions
        assert np.allclose(strategy.matrix[0], strategy.matrix[1])

        # HIGH should be distinguishable (all GOOD signal)
        assert strategy.matrix[2, 2] == 1.0  # HIGH → GOOD with prob 1

    def test_informativeness_ordering(self):
        """Test that informativeness follows expected ordering"""
        scores = {
            name: strategy.informativeness_score()
            for name, strategy in BUILTIN_STRATEGIES.items()
        }

        # Full revelation should be most informative
        assert scores["full_revelation"] > scores["slight_noise"]
        assert scores["slight_noise"] > scores["aggressive_pooling"]
        assert scores["aggressive_pooling"] > scores["complete_noise"]

        # Complete noise should be least informative
        assert scores["complete_noise"] < 0.1


class TestStrategyRegistry:
    """Test the strategy registry"""

    def test_registry_has_builtin_strategies(self):
        """Test that registry contains all built-in strategies"""
        for name in BUILTIN_STRATEGIES.keys():
            assert strategy_registry.is_registered(name)
            strategy = strategy_registry.get(name)
            assert isinstance(strategy, GarblingStrategy)

    def test_get_nonexistent_strategy_raises(self):
        """Test that getting non-existent strategy raises KeyError"""
        with pytest.raises(KeyError, match="not found"):
            strategy_registry.get("nonexistent_strategy")

    def test_list_strategies(self):
        """Test listing all strategies"""
        strategies = strategy_registry.list_strategies()

        assert isinstance(strategies, dict)
        assert "full_revelation" in strategies
        assert "complete_noise" in strategies
        assert len(strategies) >= len(BUILTIN_STRATEGIES)

    def test_register_custom_strategy(self):
        """Test registering a custom strategy"""
        # Create a custom strategy
        custom = GarblingStrategy(
            matrix=np.array([
                [0.6, 0.3, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.3, 0.6],
            ]),
            name="Custom Test Strategy"
        )

        # Register it
        strategy_registry.register("custom_test", custom)

        # Should be retrievable
        retrieved = strategy_registry.get("custom_test")
        assert np.allclose(retrieved.matrix, custom.matrix)
        assert retrieved.name == custom.name

        # Clean up
        # Note: Registry doesn't have unregister, but that's OK for tests

    def test_register_duplicate_raises(self):
        """Test that registering duplicate strategy name raises ValueError"""
        custom = GarblingStrategy(matrix=np.eye(3), name="Duplicate")

        strategy_registry.register("duplicate_test", custom)

        with pytest.raises(ValueError, match="already registered"):
            strategy_registry.register("duplicate_test", custom)
