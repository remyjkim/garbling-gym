"""
Example Python configuration with custom garbling strategy.

Demonstrates how to define and register custom strategies.
"""

import numpy as np
from garbling_gym.core.config import GameConfig
from garbling_gym.core.types import AssetQuality
from garbling_gym.core.strategies import GarblingStrategy, strategy_registry


# Define a custom garbling strategy
custom_matrix = np.array([
    [0.7, 0.2, 0.1],  # LOW → mostly BAD, some NEUTRAL
    [0.2, 0.6, 0.2],  # MEDIUM → mostly NEUTRAL
    [0.1, 0.2, 0.7],  # HIGH → mostly GOOD
])

custom_strategy = GarblingStrategy(
    matrix=custom_matrix,
    name="Custom Moderately Informative"
)

# Register the custom strategy so agents can use it
strategy_registry.register("custom_moderate", custom_strategy)

# Create configuration
config = GameConfig(
    num_rounds=30,
    use_llm=False,
    prior={
        AssetQuality.LOW: 0.2,
        AssetQuality.MEDIUM: 0.5,
        AssetQuality.HIGH: 0.3,
    }
)
