# ABOUTME: Built-in garbling strategies for the game
# ABOUTME: Predefined information structures demonstrating various garbling levels

import numpy as np
from .base import GarblingStrategy


# Pre-defined garbling strategies demonstrating different information structures
BUILTIN_STRATEGIES = {
    "full_revelation": GarblingStrategy(
        matrix=np.eye(3),
        name="Full Revelation (Perfect Information)"
    ),

    "complete_noise": GarblingStrategy(
        matrix=np.ones((3, 3)) / 3,
        name="Complete Noise (No Information)"
    ),

    "pool_low_medium": GarblingStrategy(
        matrix=np.array([
            [0.5, 0.5, 0.0],  # LOW → 50% BAD, 50% NEUTRAL
            [0.5, 0.5, 0.0],  # MEDIUM → 50% BAD, 50% NEUTRAL
            [0.0, 0.0, 1.0],  # HIGH → 100% GOOD
        ]),
        name="Pool Low with Medium"
    ),

    "pool_medium_high": GarblingStrategy(
        matrix=np.array([
            [1.0, 0.0, 0.0],  # LOW → 100% BAD
            [0.0, 0.5, 0.5],  # MEDIUM → 50% NEUTRAL, 50% GOOD
            [0.0, 0.5, 0.5],  # HIGH → 50% NEUTRAL, 50% GOOD
        ]),
        name="Pool Medium with High"
    ),

    "slight_noise": GarblingStrategy(
        matrix=np.array([
            [0.8, 0.15, 0.05],
            [0.15, 0.7, 0.15],
            [0.05, 0.15, 0.8],
        ]),
        name="Slight Noise (Mostly Accurate)"
    ),

    "aggressive_pooling": GarblingStrategy(
        matrix=np.array([
            [0.3, 0.4, 0.3],  # LOW → spread across all
            [0.2, 0.3, 0.5],  # MEDIUM → bias toward GOOD
            [0.1, 0.2, 0.7],  # HIGH → mostly GOOD
        ]),
        name="Aggressive Pooling (Optimistic Bias)"
    ),
}
