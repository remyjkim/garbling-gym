# ABOUTME: Base class for garbling strategies
# ABOUTME: Defines interface for signal generation and informativeness measurement

import numpy as np
from dataclasses import dataclass
from ..types import AssetQuality, Signal


@dataclass
class GarblingStrategy:
    """
    A garbling strategy defined by a stochastic matrix Γ.

    In Blackwell's framework, the matrix Γ[i][j] = P(signal=j | quality=i).
    Each row must be a valid probability distribution (sum to 1).

    This represents how information is transformed from true quality to observed signal.
    """
    matrix: np.ndarray  # 3x3 stochastic matrix
    name: str = "Custom"

    def __post_init__(self):
        """Validate that the matrix is a valid stochastic matrix"""
        # Ensure rows sum to 1 (stochastic matrix)
        row_sums = self.matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError(f"Rows must sum to 1.0, got {row_sums}")

        # Ensure all probabilities are non-negative
        if not np.all(self.matrix >= 0):
            raise ValueError("All probabilities must be non-negative")

        # Ensure correct shape
        if self.matrix.shape != (3, 3):
            raise ValueError(f"Matrix must be 3x3, got shape {self.matrix.shape}")

    def get_signal(self, quality: AssetQuality) -> Signal:
        """
        Sample a signal given true quality using the garbling matrix.

        Args:
            quality: The true asset quality

        Returns:
            The observed (potentially garbled) signal
        """
        probs = self.matrix[quality.value]
        signal_idx = np.random.choice(3, p=probs)
        return Signal(signal_idx)

    def informativeness_score(self) -> float:
        """
        Measure of how informative the signal is (0 = pure noise, 1 = perfect).

        Based on Frobenius distance from the identity matrix (perfect information).
        Normalized to [0, 1] range.

        Returns:
            Informativeness score between 0 and 1
        """
        # Perfect info matrix (identity)
        identity = np.eye(3)

        # Distance from perfect information
        dist = np.linalg.norm(self.matrix - identity, 'fro')

        # Maximum distance (uniform random matrix)
        max_dist = np.linalg.norm(np.ones((3, 3)) / 3 - identity, 'fro')

        # Normalize: 1 = perfect info, 0 = no info
        return 1 - (dist / max_dist)
