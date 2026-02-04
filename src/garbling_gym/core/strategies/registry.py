# ABOUTME: Strategy registry for managing garbling strategies
# ABOUTME: Allows registration and retrieval of both built-in and custom strategies

from typing import Dict
from .base import GarblingStrategy
from .builtin import BUILTIN_STRATEGIES


class StrategyRegistry:
    """
    Registry for garbling strategies.

    Manages both built-in and custom strategies, allowing for extensibility.
    """

    def __init__(self):
        """Initialize registry with built-in strategies"""
        self._strategies: Dict[str, GarblingStrategy] = {}

        # Register all built-in strategies
        for name, strategy in BUILTIN_STRATEGIES.items():
            self.register(name, strategy)

    def register(self, name: str, strategy: GarblingStrategy) -> None:
        """
        Register a strategy in the registry.

        Args:
            name: Unique name for the strategy
            strategy: GarblingStrategy instance

        Raises:
            ValueError: If strategy name already exists
        """
        if name in self._strategies:
            raise ValueError(f"Strategy '{name}' is already registered")

        self._strategies[name] = strategy

    def get(self, name: str) -> GarblingStrategy:
        """
        Retrieve a strategy by name.

        Args:
            name: Strategy name

        Returns:
            GarblingStrategy instance

        Raises:
            KeyError: If strategy not found
        """
        if name not in self._strategies:
            available = ", ".join(self._strategies.keys())
            raise KeyError(
                f"Strategy '{name}' not found. "
                f"Available strategies: {available}"
            )

        return self._strategies[name]

    def list_strategies(self) -> Dict[str, str]:
        """
        List all registered strategies.

        Returns:
            Dictionary mapping strategy names to their display names
        """
        return {name: strategy.name for name, strategy in self._strategies.items()}

    def is_registered(self, name: str) -> bool:
        """Check if a strategy is registered"""
        return name in self._strategies


# Global registry instance
strategy_registry = StrategyRegistry()
