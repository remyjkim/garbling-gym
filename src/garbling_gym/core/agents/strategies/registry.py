# ABOUTME: Registry mapping names to receiver strategy classes
# ABOUTME: Allows strategies to be selected by name from config or CLI

from typing import Dict, List, Type

from . import ReceiverStrategy
from .legacy_heuristic import LegacyHeuristicStrategy
from .bayesian import DirichletBayesianStrategy


class ReceiverStrategyRegistry:
    """
    Maps strategy names to their classes, instantiating on demand.

    Each call to get() returns a fresh instance so strategies start with
    clean state for every game.
    """

    def __init__(self) -> None:
        self._classes: Dict[str, Type[ReceiverStrategy]] = {}
        self.register("heuristic", LegacyHeuristicStrategy)
        self.register("bayesian", DirichletBayesianStrategy)

    def register(self, name: str, strategy_class: Type[ReceiverStrategy]) -> None:
        """Register a strategy class under a given name."""
        if name in self._classes:
            raise ValueError(f"Strategy '{name}' is already registered")
        self._classes[name] = strategy_class

    def get(self, name: str) -> ReceiverStrategy:
        """Return a fresh instance of the named strategy."""
        if name not in self._classes:
            available = ", ".join(self._classes)
            raise KeyError(f"Strategy '{name}' not found. Available: {available}")
        return self._classes[name]()

    def is_registered(self, name: str) -> bool:
        return name in self._classes

    def list_strategies(self) -> List[str]:
        return list(self._classes)


# Global instance used by game setup and CLI
receiver_strategy_registry = ReceiverStrategyRegistry()
