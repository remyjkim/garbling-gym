"""
Garbling strategies module.

Provides base classes, built-in strategies, and a registry for managing strategies.
"""

from .base import GarblingStrategy
from .builtin import BUILTIN_STRATEGIES
from .registry import StrategyRegistry, strategy_registry

__all__ = [
    "GarblingStrategy",
    "BUILTIN_STRATEGIES",
    "StrategyRegistry",
    "strategy_registry",
]
