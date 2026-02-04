"""
Core game engine for garbling economics simulations.

Provides game logic, agents, strategies, and configuration.
"""

from .types import AssetQuality, Signal, Action
from .config import GameConfig
from .payoffs import PayoffStructure
from .game import Game, GameState
from .results import RoundResult, GameResults

# Import submodules for convenient access
from . import agents
from . import strategies

__all__ = [
    "AssetQuality",
    "Signal",
    "Action",
    "GameConfig",
    "PayoffStructure",
    "Game",
    "GameState",
    "RoundResult",
    "GameResults",
    "agents",
    "strategies",
]
