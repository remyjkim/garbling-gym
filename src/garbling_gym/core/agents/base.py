# ABOUTME: Base class for game agents
# ABOUTME: Defines interface that all agents must implement

from abc import ABC, abstractmethod
from typing import List, Dict
from ..types import AssetQuality, Signal, Action


class Agent(ABC):
    """
    Abstract base class for game agents.

    All agents must implement decision-making methods that define their behavior.
    """

    def __init__(self, role: str):
        """
        Initialize agent with a role identifier.

        Args:
            role: String identifying the agent's role (e.g., "Sender", "Receiver")
        """
        self.role = role

    @abstractmethod
    def decide(self, observation, history: List[Dict], round_num: int, total_rounds: int):
        """
        Make a decision based on observation and game history.

        Args:
            observation: The agent's observation (quality for sender, signal for receiver)
            history: List of past rounds
            round_num: Current round number
            total_rounds: Total number of rounds in the game

        Returns:
            Decision made by the agent
        """
        pass
