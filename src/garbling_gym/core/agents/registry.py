# ABOUTME: Agent factory for creating and managing agents
# ABOUTME: Provides centralized agent instantiation with extensibility

from typing import Dict, Type
from .base import Agent
from .sender import SenderAgent
from .receiver import ReceiverAgent


class AgentFactory:
    """
    Factory for creating game agents.

    Provides centralized agent creation with support for custom agent types.
    """

    def __init__(self):
        """Initialize factory with built-in agent types"""
        self._agent_types: Dict[str, Type[Agent]] = {
            "sender": SenderAgent,
            "receiver": ReceiverAgent,
        }

    def register_agent_type(self, name: str, agent_class: Type[Agent]) -> None:
        """
        Register a custom agent type.

        Args:
            name: Unique name for the agent type
            agent_class: Agent class (must inherit from Agent)

        Raises:
            ValueError: If agent type already exists or class doesn't inherit from Agent
        """
        if name in self._agent_types:
            raise ValueError(f"Agent type '{name}' is already registered")

        if not issubclass(agent_class, Agent):
            raise ValueError(f"Agent class must inherit from Agent")

        self._agent_types[name] = agent_class

    def create_sender(self, model: str = "gpt-4o-mini") -> SenderAgent:
        """
        Create a sender agent.

        Args:
            model: LLM model to use

        Returns:
            SenderAgent instance
        """
        return SenderAgent(model=model)

    def create_receiver(self, model: str = "gpt-4o-mini", strategy_name: str = "heuristic") -> ReceiverAgent:
        """
        Create a receiver agent.

        Args:
            model: LLM model to use
            strategy_name: Name of receiver learning strategy to use

        Returns:
            ReceiverAgent instance
        """
        from .strategies.registry import receiver_strategy_registry
        strategy = receiver_strategy_registry.get(strategy_name)
        return ReceiverAgent(model=model, strategy=strategy)

    def create_agent(self, agent_type: str, **kwargs) -> Agent:
        """
        Create an agent of specified type.

        Args:
            agent_type: Type of agent to create (e.g., "sender", "receiver")
            **kwargs: Additional arguments to pass to agent constructor

        Returns:
            Agent instance

        Raises:
            KeyError: If agent type not found
        """
        if agent_type not in self._agent_types:
            available = ", ".join(self._agent_types.keys())
            raise KeyError(
                f"Agent type '{agent_type}' not found. "
                f"Available types: {available}"
            )

        agent_class = self._agent_types[agent_type]
        return agent_class(**kwargs)


# Global factory instance
agent_factory = AgentFactory()
