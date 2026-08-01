# ABOUTME: Agent factory for creating and managing agents
# ABOUTME: Provides centralized agent instantiation with extensibility

from typing import Dict, Tuple, Type
from .base import Agent
from .sender import SenderAgent
from .receiver import ReceiverAgent


def _configure_strategy(strategy, config) -> None:
    """
    Inject the configured prior and receiver payoffs into a strategy.

    Builds the ``{(action_name, quality_name): receiver_payoff}`` table from
    ``config.payoffs`` and the name-keyed prior from ``config.prior``, then
    calls ``strategy.configure(...)``.  Any exception is swallowed so that a
    strategy that does not support ``configure`` (or a config missing payoffs)
    silently keeps its defaults — preserving backward compatibility.
    """
    try:
        from ..types import Action, AssetQuality
        prior = {q.name: p for q, p in config.prior.items()}
        payoffs: Dict[Tuple[str, str], float] = {}
        for action in Action:
            for quality in AssetQuality:
                _sender, receiver = config.payoffs.get_payoffs(action, quality)
                payoffs[(action.name, quality.name)] = receiver
        strategy.configure(prior=prior, receiver_payoffs=payoffs)
    except Exception:
        pass


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

    def create_receiver(
        self,
        model: str = "gpt-4o-mini",
        strategy_name: str = "heuristic",
        config: "object | None" = None,
    ) -> ReceiverAgent:
        """
        Create a receiver agent.

        Args:
            model: LLM model to use
            strategy_name: Name of receiver learning strategy to use
            config: Optional ``GameConfig``; when provided, its prior and
                receiver payoffs are injected into the strategy via
                ``configure()`` so the strategy uses the configured game
                parameters rather than module defaults.

        Returns:
            ReceiverAgent instance
        """
        from .strategies.registry import receiver_strategy_registry
        strategy = receiver_strategy_registry.get(strategy_name)
        if config is not None:
            _configure_strategy(strategy, config)
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
