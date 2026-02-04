# ABOUTME: Base class for LLM-powered agents using pydantic-ai
# ABOUTME: Provides typed, validated agent interactions with fallback to heuristic behavior

import os
from typing import Optional
from pydantic import BaseModel
from .base import Agent


class LLMAgent(Agent):
    """
    Base class for LLM-powered game agents using pydantic-ai.

    Falls back to sophisticated heuristic agents when API unavailable.
    The heuristic agents implement rational Bayesian reasoning to demonstrate
    the economic principles even without LLM calls.
    """

    def __init__(self, role: str, model: str = "gpt-4o-mini"):
        """
        Initialize LLM agent.

        Args:
            role: Agent role identifier
            model: Model identifier (e.g., "gpt-4o-mini", "openai:gpt-4o-mini")
        """
        super().__init__(role)
        self.model_name = model
        self.use_api = False
        self.pydantic_agent = None

        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")

        if api_key:
            try:
                # Import pydantic_ai
                from pydantic_ai import Agent as PydanticAgent

                # Normalize model name for pydantic-ai
                if not model.startswith("openai:"):
                    model = f"openai:{model}"

                self.use_api = True
                print(f"  [{role}] Using pydantic-ai with {model}")
            except ImportError:
                print(f"  [{role}] pydantic-ai not available, using heuristic agent")
            except Exception as e:
                print(f"  [{role}] AI agent init failed: {e}, using heuristic agent")
        else:
            print(f"  [{role}] No API key, using heuristic agent (demonstrates same economics)")

    async def _call_llm_async(
        self,
        system_prompt: str,
        user_prompt: str,
        result_type: type[BaseModel]
    ) -> BaseModel:
        """
        Make an async API call using pydantic-ai.

        Args:
            system_prompt: System message defining agent behavior
            user_prompt: User message with current situation
            result_type: Pydantic model type for structured output

        Returns:
            Validated result matching result_type
        """
        if not self.use_api:
            # Return fallback - subclasses handle this
            raise NotImplementedError("Fallback required")

        try:
            from pydantic_ai import Agent as PydanticAgent

            # Normalize model name
            model = self.model_name
            if not model.startswith("openai:"):
                model = f"openai:{model}"

            # Create agent with system prompt and output type
            agent = PydanticAgent(
                model,
                system_prompt=system_prompt,
                output_type=result_type,
            )

            # Run agent and get structured result
            result = await agent.run(user_prompt)
            return result.output

        except Exception as e:
            print(f"API Error: {e}")
            raise

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        result_type: type[BaseModel]
    ) -> BaseModel:
        """
        Synchronous wrapper for _call_llm_async.

        Args:
            system_prompt: System message defining agent behavior
            user_prompt: User message with current situation
            result_type: Pydantic model type for structured output

        Returns:
            Validated result matching result_type
        """
        import asyncio

        try:
            # Try to get running loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop running, create one
            return asyncio.run(self._call_llm_async(system_prompt, user_prompt, result_type))

        # Loop is running, use run_in_executor for sync context
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                self._call_llm_async(system_prompt, user_prompt, result_type)
            )
            return future.result()

    def _fallback_response(self, prompt: str) -> str:
        """
        Fallback for when API is not available.

        Must be implemented by subclasses.

        Args:
            prompt: The prompt that would have been sent to the API

        Returns:
            Heuristic response
        """
        raise NotImplementedError("Subclasses must implement _fallback_response")
