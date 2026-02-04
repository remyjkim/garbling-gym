# ABOUTME: Sender agent implementation
# ABOUTME: Knows true quality and chooses garbling strategies to maximize payoffs

import random
import numpy as np
from typing import List, Dict, Literal
from pydantic import BaseModel, Field
from .llm import LLMAgent
from ..types import AssetQuality
from ..strategies.builtin import BUILTIN_STRATEGIES


class StrategyChoice(BaseModel):
    """Structured output for sender's strategy choice"""
    strategy: Literal[
        "full_revelation",
        "complete_noise",
        "pool_low_medium",
        "pool_medium_high",
        "slight_noise",
        "aggressive_pooling"
    ] = Field(description="The garbling strategy to use this round")


class SenderAgent(LLMAgent):
    """
    The Sender (Information Holder) Agent.

    Knows the true asset quality and must choose a garbling strategy.
    Goal: Maximize sales (get receiver to BUY) while considering long-term trust.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("Sender", model)
        self.system_prompt = """You are the SENDER in an information economics game.

BACKGROUND - GARBLING IN INFORMATION ECONOMICS:
Garbling is a key concept from Blackwell's (1951) theory of information. When you "garble"
information, you add noise to the signal, making it less informative. A garbling matrix Γ
transforms true states into observable signals probabilistically.

YOUR ROLE:
- You know the TRUE quality of an asset (LOW, MEDIUM, or HIGH)
- You must choose a GARBLING STRATEGY that determines what signal the buyer sees
- Your goal is to maximize your expected payoff (you earn when they BUY)

AVAILABLE STRATEGIES:
1. full_revelation - Tell truth (Signal perfectly reveals quality)
2. complete_noise - Pure noise (Signal is random, uninformative)
3. pool_low_medium - Combine LOW and MEDIUM into ambiguous signals
4. pool_medium_high - Combine MEDIUM and HIGH into ambiguous signals
5. slight_noise - Mostly truthful with small errors
6. aggressive_pooling - Bias all signals toward GOOD

ECONOMIC INSIGHT:
- Full revelation: Receiver only buys MEDIUM/HIGH → lower expected sales
- Pooling: Can get receiver to buy LOW sometimes by mixing with better qualities
- But too much deception may cause receiver to PASS on everything

PAYOFF STRUCTURE:
- If receiver BUYS: You get +10 regardless of quality
- If receiver PASSES: You get 0

Think strategically about the trade-off between short-term gains and signal credibility."""

    def choose_strategy(
        self,
        true_quality: AssetQuality,
        game_history: List[Dict],
        round_num: int,
        total_rounds: int
    ) -> str:
        """
        Choose a garbling strategy based on game state.

        Args:
            true_quality: The true asset quality known to sender
            game_history: List of past rounds
            round_num: Current round number
            total_rounds: Total rounds in game

        Returns:
            Strategy name to use
        """
        history_summary = self._summarize_history(game_history)

        user_prompt = f"""Round {round_num}/{total_rounds}

TRUE ASSET QUALITY: {true_quality.name}

GAME HISTORY:
{history_summary}

Based on the true quality and game history, choose a garbling strategy.
Consider:
1. If quality is HIGH, full revelation might be best
2. If quality is LOW, pooling strategies can help
3. If receiver has been burned before, they may be more skeptical

Respond with ONLY the strategy name from:
- full_revelation
- complete_noise
- pool_low_medium
- pool_medium_high
- slight_noise
- aggressive_pooling

Your choice:"""

        # Try to use LLM with structured output
        if self.use_api:
            try:
                result = self._call_llm(self.system_prompt, user_prompt, StrategyChoice)
                return result.strategy
            except Exception as e:
                print(f"  [{self.role}] LLM failed, using fallback: {e}")
                return self._parse_strategy(self._fallback_response(user_prompt))
        else:
            return self._parse_strategy(self._fallback_response(user_prompt))

    def decide(self, observation, history: List[Dict], round_num: int, total_rounds: int):
        """Implement abstract decide method by delegating to choose_strategy"""
        return self.choose_strategy(observation, history, round_num, total_rounds)

    def _summarize_history(self, history: List[Dict]) -> str:
        """Create a summary of recent game history"""
        if not history:
            return "No previous rounds."

        summary_lines = []
        for h in history[-5:]:  # Last 5 rounds
            summary_lines.append(
                f"  Round {h['round']}: Quality={h['quality']}, "
                f"Strategy={h['strategy']}, Signal={h['signal']}, "
                f"Action={h['action']}, Receiver_Payoff={h['receiver_payoff']}"
            )
        return "\n".join(summary_lines)

    def _parse_strategy(self, response: str) -> str:
        """Extract strategy name from LLM response"""
        response_lower = response.lower().strip()
        for strategy in BUILTIN_STRATEGIES:
            if strategy in response_lower:
                return strategy
        # Default fallback
        return "slight_noise"

    def _fallback_response(self, prompt: str) -> str:
        """
        Sophisticated heuristic strategy when API unavailable.
        Implements rational strategic reasoning based on information economics.
        """
        # Parse the quality from the prompt
        quality = None
        for q in ["LOW", "MEDIUM", "HIGH"]:
            if f"Quality: {q}" in prompt or f"QUALITY: {q}" in prompt:
                quality = q
                break

        # Parse history metrics if available
        history_lines = prompt.split('\n')
        recent_receiver_payoffs = []
        for line in history_lines:
            if 'Receiver_Payoff=' in line:
                try:
                    payoff = float(line.split('Receiver_Payoff=')[-1].split()[0])
                    recent_receiver_payoffs.append(payoff)
                except:
                    pass

        # Calculate trust level (how skeptical is receiver?)
        receiver_avg = np.mean(recent_receiver_payoffs) if recent_receiver_payoffs else 0

        # Strategic reasoning based on quality and trust
        if quality == "HIGH":
            # HIGH quality: Full revelation is often best - receiver will buy
            # Occasionally use slight_noise to maintain uncertainty about our strategy
            return random.choices(
                ["full_revelation", "slight_noise"],
                weights=[0.7, 0.3]
            )[0]

        elif quality == "MEDIUM":
            # MEDIUM quality: Flexible strategy
            # If receiver has been profitable, they're likely buying - can reveal
            # If receiver is skeptical, try pooling with HIGH
            if receiver_avg > 0:
                return random.choices(
                    ["full_revelation", "slight_noise", "pool_medium_high"],
                    weights=[0.4, 0.3, 0.3]
                )[0]
            else:
                return random.choices(
                    ["pool_medium_high", "slight_noise", "aggressive_pooling"],
                    weights=[0.4, 0.3, 0.3]
                )[0]

        else:  # LOW quality
            # LOW quality: This is where garbling matters most!
            # Try to pool with better qualities to induce buying
            # The economic insight: sender gains from reducing informativeness
            if receiver_avg < -3:  # Receiver very unhappy - may pass on everything
                # Try less aggressive tactics
                return random.choices(
                    ["pool_low_medium", "complete_noise", "slight_noise"],
                    weights=[0.4, 0.3, 0.3]
                )[0]
            else:
                # Receiver still willing to buy - use more aggressive pooling
                return random.choices(
                    ["pool_low_medium", "aggressive_pooling", "pool_medium_high"],
                    weights=[0.4, 0.35, 0.25]
                )[0]
