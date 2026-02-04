# ABOUTME: Receiver agent implementation
# ABOUTME: Observes signals and makes BUY/PASS decisions using Bayesian reasoning

import random
import numpy as np
from typing import List, Dict, Tuple, Literal
from pydantic import BaseModel, Field
from .llm import LLMAgent
from ..types import Signal, Action


class ActionChoice(BaseModel):
    """Structured output for receiver's action choice"""
    action: Literal["BUY", "PASS"] = Field(
        description="The action to take: BUY the asset or PASS on it"
    )


class ReceiverAgent(LLMAgent):
    """
    The Receiver (Buyer/Decision Maker) Agent.

    Observes a (potentially garbled) signal and must decide to BUY or PASS.
    Goal: Maximize expected utility by correctly inferring quality from signals.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("Receiver", model)
        self.system_prompt = """You are the RECEIVER in an information economics game.

BACKGROUND - VALUE OF INFORMATION:
In information economics, the value of information depends on how it affects decisions.
When information is "garbled" (made noisy), your ability to make good decisions decreases.
Blackwell's theorem says more informative signals are always better for decision-makers.

YOUR ROLE:
- You observe a SIGNAL about an asset's quality (BAD, NEUTRAL, or GOOD)
- The signal may be GARBLED (noisy/misleading) by the sender
- You must decide: BUY or PASS

YOUR PAYOFFS:
- BUY + LOW quality: -15 (big loss!)
- BUY + MEDIUM quality: +5 (small gain)
- BUY + HIGH quality: +20 (big gain!)
- PASS: 0 (safe but no upside)

PRIOR BELIEFS (without any signal):
- P(LOW) = 30%
- P(MEDIUM) = 40%
- P(HIGH) = 30%

BAYESIAN REASONING:
Use the signal and history to update your beliefs about quality.
- If sender has been mostly honest, GOOD signal → likely HIGH quality
- If sender has been deceptive, any signal is less informative
- A rational sender with LOW quality has incentive to garble

EXPECTED VALUE CALCULATION:
E[BUY] = P(LOW|signal)×(-15) + P(MEDIUM|signal)×(5) + P(HIGH|signal)×(20)
E[PASS] = 0

Only BUY if E[BUY] > 0."""

    def make_decision(
        self,
        signal: Signal,
        game_history: List[Dict],
        round_num: int,
        total_rounds: int
    ) -> Action:
        """
        Decide whether to BUY or PASS based on signal and history.

        Args:
            signal: Observed signal
            game_history: List of past rounds
            round_num: Current round number
            total_rounds: Total rounds in game

        Returns:
            Action (BUY or PASS)
        """
        history_summary = self._summarize_history(game_history)
        trust_analysis = self._analyze_trust(game_history)

        user_prompt = f"""Round {round_num}/{total_rounds}

OBSERVED SIGNAL: {signal.name}

GAME HISTORY:
{history_summary}

TRUST ANALYSIS:
{trust_analysis}

Based on the signal and history, decide your action.
Consider:
1. What does this signal typically mean given the sender's past behavior?
2. How much can you trust the signal?
3. What is your expected value of buying vs. passing?

Respond with ONLY: BUY or PASS

Your decision:"""

        # Try to use LLM with structured output
        if self.use_api:
            try:
                result = self._call_llm(self.system_prompt, user_prompt, ActionChoice)
                return Action.BUY if result.action == "BUY" else Action.PASS
            except Exception as e:
                print(f"  [{self.role}] LLM failed, using fallback: {e}")
                return self._parse_action(self._fallback_response(user_prompt))
        else:
            return self._parse_action(self._fallback_response(user_prompt))

    def decide(self, observation, history: List[Dict], round_num: int, total_rounds: int):
        """Implement abstract decide method by delegating to make_decision"""
        return self.make_decision(observation, history, round_num, total_rounds)

    def _summarize_history(self, history: List[Dict]) -> str:
        """Create a summary of recent game history"""
        if not history:
            return "No previous rounds - no information about sender's behavior."

        summary_lines = []
        for h in history[-5:]:
            summary_lines.append(
                f"  Round {h['round']}: Signal={h['signal']}, "
                f"You chose {h['action']}, True quality was {h['quality']}, "
                f"Your payoff: {h['receiver_payoff']}"
            )
        return "\n".join(summary_lines)

    def _analyze_trust(self, history: List[Dict]) -> str:
        """Analyze sender's trustworthiness based on history"""
        if not history:
            return "No history to analyze trust."

        # Calculate deception rate
        deceptions = 0
        for h in history:
            signal = h['signal']
            quality = h['quality']
            # Check if signal was misleading
            if (signal == 'GOOD' and quality == 'LOW') or \
               (signal == 'BAD' and quality == 'HIGH'):
                deceptions += 1

        deception_rate = deceptions / len(history)

        # Calculate receiver's historical performance
        receiver_payoffs = [h['receiver_payoff'] for h in history]
        avg_payoff = np.mean(receiver_payoffs)

        return f"""- Apparent deception rate: {deception_rate:.1%}
- Your average payoff: {avg_payoff:.1f}
- Total rounds observed: {len(history)}"""

    def _parse_action(self, response: str) -> Action:
        """Extract action from LLM response"""
        response_upper = response.upper().strip()
        if "BUY" in response_upper:
            return Action.BUY
        return Action.PASS

    def _fallback_response(self, prompt: str) -> str:
        """
        Sophisticated heuristic decision when API unavailable.
        Implements Bayesian updating and expected utility calculation.
        """
        # Parse the signal
        signal = None
        for s in ["BAD", "NEUTRAL", "GOOD"]:
            if f"Signal: {s}" in prompt or f"SIGNAL: {s}" in prompt:
                signal = s
                break

        # Parse history to estimate sender's honesty
        history_lines = prompt.split('\n')
        signal_quality_pairs = []
        recent_payoffs = []

        for line in history_lines:
            if 'Signal=' in line and 'quality was' in line:
                try:
                    sig = line.split('Signal=')[1].split(',')[0].strip()
                    qual = line.split('quality was')[1].split(',')[0].strip()
                    signal_quality_pairs.append((sig, qual))
                except:
                    pass
            if 'payoff:' in line.lower():
                try:
                    payoff = float(line.split(':')[-1].strip())
                    recent_payoffs.append(payoff)
                except:
                    pass

        # Estimate signal reliability using historical data
        reliability = self._estimate_reliability(signal_quality_pairs, signal)

        # Prior probabilities
        prior = {
            'LOW': 0.3,
            'MEDIUM': 0.4,
            'HIGH': 0.3
        }

        # Compute posterior using Bayes rule with reliability estimate
        posterior = self._compute_posterior(signal, reliability, prior)

        # Expected value calculation
        expected_buy = (
            posterior['LOW'] * (-15) +
            posterior['MEDIUM'] * 5 +
            posterior['HIGH'] * 20
        )

        # Add risk aversion factor based on recent experience
        if recent_payoffs:
            avg_payoff = np.mean(recent_payoffs)
            if avg_payoff < -5:
                # Been burned - add risk premium
                expected_buy -= 3
            elif avg_payoff > 5:
                # Doing well - slightly more confident
                expected_buy += 1

        # Decision with some stochasticity (bounded rationality)
        prob_buy = 1 / (1 + np.exp(-(expected_buy - 1)))  # Sigmoid with small threshold

        if random.random() < prob_buy:
            return "BUY"
        return "PASS"

    def _estimate_reliability(
        self,
        history: List[Tuple[str, str]],
        current_signal: str
    ) -> Dict:
        """
        Estimate P(Signal | Quality) from historical data.
        This is the receiver's model of the sender's garbling strategy.
        """
        # Default uninformative prior
        default = {
            'GOOD': {'LOW': 0.2, 'MEDIUM': 0.4, 'HIGH': 0.8},
            'NEUTRAL': {'LOW': 0.3, 'MEDIUM': 0.4, 'HIGH': 0.15},
            'BAD': {'LOW': 0.5, 'MEDIUM': 0.2, 'HIGH': 0.05}
        }

        if len(history) < 3:
            return default.get(current_signal, default['NEUTRAL'])

        # Count signal-quality co-occurrences
        counts = {q: {s: 0 for s in ['BAD', 'NEUTRAL', 'GOOD']} for q in ['LOW', 'MEDIUM', 'HIGH']}
        quality_totals = {q: 0 for q in ['LOW', 'MEDIUM', 'HIGH']}

        for sig, qual in history:
            if qual in counts and sig in counts[qual]:
                counts[qual][sig] += 1
                quality_totals[qual] += 1

        # Estimate P(current_signal | quality) with Laplace smoothing
        estimated = {}
        for qual in ['LOW', 'MEDIUM', 'HIGH']:
            if quality_totals[qual] > 0:
                estimated[qual] = (counts[qual].get(current_signal, 0) + 1) / (quality_totals[qual] + 3)
            else:
                estimated[qual] = default.get(current_signal, default['NEUTRAL']).get(qual, 0.33)

        return estimated

    def _compute_posterior(self, signal: str, likelihood: Dict, prior: Dict) -> Dict:
        """
        Compute P(Quality | Signal) using Bayes' rule.
        """
        # Compute unnormalized posterior
        unnormalized = {}
        for qual in ['LOW', 'MEDIUM', 'HIGH']:
            unnormalized[qual] = likelihood.get(qual, 0.33) * prior[qual]

        # Normalize
        total = sum(unnormalized.values())
        if total == 0:
            return prior

        return {q: v / total for q, v in unnormalized.items()}
