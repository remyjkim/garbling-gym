# ABOUTME: Main game orchestration and state management
# ABOUTME: Coordinates agents, strategies, and payoff computation

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .types import AssetQuality, Signal, Action
from .config import GameConfig
from .results import RoundResult, GameResults


@dataclass
class GameState:
    """Tracks the state of the game"""
    round_num: int = 0
    sender_total: float = 0.0
    receiver_total: float = 0.0
    history: List[Dict] = field(default_factory=list)


class Game:
    """
    Base game class for garbling economics simulations.
    Extensible for game variants.
    """

    def __init__(self, config: GameConfig, sender=None, receiver=None):
        """
        Initialize game with configuration and agents.

        Args:
            config: Game configuration
            sender: Sender agent (will be set up in agent system)
            receiver: Receiver agent (will be set up in agent system)
        """
        self.config = config
        self.state = GameState()
        self.sender = sender
        self.receiver = receiver

    def sample_quality(self) -> AssetQuality:
        """Sample true asset quality from prior distribution"""
        qualities = list(self.config.prior.keys())
        probs = list(self.config.prior.values())
        return random.choices(qualities, probs)[0]

    def compute_payoffs(self, action: Action, quality: AssetQuality) -> Tuple[float, float]:
        """Compute (sender_payoff, receiver_payoff)"""
        return self.config.payoffs.get_payoffs(action, quality)

    def play_round(self) -> Dict:
        """
        Play a single round of the game.

        Returns:
            Dictionary with round results
        """
        if self.sender is None or self.receiver is None:
            raise ValueError("Agents must be set before playing")

        self.state.round_num += 1
        round_num = self.state.round_num

        # 1. Nature draws true quality
        true_quality = self.sample_quality()

        # 2. Sender chooses garbling strategy
        strategy_name = self.sender.choose_strategy(
            true_quality,
            self.state.history,
            round_num,
            self.config.num_rounds
        )

        # Import here to avoid circular dependency
        from .strategies.registry import strategy_registry
        garbling_strategy = strategy_registry.get(strategy_name)

        # 3. Signal is generated through garbling strategy
        signal = garbling_strategy.get_signal(true_quality)

        # 4. Receiver observes signal and makes decision
        action = self.receiver.make_decision(
            signal,
            self.state.history,
            round_num,
            self.config.num_rounds
        )

        # 5. Compute payoffs
        sender_payoff, receiver_payoff = self.compute_payoffs(action, true_quality)

        # 6. Let receiver learn from the revealed outcome
        self.receiver.learn(
            signal=signal,
            action=action,
            true_quality=true_quality,
            sender_payoff=sender_payoff,
            receiver_payoff=receiver_payoff,
        )

        # 7. Update state
        self.state.sender_total += sender_payoff
        self.state.receiver_total += receiver_payoff

        # 8. Record round
        round_record = {
            'round': round_num,
            'quality': true_quality.name,
            'strategy': strategy_name,
            'garbling_info': garbling_strategy.informativeness_score(),
            'signal': signal.name,
            'action': action.name,
            'sender_payoff': sender_payoff,
            'receiver_payoff': receiver_payoff,
        }
        self.state.history.append(round_record)

        return round_record

    def play_game(self, verbose: bool = True) -> GameResults:
        """
        Play the full game and return results.

        Args:
            verbose: If True, print progress

        Returns:
            GameResults object with complete game summary
        """
        if verbose:
            print("=" * 70)
            print("GARBLING ECONOMICS GAME")
            print("Demonstrating: Blackwell Informativeness & Strategic Information Design")
            print("=" * 70)
            print(f"\nRounds: {self.config.num_rounds}")
            print()

        for _ in range(self.config.num_rounds):
            round_result = self.play_round()

            if verbose:
                self._print_round(round_result)

            # Small delay if using LLM API
            if self.config.use_llm:
                time.sleep(0.5)

        # Compute summary statistics
        summary = self._compute_summary()

        if verbose:
            self._print_summary(summary)

        return summary

    def _print_round(self, r: Dict):
        """Print round results"""
        print(f"Round {r['round']:2d} | "
              f"Quality: {r['quality']:6s} | "
              f"Strategy: {r['strategy']:20s} | "
              f"Signal: {r['signal']:7s} | "
              f"Action: {r['action']:4s} | "
              f"Payoffs (S/R): {r['sender_payoff']:+3.0f}/{r['receiver_payoff']:+3.0f}")

    def _compute_summary(self) -> GameResults:
        """Compute game summary statistics"""
        history = self.state.history

        # Strategy usage
        strategies_used = {}
        for h in history:
            s = h['strategy']
            strategies_used[s] = strategies_used.get(s, 0) + 1

        # Action distribution
        buys = sum(1 for h in history if h['action'] == 'BUY')
        passes = len(history) - buys

        # Payoff analysis by quality
        quality_stats = {q.name: {'count': 0, 'bought': 0, 'sender_total': 0, 'receiver_total': 0}
                        for q in AssetQuality}
        for h in history:
            q = h['quality']
            quality_stats[q]['count'] += 1
            if h['action'] == 'BUY':
                quality_stats[q]['bought'] += 1
            quality_stats[q]['sender_total'] += h['sender_payoff']
            quality_stats[q]['receiver_total'] += h['receiver_payoff']

        # Information quality analysis
        avg_informativeness = np.mean([h['garbling_info'] for h in history])

        # Calculate regret (vs. perfect information benchmark)
        perfect_info_receiver = 0
        for h in history:
            quality = AssetQuality[h['quality']]
            if quality == AssetQuality.MEDIUM:
                perfect_info_receiver += 5
            elif quality == AssetQuality.HIGH:
                perfect_info_receiver += 20

        actual_receiver = self.state.receiver_total
        receiver_regret = perfect_info_receiver - actual_receiver

        return GameResults(
            total_rounds=len(history),
            sender_total=self.state.sender_total,
            receiver_total=self.state.receiver_total,
            strategies_used=strategies_used,
            buy_rate=buys / len(history) if history else 0,
            quality_stats=quality_stats,
            avg_informativeness=avg_informativeness,
            receiver_regret=receiver_regret,
            perfect_info_benchmark=perfect_info_receiver,
            history=history,
        )

    def _print_summary(self, summary: GameResults):
        """Print game summary"""
        print("\n" + "=" * 70)
        print("GAME SUMMARY - ECONOMICS OF GARBLING")
        print("=" * 70)

        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Sender Total Payoff:   {summary.sender_total:+.0f}")
        print(f"   Receiver Total Payoff: {summary.receiver_total:+.0f}")
        print(f"   Buy Rate:              {summary.buy_rate:.1%}")

        print(f"\n📉 INFORMATION QUALITY:")
        print(f"   Avg Informativeness:   {summary.avg_informativeness:.2f} (0=noise, 1=perfect)")
        print(f"   Receiver Regret:       {summary.receiver_regret:+.0f}")
        print(f"   (vs. perfect info:     {summary.perfect_info_benchmark:.0f})")

        print(f"\n🎯 STRATEGY USAGE:")
        for strategy, count in sorted(summary.strategies_used.items(),
                                     key=lambda x: -x[1]):
            pct = count / summary.total_rounds * 100
            print(f"   {strategy:25s}: {count:2d} ({pct:4.1f}%)")

        print(f"\n📈 BY QUALITY:")
        for q_name, stats in summary.quality_stats.items():
            if stats['count'] > 0:
                buy_rate = stats['bought'] / stats['count'] * 100
                print(f"   {q_name:6s}: {stats['count']:2d} occurrences, "
                      f"{buy_rate:5.1f}% bought, "
                      f"Sender={stats['sender_total']:+4.0f}, "
                      f"Receiver={stats['receiver_total']:+4.0f}")

        print("\n" + "=" * 70)
