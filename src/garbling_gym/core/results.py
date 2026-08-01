# ABOUTME: Result dataclasses for game outcomes
# ABOUTME: Structures for round results and complete game summaries

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RoundResult:
    """Result from a single round of the game"""
    round: int
    quality: str
    strategy: str
    garbling_info: float
    signal: str
    action: str
    sender_payoff: float
    receiver_payoff: float


@dataclass
class GameResults:
    """Complete results from a game execution"""
    total_rounds: int
    sender_total: float
    receiver_total: float
    strategies_used: Dict[str, int]
    buy_rate: float
    quality_stats: Dict[str, Dict]
    avg_informativeness: float
    receiver_regret: float
    perfect_info_benchmark: float
    history: List[RoundResult] = field(default_factory=list)
    # Theory benchmarks (Proposal 08): the solver's computed ground-truth ladder
    # for this game's (prior, payoffs). None if the solver could not run.
    benchmarks: Optional[Dict[str, Any]] = None
    # Realized empirical channel estimated from this run's history, plus its
    # information-theoretic summary. None if not computed.
    realized: Optional[Dict[str, Any]] = None
