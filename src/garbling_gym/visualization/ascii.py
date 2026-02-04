# ABOUTME: ASCII visualization functions for garbling economics game
# ABOUTME: Creates charts, analysis visualizations, and Bayesian updating displays

import json
import numpy as np
from typing import Dict, List

def create_ascii_bar(value: float, max_value: float, width: int = 40) -> str:
    """Create an ASCII bar chart element"""
    if max_value == 0:
        return ""
    filled = int((abs(value) / max_value) * width)
    if value >= 0:
        return "█" * filled + "░" * (width - filled)
    else:
        return "▓" * filled + "░" * (width - filled) + " (negative)"

def visualize_results(results: Dict) -> str:
    """Generate ASCII visualization of game results"""
    output = []
    
    output.append("\n" + "=" * 70)
    output.append("📊 VISUAL ANALYSIS OF GARBLING ECONOMICS")
    output.append("=" * 70)
    
    # 1. Strategy distribution
    output.append("\n┌─────────────────────────────────────────────────────────────────────┐")
    output.append("│ GARBLING STRATEGY DISTRIBUTION                                      │")
    output.append("├─────────────────────────────────────────────────────────────────────┤")
    
    strategies = results.get('strategies_used', {})
    total_rounds = results.get('total_rounds', 1)
    max_count = max(strategies.values()) if strategies else 1
    
    for strategy, count in sorted(strategies.items(), key=lambda x: -x[1]):
        pct = count / total_rounds * 100
        bar = create_ascii_bar(count, max_count, 30)
        output.append(f"│ {strategy:24s} │ {bar} {count:2d} ({pct:4.1f}%) │")
    
    output.append("└─────────────────────────────────────────────────────────────────────┘")
    
    # 2. Payoff timeline
    output.append("\n┌─────────────────────────────────────────────────────────────────────┐")
    output.append("│ CUMULATIVE PAYOFFS OVER TIME                                        │")
    output.append("├─────────────────────────────────────────────────────────────────────┤")
    
    history = results.get('history', [])
    sender_cumulative = 0
    receiver_cumulative = 0
    
    # Show every 5 rounds
    for i, h in enumerate(history):
        sender_cumulative += h['sender_payoff']
        receiver_cumulative += h['receiver_payoff']
        if (i + 1) % 5 == 0 or i == len(history) - 1:
            output.append(f"│ Round {i+1:2d}: Sender={sender_cumulative:+4.0f}  Receiver={receiver_cumulative:+4.0f}  │")
    
    output.append("└─────────────────────────────────────────────────────────────────────┘")
    
    # 3. Information quality vs outcomes
    output.append("\n┌─────────────────────────────────────────────────────────────────────┐")
    output.append("│ INFORMATION QUALITY IMPACT                                          │")
    output.append("├─────────────────────────────────────────────────────────────────────┤")
    
    # Group by informativeness level
    low_info = [h for h in history if h['garbling_info'] < 0.3]
    med_info = [h for h in history if 0.3 <= h['garbling_info'] < 0.7]
    high_info = [h for h in history if h['garbling_info'] >= 0.7]
    
    for label, group in [("Low (Garbled)", low_info), ("Medium", med_info), ("High (Clear)", high_info)]:
        if group:
            avg_recv = np.mean([h['receiver_payoff'] for h in group])
            buy_rate = sum(1 for h in group if h['action'] == 'BUY') / len(group)
            output.append(f"│ {label:14s}: Avg Receiver Payoff={avg_recv:+5.1f}, Buy Rate={buy_rate:.1%}     │")
    
    output.append("└─────────────────────────────────────────────────────────────────────┘")
    
    # 4. Blackwell ordering demonstration
    output.append("\n┌─────────────────────────────────────────────────────────────────────┐")
    output.append("│ BLACKWELL ORDERING ANALYSIS                                         │")
    output.append("├─────────────────────────────────────────────────────────────────────┤")
    output.append("│                                                                     │")
    output.append("│  Blackwell's Theorem: σ Blackwell-dominates σ' if σ' = Γσ          │")
    output.append("│  (σ' is a garbled version of σ)                                    │")
    output.append("│                                                                     │")
    output.append("│  Informativeness Ranking (most to least informative):              │")
    output.append("│                                                                     │")
    output.append("│  1.00 ━━━━━━━━━━ Full Revelation (Perfect Information)             │")
    output.append("│    │                                                               │")
    output.append("│  0.64 ━━━━━━━━━━ Slight Noise                                      │")
    output.append("│    │              ↓ (garbling)                                     │")
    output.append("│  0.29 ━━━━━━━━━━ Pool Low/Medium, Pool Medium/High                 │")
    output.append("│    │              ↓ (more garbling)                                │")
    output.append("│  0.09 ━━━━━━━━━━ Aggressive Pooling                                │")
    output.append("│    │              ↓ (maximum garbling)                             │")
    output.append("│  0.00 ━━━━━━━━━━ Complete Noise (No Information)                   │")
    output.append("│                                                                     │")
    output.append(f"│  This game's average: {results.get('avg_informativeness', 0):.2f}                                     │")
    output.append("└─────────────────────────────────────────────────────────────────────┘")
    
    # 5. Economic welfare analysis
    output.append("\n┌─────────────────────────────────────────────────────────────────────┐")
    output.append("│ WELFARE ANALYSIS                                                    │")
    output.append("├─────────────────────────────────────────────────────────────────────┤")
    
    sender_total = results.get('sender_total', 0)
    receiver_total = results.get('receiver_total', 0)
    social_welfare = sender_total + receiver_total
    
    # Calculate perfect info benchmark from history
    history = results.get('history', [])
    perfect_info = sum(
        5 if h['quality'] == 'MEDIUM' else (20 if h['quality'] == 'HIGH' else 0)
        for h in history
    )
    deadweight_loss = perfect_info - receiver_total
    
    output.append(f"│                                                                     │")
    output.append(f"│  Sender Surplus:        {sender_total:+6.0f}                                    │")
    output.append(f"│  Receiver Surplus:      {receiver_total:+6.0f}                                    │")
    output.append(f"│  Total Social Welfare:  {social_welfare:+6.0f}                                    │")
    output.append(f"│                                                                     │")
    output.append(f"│  Perfect Info Benchmark:{perfect_info:+6.0f} (receiver under full revelation)   │")
    output.append(f"│  Receiver Regret:       {deadweight_loss:+6.0f} (loss from garbled information)  │")
    output.append("│                                                                     │")
    output.append("│  Key Insight: Garbling transfers value from receiver to sender     │")
    output.append("│  and can create or destroy value depending on the outcomes.        │")
    output.append("└─────────────────────────────────────────────────────────────────────┘")
    
    return "\n".join(output)


def analyze_bayesian_updating(history: List[Dict]) -> str:
    """Analyze how receiver's beliefs should have evolved"""
    output = []
    
    output.append("\n" + "=" * 70)
    output.append("📈 BAYESIAN UPDATING ANALYSIS")
    output.append("=" * 70)
    
    # Track signal-quality relationships
    signal_quality_matrix = {
        'BAD': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0},
        'NEUTRAL': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0},
        'GOOD': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
    }
    
    for h in history:
        sig = h['signal']
        qual = h['quality']
        signal_quality_matrix[sig][qual] += 1
    
    output.append("\nObserved Signal-Quality Relationships:")
    output.append("(What quality was actually associated with each signal)")
    output.append("")
    output.append("Signal    │   LOW    MEDIUM    HIGH   │ Total")
    output.append("──────────┼──────────────────────────┼──────")
    
    for signal in ['BAD', 'NEUTRAL', 'GOOD']:
        counts = signal_quality_matrix[signal]
        total = sum(counts.values())
        output.append(f"{signal:9s} │ {counts['LOW']:5d}    {counts['MEDIUM']:5d}     {counts['HIGH']:4d}   │  {total:3d}")
    
    output.append("")
    output.append("Implied Posterior Probabilities P(Quality | Signal):")
    output.append("(What a Bayesian receiver should believe after seeing each signal)")
    output.append("")
    
    for signal in ['BAD', 'NEUTRAL', 'GOOD']:
        counts = signal_quality_matrix[signal]
        total = sum(counts.values())
        if total > 0:
            probs = {q: c/total for q, c in counts.items()}
            expected_value = probs['LOW'] * (-15) + probs['MEDIUM'] * 5 + probs['HIGH'] * 20
            optimal_action = "BUY" if expected_value > 0 else "PASS"
            output.append(f"Given signal {signal}:")
            output.append(f"  P(LOW)={probs['LOW']:.2f}, P(MED)={probs['MEDIUM']:.2f}, P(HIGH)={probs['HIGH']:.2f}")
            output.append(f"  E[BUY] = {probs['LOW']:.2f}×(-15) + {probs['MEDIUM']:.2f}×(5) + {probs['HIGH']:.2f}×(20) = {expected_value:+.1f}")
            output.append(f"  Optimal action: {optimal_action}")
            output.append("")
    
    return "\n".join(output)
