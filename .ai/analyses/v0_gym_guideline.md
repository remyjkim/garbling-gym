# Garbling Economics: An Adversarial LLM Game

## What is Garbling in Information Economics?

**Garbling** is a fundamental concept in information economics that describes the process of adding "noise" to information, making it less informative. The term comes from the seminal work of David Blackwell (1951, 1953) on comparing information structures.

### Formal Definition

An information structure (or "experiment") σ maps true states of the world Ω to probability distributions over signals S:

```
σ: Ω → Δ(S)
```

Experiment σ is **garbled** to produce σ' if there exists a stochastic (Markov) matrix Γ such that:

```
σ' = Γσ
```

The matrix Γ is called the **garbling matrix**. Each row of Γ represents a probability distribution over output signals given an input signal.

### Example: Perfect vs. Garbled Information

Consider an asset with three possible qualities: LOW, MEDIUM, HIGH.

**Perfect Information (Identity Matrix):**
```
         Signal→  BAD   NEUTRAL  GOOD
Quality↓
LOW              1.0     0.0     0.0
MEDIUM           0.0     1.0     0.0
HIGH             0.0     0.0     1.0
```
This perfectly reveals the true quality.

**Garbled Information (Pooling Matrix):**
```
         Signal→  BAD   NEUTRAL  GOOD
Quality↓
LOW              0.5     0.5     0.0    ← Pool LOW with MEDIUM
MEDIUM           0.5     0.5     0.0    ← Pool LOW with MEDIUM
HIGH             0.0     0.0     1.0    ← HIGH still revealed
```
This "pools" LOW and MEDIUM together, making them indistinguishable.

## Blackwell's Informativeness Theorem

Blackwell proved a remarkable equivalence theorem:

> **Theorem (Blackwell, 1951):** The following are equivalent:
> 1. σ is more informative than σ' (in the sense that σ' = Γσ for some garbling Γ)
> 2. Any expected utility maximizer weakly prefers σ to σ'
> 3. The set of feasible distributions over posteriors under σ' is contained in the set under σ

This establishes a **partial ordering** over information structures called the **Blackwell order**.

### Key Implications

1. **More information is always better** for a single decision-maker (when information is free)
2. **Garbling can only hurt** the decision-maker's expected utility
3. **The ordering is partial**: Many pairs of experiments cannot be ranked

## Strategic Garbling: Bayesian Persuasion

While more information is always better for receivers, senders often have different preferences. This leads to **strategic information disclosure**, famously formalized by Kamenica & Gentzkow (2011) as **Bayesian Persuasion**.

### The Sender's Problem

A sender observes the true state and designs a signal structure to maximize their expected utility, subject to Bayesian updating by the receiver.

**Key Insight:** The sender can benefit from partial revelation (garbling) when their interests diverge from the receiver's.

### Example from This Game

In our asset sale game:
- **Sender** gains +10 whenever the receiver BUYs (regardless of quality)
- **Receiver** gains +20 (HIGH), +5 (MEDIUM), or -15 (LOW) from buying

Under full revelation, the receiver would only buy MEDIUM and HIGH quality assets.

But through strategic garbling, the sender can:
1. Pool LOW with MEDIUM → Receiver sometimes buys LOW
2. Maintain separation of HIGH → Receiver still always buys HIGH

This transfers value from receiver to sender.

## The Economic Value of Garbling

### For the Sender (Information Designer)
- **Benefit:** Can induce actions the receiver wouldn't take under full information
- **Cost:** May destroy trust if receiver becomes too skeptical
- **Optimal strategy:** Balance informativeness with persuasion

### For the Receiver (Decision Maker)  
- **Cost:** Garbled information leads to suboptimal decisions
- **Measured by:** Receiver's regret vs. perfect information benchmark
- **Response:** Bayesian updating, learning sender's strategy over time

### For Society (Total Welfare)
- **Deadweight loss:** Some valuable trades don't happen
- **Value destruction:** Some harmful trades do happen
- **Information asymmetry:** Creates inefficiency

## Running the Game

### Requirements
```bash
pip install numpy
# Optional for LLM agents:
pip install openai
export OPENAI_API_KEY="your-key-here"
```

### Basic Usage
```bash
python garbling_game.py
```

### View Detailed Visualizations
```bash
python visualize_game.py
```

## Game Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GARBLING ECONOMICS GAME                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    ┌──────────────┐                      ┌──────────────┐      │
│    │   NATURE     │                      │   RECEIVER   │      │
│    │  (draws      │                      │  (Bayesian   │      │
│    │   quality)   │                      │   updater)   │      │
│    └──────┬───────┘                      └──────▲───────┘      │
│           │                                     │              │
│           ▼                                     │              │
│    ┌──────────────┐      ┌──────────┐          │              │
│    │   SENDER     │─────▶│ GARBLING │──────────┘              │
│    │  (knows      │      │  MATRIX  │      (sends signal)     │
│    │   quality)   │      └──────────┘                         │
│    └──────────────┘                                           │
│                                                                │
│    Payoffs: BUY(LOW)=-15/+10, BUY(MED)=+5/+10, BUY(HIGH)=+20/+10│
│             PASS = 0/0 for both                                │
└─────────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Description |
|------|-------------|
| `garbling_game.py` | Main game implementation with LLM agents |
| `visualize_game.py` | Visualization and analysis tools |
| `garbling_game_results.json` | Results from most recent game |

## Economic Concepts Demonstrated

1. **Blackwell Informativeness**: How to rank information structures
2. **Garbling Matrices**: How noise is added to signals
3. **Bayesian Persuasion**: Strategic information disclosure
4. **Value of Information**: Quantifying the cost of garbled signals
5. **Receiver Regret**: Measuring decision quality vs. perfect info
6. **Deadweight Loss**: Social welfare lost to information asymmetry

## References

1. Blackwell, D. (1951). "Comparison of Experiments." *Proceedings of the Second Berkeley Symposium on Mathematical Statistics and Probability*, 93-102.

2. Blackwell, D. (1953). "Equivalent Comparisons of Experiments." *Annals of Mathematical Statistics*, 24, 265-272.

3. Kamenica, E., & Gentzkow, M. (2011). "Bayesian Persuasion." *American Economic Review*, 101(6), 2590-2615.

4. Bergemann, D., & Morris, S. (2019). "Information Design: A Unified Perspective." *Journal of Economic Literature*, 57(1), 44-95.

5. Lehrer, E., Rosenberg, D., & Shmaya, E. (2013). "Garbling of Signals and Outcome Equivalence." *Games and Economic Behavior*, 81, 179-191.

## License

MIT License - Feel free to use for educational and research purposes.
