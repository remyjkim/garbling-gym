# Experimental Setup Assumptions and Parameter Definitions

## Overview

This document defines the complete experimental setup for garbling-gym: the assumptions underpinning the simulation, all configurable and fixed parameters, the garbling strategies available, agent decision models, and the metrics used to evaluate outcomes. It serves as the authoritative reference for reproducing experiments and understanding what the simulation does and does not model.

## Table of Contents

1. [Theoretical Framework](#theoretical-framework)
2. [State Space Definitions](#state-space-definitions)
3. [Game Configuration Parameters](#game-configuration-parameters)
4. [Payoff Structure](#payoff-structure)
5. [Garbling Strategies](#garbling-strategies)
6. [Informativeness Metric](#informativeness-metric)
7. [Agent Decision Models](#agent-decision-models)
8. [Game Flow and Timing](#game-flow-and-timing)
9. [Derived Metrics and Summary Statistics](#derived-metrics-and-summary-statistics)
10. [Role of the LLM in the Current Architecture](#role-of-the-llm-in-the-current-architecture)
11. [Signal Channel Analysis](#signal-channel-analysis)
12. [Design Direction: LLM-as-Garbler](#design-direction-llm-as-garbler)
13. [Key Assumptions and Limitations](#key-assumptions-and-limitations)

---

## Theoretical Framework

Garbling-gym models a repeated Bayesian persuasion game between two agents:

- **Sender** (information holder): Observes true asset quality, chooses how much noise to inject into the signal sent to the receiver.
- **Receiver** (decision maker): Observes a potentially garbled signal and decides whether to BUY or PASS.

The framework draws on two foundational results:

1. **Blackwell's Informativeness Theorem (1951)**: An information structure σ is "more informative" than σ' if σ' can be derived from σ by applying a stochastic matrix Γ (a garbling). More informative signals are always weakly preferred by decision-makers.

2. **Bayesian Persuasion (Kamenica & Gentzkow, 2011)**: A sender who commits to an information structure before observing the state can benefit from strategic garbling when their interests diverge from the receiver's.

The simulation does **not** model commitment — the sender chooses a strategy each round after observing the true quality. This makes it a cheap-talk variant rather than a pure Bayesian persuasion game.

---

## State Space Definitions

### Asset Quality (True State)

Three-state discrete enum (`AssetQuality`):

| Value | Enum | Description |
|-------|------|-------------|
| 0 | `LOW` | Low-quality asset |
| 1 | `MEDIUM` | Medium-quality asset |
| 2 | `HIGH` | High-quality asset |

### Signal (Observed by Receiver)

Three-state discrete enum (`Signal`), same cardinality as quality:

| Value | Enum | Description |
|-------|------|-------------|
| 0 | `BAD` | Negative signal |
| 1 | `NEUTRAL` | Ambiguous signal |
| 2 | `GOOD` | Positive signal |

### Action (Receiver's Choice)

Binary discrete enum (`Action`):

| Value | Enum | Description |
|-------|------|-------------|
| 0 | `PASS` | Decline to purchase |
| 1 | `BUY` | Purchase the asset |

---

## Game Configuration Parameters

All parameters are encapsulated in the `GameConfig` dataclass (`core/config.py`).

### Prior Distribution — `prior`

| Parameter | Type | Default | Constraint |
|-----------|------|---------|------------|
| `prior[LOW]` | float | 0.3 | ≥ 0 |
| `prior[MEDIUM]` | float | 0.4 | ≥ 0 |
| `prior[HIGH]` | float | 0.3 | ≥ 0 |

**Constraint**: Must sum to 1.0 (tolerance: 1e-6).

The prior P(quality) is common knowledge — both sender and receiver know it. Nature samples each round's true quality from this distribution using `random.choices`.

### Round Count — `num_rounds`

| Parameter | Type | Default | Constraint |
|-----------|------|---------|------------|
| `num_rounds` | int | 20 | > 0 |

Number of sequential rounds in a single game. Each round is an independent draw from the prior (qualities are i.i.d.), but agents observe and can condition on the full history of prior rounds.

### LLM Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_llm` | bool | `False` | Enable LLM-powered agents via pydantic-ai |
| `llm_model` | str | `"gpt-4o-mini"` | Model identifier (auto-prefixed with `openai:` if needed) |

When `use_llm=False`, agents use heuristic Bayesian strategies. When `True`, requires `OPENAI_API_KEY` in the environment; falls back to heuristics if the API call fails.

---

## Payoff Structure

Defined in `PayoffStructure` (`core/payoffs.py`). Payoffs are indexed by `(action, quality)` and return a `(sender_payoff, receiver_payoff)` tuple.

### Payoff Matrix

|  | LOW | MEDIUM | HIGH |
|--|-----|--------|------|
| **BUY** | Sender: +10, Receiver: **-15** | Sender: +10, Receiver: +5 | Sender: +10, Receiver: +20 |
| **PASS** | Sender: 0, Receiver: 0 | Sender: 0, Receiver: 0 | Sender: 0, Receiver: 0 |

### Key Properties

- **Sender payoff is action-dependent, quality-independent**: The sender earns +10 on any BUY, 0 on any PASS. This creates the core misalignment — the sender wants BUY regardless of quality.
- **Receiver payoff is both action- and quality-dependent**: The receiver bears all quality risk. BUY+LOW is a large loss (-15), BUY+HIGH is a large gain (+20).
- **PASS is always zero-sum**: Both agents earn 0 when the receiver passes. PASS is the safe outside option.

### Expected Receiver Payoff Under Full Information

If the receiver always sees the true quality and acts optimally (BUY on MEDIUM/HIGH, PASS on LOW):

```
E[receiver | full info] = P(MEDIUM) × 5 + P(HIGH) × 20
                        = 0.4 × 5 + 0.3 × 20
                        = 2 + 6 = 8 per round
```

This is the **perfect information benchmark** used to compute receiver regret.

### Expected Receiver Payoff Under No Information

If the receiver sees a completely uninformative signal, their expected payoff from BUY is:

```
E[BUY | no info] = P(LOW)×(-15) + P(MEDIUM)×5 + P(HIGH)×20
                 = 0.3×(-15) + 0.4×5 + 0.3×20
                 = -4.5 + 2 + 6 = 3.5
```

Since 3.5 > 0, a risk-neutral receiver would still BUY under the default prior even with no information. This is a design property: the prior is optimistic enough that the sender's garbling challenge is about maximizing sales of LOW-quality assets specifically, not about inducing buying in general.

---

## Garbling Strategies

A garbling strategy is a 3×3 **row-stochastic matrix** Γ where Γ[i][j] = P(Signal=j | Quality=i). Each row sums to 1.0, all entries are non-negative. Defined via the `GarblingStrategy` dataclass (`core/strategies/base.py`).

### Built-in Strategies

Six strategies are pre-registered in `BUILTIN_STRATEGIES` (`core/strategies/builtin.py`):

#### 1. `full_revelation` — Identity Matrix

```
Γ = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
```

Signal perfectly reveals quality. LOW→BAD, MEDIUM→NEUTRAL, HIGH→GOOD with certainty. Informativeness: **1.0**.

#### 2. `complete_noise` — Uniform Matrix

```
Γ = [[1/3, 1/3, 1/3],
     [1/3, 1/3, 1/3],
     [1/3, 1/3, 1/3]]
```

Signal is independent of quality — pure noise. Informativeness: **0.0**.

#### 3. `pool_low_medium`

```
Γ = [[0.5, 0.5, 0.0],
     [0.5, 0.5, 0.0],
     [0.0, 0.0, 1.0]]
```

LOW and MEDIUM produce identical signal distributions → receiver cannot distinguish them. HIGH is fully revealed. Informativeness: **~0.29**.

#### 4. `pool_medium_high`

```
Γ = [[1.0, 0.0, 0.0],
     [0.0, 0.5, 0.5],
     [0.0, 0.5, 0.5]]
```

MEDIUM and HIGH produce identical signal distributions → receiver cannot distinguish them. LOW is fully revealed. Informativeness: **~0.29**.

#### 5. `slight_noise`

```
Γ = [[0.80, 0.15, 0.05],
     [0.15, 0.70, 0.15],
     [0.05, 0.15, 0.80]]
```

Mostly truthful with small cross-contamination. Diagonal-dominant. Informativeness: **~0.64**.

#### 6. `aggressive_pooling`

```
Γ = [[0.3, 0.4, 0.3],
     [0.2, 0.3, 0.5],
     [0.1, 0.2, 0.7]]
```

Optimistic bias: all qualities have substantial probability of producing a GOOD signal. LOW quality is spread nearly uniformly. Informativeness: **~0.09**.

### Custom Strategies

Custom strategies can be registered at runtime via `strategy_registry.register(name, GarblingStrategy(matrix=...))`. The registry prevents duplicate names (raises `ValueError`).

---

## Informativeness Metric

Defined in `GarblingStrategy.informativeness_score()` (`core/strategies/base.py`).

### Formula

```
informativeness = 1 - (‖Γ - I‖_F / ‖U - I‖_F)
```

Where:
- Γ is the garbling matrix
- I is the 3×3 identity matrix (perfect information)
- U is the 3×3 uniform matrix (1/3 everywhere, no information)
- ‖·‖_F is the Frobenius norm

### Properties

- Range: [0, 1]
- `full_revelation` (Γ = I) → 1.0
- `complete_noise` (Γ = U) → 0.0
- Monotonic: matrices closer to the identity (in Frobenius distance) score higher
- **Not a true Blackwell ordering**: Two strategies can have the same informativeness score without being Blackwell-comparable. The score is a scalar summary, not a partial order.

---

## Agent Decision Models

### Sender Heuristic Strategy

When `use_llm=False`, the sender uses a quality-conditional mixed strategy that adapts to the receiver's recent payoff history.

**Inputs**: True quality, receiver's average payoff over the last 5 rounds.

| Quality | Condition | Strategy Distribution |
|---------|-----------|----------------------|
| HIGH | Always | 70% `full_revelation`, 30% `slight_noise` |
| MEDIUM | `receiver_avg > 0` | 40% `full_revelation`, 30% `slight_noise`, 30% `pool_medium_high` |
| MEDIUM | `receiver_avg ≤ 0` | 40% `pool_medium_high`, 30% `slight_noise`, 30% `aggressive_pooling` |
| LOW | `receiver_avg < -3` | 40% `pool_low_medium`, 30% `complete_noise`, 30% `slight_noise` |
| LOW | `receiver_avg ≥ -3` | 40% `pool_low_medium`, 35% `aggressive_pooling`, 25% `pool_medium_high` |

**History window**: Last 5 rounds (parsed from prompt text).

### Receiver Heuristic Decision

When `use_llm=False`, the receiver implements a three-stage Bayesian decision model:

#### Stage 1: Estimate P(Signal | Quality) — Signal Reliability

- **Cold start** (< 3 history rounds): Uses fixed uninformative priors:

  | Signal | P(signal \| LOW) | P(signal \| MEDIUM) | P(signal \| HIGH) |
  |--------|------------------|--------------------|--------------------|
  | GOOD | 0.2 | 0.4 | 0.8 |
  | NEUTRAL | 0.3 | 0.4 | 0.15 |
  | BAD | 0.5 | 0.2 | 0.05 |

- **With history** (≥ 3 rounds): Empirical estimation from signal-quality co-occurrences with **Laplace smoothing** (add-1 smoothing, denominator + 3):

  ```
  P̂(signal | quality) = (count(signal, quality) + 1) / (count(quality) + 3)
  ```

#### Stage 2: Bayesian Posterior Update

Computes P(Quality | Signal) via Bayes' rule:

```
P(quality | signal) = P(signal | quality) × P(quality) / P(signal)
```

Where P(quality) is the fixed prior {LOW: 0.3, MEDIUM: 0.4, HIGH: 0.3}. Normalization is computed over the three quality states.

#### Stage 3: Expected Utility and Decision

1. **Expected value of BUY**:
   ```
   E[BUY] = P(LOW|signal)×(-15) + P(MEDIUM|signal)×5 + P(HIGH|signal)×20
   ```

2. **Risk adjustment** based on recent average payoff:
   - If `avg_payoff < -5`: subtract 3 from E[BUY] (risk aversion after losses)
   - If `avg_payoff > 5`: add 1 to E[BUY] (confidence after gains)

3. **Sigmoid decision** with bounded rationality:
   ```
   P(BUY) = sigmoid(E[BUY] - 1) = 1 / (1 + exp(-(E[BUY] - 1)))
   ```
   The threshold shift of -1 adds a slight bias toward PASS (requires positive expected value to favor buying).

4. **Stochastic action**: Sample uniformly in [0, 1]; BUY if sample < P(BUY).

### LLM Agent Mode

When `use_llm=True`, both agents receive detailed system prompts explaining the game theory and use pydantic-ai for structured output:

- **Sender**: Returns `StrategyChoice` (constrained `Literal` over the 6 built-in strategy names)
- **Receiver**: Returns `ActionChoice` (constrained `Literal["BUY", "PASS"]`)

Both agents receive the last 5 rounds of history as context. If the LLM call fails, agents fall back to the heuristic strategies above.

### Trust Analysis (Receiver)

The receiver tracks an **apparent deception rate** across all observed rounds:

```
deception = count of rounds where (signal=GOOD ∧ quality=LOW) ∨ (signal=BAD ∧ quality=HIGH)
deception_rate = deception / total_rounds
```

This is provided as context to the LLM agent but is not directly used by the heuristic — the heuristic relies on the Bayesian signal reliability estimation instead.

---

## Game Flow and Timing

Each game consists of `num_rounds` sequential rounds. Within each round:

1. **Nature samples quality** from `prior` via `random.choices`
2. **Sender observes quality**, chooses strategy name (string)
3. **Strategy looked up** in `strategy_registry`
4. **Signal sampled** from garbling matrix row for the given quality via `np.random.choice`
5. **Receiver observes signal** (not quality, not strategy), chooses action
6. **Payoffs computed** from `(action, quality)` pair
7. **Round recorded** in `GameState.history`

### Information Sets

| Agent | Knows at Decision Time |
|-------|----------------------|
| Sender | True quality, full history of all prior rounds (quality, strategy, signal, action, payoffs) |
| Receiver | Current signal, full history of all prior rounds (signal, action, true quality revealed post-hoc, payoffs) |

Both agents have access to the complete history. The receiver learns the true quality **after each round** (post-hoc revelation), enabling learning about the sender's garbling behavior over time.

### Timing Between Rounds

- Heuristic mode: No delay between rounds
- LLM mode: 0.5-second delay between rounds (`time.sleep(0.5)`)

---

## Derived Metrics and Summary Statistics

Computed in `Game._compute_summary()` and stored in `GameResults`.

| Metric | Type | Description |
|--------|------|-------------|
| `total_rounds` | int | Number of rounds played |
| `sender_total` | float | Cumulative sender payoff |
| `receiver_total` | float | Cumulative receiver payoff |
| `strategies_used` | Dict[str, int] | Count of each strategy name used |
| `buy_rate` | float | Proportion of rounds where receiver chose BUY |
| `avg_informativeness` | float | Mean informativeness score across all rounds |
| `perfect_info_benchmark` | float | Receiver payoff under optimal play with full information |
| `receiver_regret` | float | `perfect_info_benchmark - receiver_total` |
| `quality_stats` | Dict[str, Dict] | Per-quality breakdown (count, bought, sender_total, receiver_total) |

### Perfect Information Benchmark

```
perfect_info_benchmark = Σ over all rounds:
    +5   if quality == MEDIUM
    +20  if quality == HIGH
    0    if quality == LOW
```

This assumes the receiver buys MEDIUM and HIGH, passes on LOW — which is optimal under the default payoff structure.

### Receiver Regret

```
receiver_regret = perfect_info_benchmark - receiver_total
```

Measures the cost of information garbling to the receiver. Always non-negative under rational play. A regret of 0 would mean the receiver achieved perfect-information performance despite garbling.

---

## Role of the LLM in the Current Architecture

A critical clarification: **the LLM does not perform the garbling**. The LLM's role is limited to agent decision-making — choosing which pre-defined matrix to use (sender) and how to react to a discrete signal (receiver). The actual information transformation between sender and receiver is always a single `np.random.choice(3, p=matrix_row)` call.

### What the LLM Does

Per round, LLM mode makes **two API calls** via pydantic-ai:

1. **Sender call**: Receives true quality + last 5 rounds of history. Returns a strategy name (one of 6 strings). The LLM is choosing which garbling matrix to apply — it is not constructing or transmitting the signal itself.

2. **Receiver call**: Receives the discrete signal token (BAD/NEUTRAL/GOOD) + last 5 rounds of history + trust analysis. Returns BUY or PASS.

### What the LLM Does Not Do

- The LLM never crafts, phrases, or modulates the signal
- The LLM never communicates directly with the other agent
- The LLM has no influence on the garbling operation itself (step 4 in the game flow)
- The entire sender-to-receiver channel is a single enum value sampled from a matrix row

The LLM is a strategy selector and action selector, not an information transformer.

---

## Signal Channel Analysis

### Channel Bandwidth

The signal channel between sender and receiver is extremely narrow:

```
Sender's knowledge:  AssetQuality ∈ {LOW, MEDIUM, HIGH}     — 3 states
Channel output:      Signal ∈ {BAD, NEUTRAL, GOOD}           — 3 tokens
Receiver's input:    one Signal token per round               — ~1.58 bits max
```

The receiver sees exactly one of three labels each round. There is no continuous value, no natural language, no partial disclosure, no structured report. The "message" is a single enum.

### Why the Matrix is 3×3

The garbling matrix Γ is 3×3 because it maps 3 quality states (rows) to 3 signal states (columns). Each cell Γ[i][j] = P(Signal=j | Quality=i) defines how likely each signal is for a given true quality. The full matrix is needed because the garbling pattern can differ per quality — e.g., `pool_low_medium` makes LOW and MEDIUM indistinguishable while fully revealing HIGH.

### Informativeness Score vs. Garbling Matrix

The informativeness score (a single decimal in [0, 1]) is a **lossy scalar summary** of the 3×3 matrix, not the garbling mechanism itself. The matrix defines the channel; the score summarizes how close it is to perfect information.

This distinction matters: two matrices can have the **same informativeness score** but garble in structurally different ways with different economic consequences:

- `pool_low_medium` (score ~0.29): Hides LOW among MEDIUM, reveals HIGH
- `pool_medium_high` (score ~0.29): Hides MEDIUM among HIGH, reveals LOW

Same score, opposite strategic implications. The sender choosing between these two is making a meaningful economic decision that the scalar does not capture.

### Receiver's Inference Task

Despite the narrow channel, the receiver's job is substantive. Given a single signal token, the receiver must:

1. Estimate how reliably the sender is transmitting (what garbling strategy is likely in play?)
2. Update beliefs about true quality via Bayes' rule
3. Compute expected value of BUY vs. PASS
4. Act under uncertainty

The receiver builds a model of the sender's garbling behavior from history and uses it to "see through" the noise. This is the core game-theoretic tension: the sender garbles to exploit, the receiver learns to discount.

---

## Design Direction: LLM-as-Garbler

The current architecture uses the LLM for decision-making but not for the garbling itself. A natural extension is to make the LLM the garbling mechanism — replacing the matrix channel with natural language communication.

### Current vs. Proposed Flow

**Current (matrix garbling):**
```
Quality → [sender picks matrix] → [np.random.choice from matrix row] → discrete Signal → Receiver
```

**Proposed (LLM garbling):**
```
Quality → [sender LLM crafts natural language message] → free-form text → Receiver LLM → Action
```

In the proposed model, the sender writes a natural language description of the asset (knowing the true quality), and the receiver reads that message and decides. The garbling *is* the language — what the sender emphasizes, omits, frames, or spins. This is closer to real-world persuasion: a seller describing a car, an analyst writing a research note, a company's earnings call.

### What This Would Change

| Dimension | Matrix Mode | LLM-as-Garbler Mode |
|-----------|------------|---------------------|
| Signal space | 3 discrete tokens | Free-form natural language |
| Sender's expressive power | Pick 1 of 6 matrices | Unlimited linguistic framing |
| Garbling mechanism | Probabilistic sampling from matrix row | Strategic language generation |
| Measurability | Exact (informativeness from matrix) | Requires proxy metrics |
| Reproducibility | Deterministic given seed | Stochastic, model-dependent |
| Ecological validity | Low | High |
| Heuristic fallback | Works (matrix math) | No clear non-LLM equivalent |

### Open Design Questions

1. **Coexistence**: Should this be a separate game variant (preserving the matrix mode for controlled experiments) or a replacement?

2. **Sender constraints**: Should the sender have a message budget (word count, template), or unconstrained free text? Constraints make analysis easier; free text makes the persuasion dynamics richer.

3. **Measuring garbling post-hoc**: Without a matrix, how do you quantify how much information the message reveals? Possible approaches:
   - An evaluator LLM that scores message informativeness
   - Empirical accuracy: how often does the receiver correctly infer quality from the message alone?
   - Embedding-space distance between messages for different qualities (do LOW and HIGH messages cluster separately?)

4. **Architecture**: A `GarblingChannel` interface that both `GarblingStrategy` (matrix) and an `LLMGarblingChannel` implement. The game loop calls `channel.transform(quality) → signal`, where `signal` is either a discrete enum or a string. This preserves the game flow while swapping the channel.

5. **What question does each mode answer?**
   - Matrix mode: "Given a known information structure, how do rational agents behave?"
   - LLM mode: "How well can an LLM strategically persuade another LLM, and can the receiver learn to see through it?"

---

## Key Assumptions and Limitations

### Modeling Assumptions

1. **No commitment**: The sender chooses a strategy after observing quality each round. This is a key departure from classical Bayesian persuasion where the sender commits to an information structure ex-ante.

2. **i.i.d. quality draws**: Each round's quality is drawn independently from the same prior. There is no state persistence, trending, or correlation across rounds.

3. **Discrete, finite state spaces**: 3 quality states, 3 signal states, 2 actions. No continuous distributions or infinite signal spaces. The signal channel carries at most ~1.58 bits per round.

4. **Common prior**: Both agents know and agree on P(quality). There is no disagreement about priors.

5. **Post-hoc quality revelation**: The receiver learns the true quality after each round. This enables learning but is a strong assumption — in many real markets, quality is never fully revealed.

6. **Fixed payoff structure**: Payoffs do not change across rounds or adapt to behavior. There is no discounting, no budget constraints, no transaction costs.

7. **Single sender, single receiver**: No competition among senders or receivers. No market clearing or equilibrium pricing.

8. **No reputation mechanism**: There is no formal reputation score or trust signal. Trust is implicitly modeled through agents observing history, but there is no explicit punishment mechanism for deception.

### Heuristic Agent Limitations

9. **Finite history window**: Both heuristic agents only consider the last 5 rounds of history when making decisions, even though the full history is available.

10. **Prompt-parsing fragility**: Heuristic agents parse their inputs from formatted prompt strings rather than structured data, making them sensitive to prompt format changes.

11. **Non-equilibrium play**: The heuristic strategies are hand-crafted reasonable behaviors, not Nash equilibria or optimal policies. They are designed to demonstrate the economic dynamics, not to represent theoretically optimal play.

12. **Bounded rationality via sigmoid**: The receiver's sigmoid decision function with a threshold of 1 introduces stochastic behavior. This prevents the receiver from being a perfect expected-value maximizer but models realistic bounded rationality.

### Metric Limitations

13. **Informativeness is Frobenius-based, not Blackwell**: The scalar informativeness score based on Frobenius distance from the identity does not capture the full Blackwell partial order. Two strategies with the same score may not be Blackwell-comparable.

14. **Regret assumes optimal BUY/PASS threshold**: The perfect information benchmark assumes the receiver always buys MEDIUM (payoff +5) and HIGH (+20). Under alternative payoff structures where BUY+MEDIUM is negative, this benchmark would need recalculation.

### Architectural Limitations

15. **LLM does not garble**: The LLM is used only for agent decision-making (strategy selection and action choice), not for the information transformation itself. The garbling operation is a single `np.random.choice` call from a pre-defined matrix. This means the simulation does not model natural language persuasion, strategic framing, or any form of linguistic garbling.

16. **Narrow signal channel**: The entire sender-to-receiver communication is one of 3 discrete tokens per round. There is no mechanism for partial disclosure, continuous signals, structured reports, or natural language messages. The sender's expressive power is limited to selecting among 6 pre-defined matrices.

17. **Informativeness score is lossy**: The Frobenius-based scalar collapses a 3×3 matrix (9 parameters) to a single number. Two matrices with the same score can have opposite strategic implications (e.g., `pool_low_medium` vs. `pool_medium_high` both score ~0.29 but hide different qualities).

### Reproducibility

18. **Randomness**: Signal generation uses `np.random.choice` and quality sampling uses `random.choices`. For deterministic reproduction, both `random.seed()` and `np.random.seed()` must be set. The framework stores the git commit hash per run but does not store or set random seeds automatically.
