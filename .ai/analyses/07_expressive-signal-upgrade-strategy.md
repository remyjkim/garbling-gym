# Expressive Signal Upgrade Strategy

## Problem Statement

The garbling-gym signal channel is a single discrete enum per round: one of {BAD, NEUTRAL, GOOD} (~1.58 bits). The garbling operation is a `np.random.choice` call from a pre-defined 3×3 matrix row. The LLM, when used, only selects which matrix to sample from (sender) or reacts to the resulting token (receiver) — it never touches the signal itself.

This makes the signal space too narrow for studying realistic persuasion dynamics. A seller describing a car doesn't emit one of three labels; they choose what to emphasize, quantify, omit, and frame. The current architecture cannot model partial disclosure, continuous quality estimates, structured reports, or natural language persuasion.

This document proposes a three-tier upgrade path for signal expressiveness, each tier building on the previous, unified under a common `GarblingChannel` interface.

---

## Design Constraint: The Game Loop Stays the Same

All three tiers must plug into the existing round flow without changing the game orchestration logic. The game loop currently does:

```python
# step 2: sender picks strategy
strategy_name = sender.choose_strategy(quality, history, round_num, total_rounds)
# step 3: strategy generates signal
garbling_strategy = registry.get(strategy_name)
signal = garbling_strategy.get_signal(quality)
# step 4: receiver acts on signal
action = receiver.make_decision(signal, history, round_num, total_rounds)
```

The upgrade should generalize steps 2-4 without rewriting the game loop. The key abstraction: replace the tightly coupled (strategy selection → matrix lookup → sample) pipeline with a `GarblingChannel` that takes quality and returns a signal of any type.

### Proposed Interface

```python
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

S = TypeVar('S')  # Signal type

class GarblingChannel(ABC, Generic[S]):
    """Transforms true quality into an observable signal."""

    @abstractmethod
    def transmit(self, quality: AssetQuality, context: ChannelContext) -> S:
        """Generate a signal from true quality."""
        ...

    @abstractmethod
    def informativeness_score(self) -> float | None:
        """
        Return informativeness in [0, 1] if computable, None otherwise.
        Matrix channels can compute this exactly.
        LLM channels may return None or a post-hoc estimate.
        """
        ...
```

Where `ChannelContext` carries the history, round number, and any sender-side reasoning the channel needs. The current `GarblingStrategy` becomes a `MatrixChannel` implementing this interface.

---

## Tier 1: Continuous Scalar Signal

### What Changes

The signal becomes a float in [0, 1] instead of a discrete enum. The garbling operation adds calibrated noise to a continuous quality score.

```
Current:   Quality(LOW)  → matrix row [0.3, 0.4, 0.3] → np.random.choice → Signal.GOOD
Tier 1:    Quality(LOW)  → base_score=0.1 → add noise(σ) → clamp to [0,1] → signal=0.47
```

### Signal Generation

Each quality maps to a base score:

| Quality | Base Score |
|---------|-----------|
| LOW | 0.15 |
| MEDIUM | 0.50 |
| HIGH | 0.85 |

The sender chooses a **noise level** σ (standard deviation of Gaussian noise). Higher σ = more garbling.

```python
class ContinuousChannel(GarblingChannel[float]):
    def __init__(self, noise_sigma: float, base_scores: dict[AssetQuality, float]):
        self.noise_sigma = noise_sigma
        self.base_scores = base_scores

    def transmit(self, quality: AssetQuality, context: ChannelContext) -> float:
        base = self.base_scores[quality]
        noise = np.random.normal(0, self.noise_sigma)
        return np.clip(base + noise, 0.0, 1.0)

    def informativeness_score(self) -> float:
        # Signal-to-noise ratio normalized to [0, 1]
        score_spread = max(self.base_scores.values()) - min(self.base_scores.values())
        if self.noise_sigma == 0:
            return 1.0
        snr = score_spread / self.noise_sigma
        return 1 - np.exp(-snr)  # exponential decay: high noise → low info
```

### Sender's Choice

Instead of picking from 6 named strategies, the sender chooses σ:
- σ = 0: full revelation (base scores are deterministic and separable)
- σ = 0.1: slight noise (distributions overlap a little)
- σ = 0.4: heavy garbling (distributions overlap substantially)
- σ → ∞: complete noise (all qualities produce ~Uniform[0,1])

This gives the sender a continuous strategy space instead of 6 discrete options.

### Receiver's Inference

The receiver observes a decimal and must infer quality. With known σ and base scores:

```
P(quality | signal=x) ∝ P(signal=x | quality) × P(quality)
                      ∝ N(x; base_score[quality], σ²) × prior[quality]
```

This is a standard Gaussian posterior computation — the receiver maintains a continuous Bayesian model instead of counting signal-quality co-occurrences.

### Informativeness Measurement

Computable analytically from σ and the base score spread. As σ increases relative to score separation, informativeness decreases. Unlike the Frobenius distance, this has a natural interpretation as signal-to-noise ratio.

### What This Enables

- Sender can fine-tune garbling intensity (not just pick from 6 presets)
- Receiver sees a nuanced signal, not a coarse label
- Bayesian updating becomes continuous and more realistic
- The game can model scenarios where "the signal is 0.72" carries more information than "GOOD"
- Can study optimal noise levels, receiver learning rates, information value curves

### What This Does Not Change

- Still a scalar channel — one number per round
- Garbling is still a mathematical operation (add noise), not linguistic
- LLM is still only a decision-maker, not the garbler
- The sender doesn't craft a message; it chooses a noise parameter

---

## Tier 2: Structured Multi-Dimensional Signal

### What Changes

The signal becomes a structured object with multiple dimensions, each independently garblable. This models scenarios where a sender reveals partial information — for example, a product listing with a quality rating, a confidence level, and a category tag.

```
Current:   Quality(MEDIUM) → Signal.NEUTRAL
Tier 1:    Quality(MEDIUM) → 0.53
Tier 2:    Quality(MEDIUM) → SignalReport(score=0.53, confidence="high", category="standard")
```

### Signal Structure

```python
@dataclass
class StructuredSignal:
    score: float           # continuous quality estimate in [0, 1]
    confidence: str        # "low" | "medium" | "high" — how sure the sender claims to be
    attributes: dict       # optional key-value pairs (e.g., {"durability": 0.7, "warranty": True})
```

### Garbling Dimensions

The sender can garble each dimension independently:

| Dimension | Garbling Mechanism | Example |
|-----------|--------------------|---------|
| `score` | Gaussian noise (Tier 1) | True base=0.5, report=0.71 |
| `confidence` | Mismatch: claim "high" confidence for noisy scores | LOW quality but "high" confidence |
| `attributes` | Selective disclosure: include/omit attributes | Omit "durability" for LOW quality |

This creates a richer strategic space: the sender can be truthful on one dimension while garbling another. The receiver must triangulate across dimensions — "the score says 0.7 but confidence is high and durability is missing... suspicious."

### Sender's Choice

The sender chooses a **disclosure policy** — a set of per-dimension garbling parameters:

```python
@dataclass
class DisclosurePolicy:
    score_noise: float          # σ for the score dimension
    confidence_honesty: float   # P(reported confidence matches true confidence)
    attribute_reveal_rate: float # P(each attribute is included)
```

### Receiver's Inference

The receiver now runs inference across multiple dimensions. Signal dimensions that agree reinforce confidence; dimensions that disagree trigger suspicion. This naturally models the kind of "something feels off" intuition that human buyers have.

### Informativeness Measurement

Per-dimension informativeness can be computed and then combined:
- Score dimension: SNR as in Tier 1
- Confidence dimension: mutual information between reported and true confidence
- Attribute dimension: fraction of information-bearing attributes disclosed

Overall informativeness is a weighted combination — but the weights themselves become a design parameter.

### What This Enables

- Models partial disclosure: reveal some facts, hide others
- Sender can build credibility on one dimension to exploit another
- Receiver learns which dimensions to trust
- Captures real-world dynamics: a car listing shows mileage (hard to fake) but omits accident history

---

## Tier 3: Natural Language Signal (LLM-as-Garbler)

### What Changes

The signal becomes free-form natural language generated by the sender LLM. The garbling is no longer a mathematical operation — it is the linguistic act of describing an asset while knowing its true quality. The receiver LLM reads the message and decides.

```
Current:   Quality(LOW) → Signal.GOOD (sampled from matrix)
Tier 3:    Quality(LOW) → "This asset shows promising indicators with strong
            upside potential. Recent performance metrics suggest consistent
            improvement across key areas."
```

### Signal Generation

```python
class LLMChannel(GarblingChannel[str]):
    def __init__(self, sender_model: str, constraints: MessageConstraints):
        self.sender_model = sender_model
        self.constraints = constraints

    def transmit(self, quality: AssetQuality, context: ChannelContext) -> str:
        prompt = self._build_sender_prompt(quality, context)
        message = call_llm(self.sender_model, prompt, self.constraints)
        return message

    def informativeness_score(self) -> float | None:
        return None  # cannot be computed analytically
```

### Sender Constraints

Unconstrained free text is hard to analyze. Possible constraint levels:

| Constraint Level | Description | Analysability |
|-----------------|-------------|---------------|
| **Template** | Fill in structured fields: "Quality assessment: ___, Key risks: ___, Outlook: ___" | High — can parse fields |
| **Word budget** | Free text, max N words (e.g., 50) | Medium — finite but open-ended |
| **Unconstrained** | Any text the LLM produces | Low — maximum realism |

Starting with templates is advisable — it preserves some structure for analysis while giving the sender linguistic freedom within each field.

### Receiver's Inference

The receiver LLM reads the natural language message and decides BUY or PASS. It can also be prompted to output its posterior beliefs:

```python
class ReceiverAssessment(BaseModel):
    action: Literal["BUY", "PASS"]
    estimated_quality: Literal["LOW", "MEDIUM", "HIGH"]
    confidence: float  # 0 to 1
    reasoning: str
```

The `estimated_quality` and `confidence` fields enable measuring how well the receiver "sees through" the garbling, independent of the action taken.

### Informativeness Measurement (Post-Hoc)

Since informativeness cannot be computed from a matrix, proxy metrics are needed:

**1. Receiver Accuracy**
```
accuracy = count(estimated_quality == true_quality) / total_rounds
```
Simple and direct. Measures how much information survives the garbling channel. Full revelation ≈ 100% accuracy; complete noise ≈ 33% (random guess over 3 classes).

**2. Discriminability Score**
Run the receiver (or an evaluator LLM) on many messages for each quality. Measure how separable the quality classes are in the receiver's posterior:
```
discriminability = 1 - H(quality | receiver_estimate) / H(quality)
```
Where H is entropy. If the receiver's estimates are independent of true quality, discriminability = 0. If perfect, discriminability = 1.

**3. Embedding-Space Analysis**
Embed all messages using a sentence encoder. Measure whether messages for different qualities cluster separately:
- Compute centroid for each quality's messages
- Measure inter-cluster distance vs. intra-cluster variance
- Higher separation → more informative channel

**4. Evaluator LLM**
A third-party LLM (not the sender or receiver) reads the message and scores "how much does this message reveal about the asset's true quality?" on a scale. This is subjective but captures nuances that accuracy alone misses — a message can be technically accurate but framed to mislead.

### Game Dynamic Differences

In matrix mode, the garbling is fixed per strategy — the sender picks a matrix and randomness does the rest. In LLM mode, the sender actively constructs each message. This changes the game dynamics:

- **No separation between strategy choice and signal generation**: The sender doesn't "pick a strategy then sample." It writes a message that *is* the strategy.
- **Receiver can analyze language patterns**: Beyond Bayesian updating on signal-quality correlations, the receiver can look for linguistic tells — hedging language, vague superlatives, missing specifics.
- **History becomes richer**: Past rounds include full messages, enabling the receiver to learn the sender's rhetorical patterns.
- **Trust is about language, not matrices**: The receiver asks "do I believe this description?" not "what's P(quality|signal)?"

---

## Implementation Strategy

### Phase 1: Channel Abstraction (Foundation)

**Goal**: Introduce the `GarblingChannel` interface without changing current behavior.

1. Define `GarblingChannel` abstract base class with `transmit()` and `informativeness_score()`
2. Implement `MatrixChannel` wrapping the existing `GarblingStrategy` — same behavior, new interface
3. Generalize `Game.play_round()` to call `channel.transmit(quality, context)` instead of the current strategy lookup + `get_signal()` pipeline
4. Generalize the signal type in `RoundResult` from `str` (enum name) to `Any` (or a union type)
5. Ensure all existing tests pass with `MatrixChannel`

**Risk**: The history format changes if signal is no longer always a string. Need to handle serialization for storage.

**Estimated scope**: Small. Mostly interface extraction and one level of indirection.

### Phase 2: Continuous Scalar Channel (Tier 1)

**Goal**: Implement and validate continuous signals.

1. Implement `ContinuousChannel` with configurable base scores and noise σ
2. Update `SenderAgent` to choose σ (heuristic: map quality-conditional logic to noise levels; LLM: return a float)
3. Update `ReceiverAgent` to perform Gaussian posterior inference on a decimal signal
4. Implement SNR-based informativeness metric
5. Update visualization and export to handle decimal signals
6. Write tests: verify noise properties, posterior computation, informativeness bounds
7. Run comparison experiments: matrix channel vs. continuous channel, same games

**Depends on**: Phase 1.

**Key decision**: Should the sender choose σ per-round (adaptive noise) or commit to a fixed σ per-game (closer to Bayesian persuasion)? Start with per-round to match current behavior, add commitment mode later.

### Phase 3: Structured Signal (Tier 2)

**Goal**: Multi-dimensional signals with per-dimension garbling.

1. Define `StructuredSignal` dataclass
2. Implement `StructuredChannel` that garbles each dimension according to a `DisclosurePolicy`
3. Update sender to choose disclosure policies
4. Update receiver to triangulate across dimensions
5. Define per-dimension and aggregate informativeness metrics
6. Update serialization, visualization, export

**Depends on**: Phase 2 (reuses continuous score dimension).

**Key decision**: Which dimensions to include? Start minimal (score + confidence) and add attributes later. Avoid over-designing the schema before seeing experimental results.

### Phase 4: Natural Language Channel (Tier 3)

**Goal**: LLM-generated messages as the garbling mechanism.

1. Implement `LLMChannel` with configurable sender model and message constraints
2. Start with template-based constraints (structured fields the sender fills in)
3. Update receiver to read natural language and output `ReceiverAssessment`
4. Implement post-hoc informativeness proxies (receiver accuracy, discriminability)
5. Update history format to store full messages
6. Run experiments comparing LLM channel to matrix channel on same priors/payoffs

**Depends on**: Phase 1 (channel interface). Does not depend on Phases 2-3 — can be developed in parallel.

**Key decision**: Template vs. free text first? Start with templates — they're easier to analyze and debug, and you can relax constraints incrementally.

### Phase 5: Cross-Channel Comparison Framework

**Goal**: Unified tooling for comparing garbling channels.

1. Experiment configs that specify channel type alongside other parameters
2. Normalized metrics that work across channel types (receiver accuracy, regret, buy rate)
3. Comparison visualizations: same game, different channels
4. Statistical tests for whether channel type significantly affects outcomes

**Depends on**: At least two channel implementations (Phases 1+2, or 1+4).

---

## Migration Concerns

### Backward Compatibility

The matrix channel must remain the default. Existing experiment configs, CLI flags, stored results, and tests must work without modification. The channel type is a new parameter, defaulting to `"matrix"`.

### Storage Format

Current `RoundResult` stores signal as a string (enum name like "GOOD"). With richer signals:
- Tier 1: signal is a float → store as number in JSON
- Tier 2: signal is a structured object → store as nested JSON
- Tier 3: signal is a string (natural language) → store as string in JSON, but much longer

The `results.json` format needs a signal type indicator so that deserialization knows what to expect. Proposed: add `"channel_type": "matrix"|"continuous"|"structured"|"llm"` to the run metadata.

### Informativeness Comparability

Informativeness scores from different channel types are **not directly comparable**. A 0.5 from the Frobenius-based matrix metric means something different from a 0.5 SNR-based continuous metric or a 0.5 receiver-accuracy-based LLM metric.

Options:
- Accept incomparability and label the metric source
- Define a "universal" informativeness metric based on receiver accuracy (works across all channel types, but is empirical not analytical)
- Report both channel-specific and universal metrics

Recommendation: Report channel-specific metrics for within-channel analysis, receiver accuracy for cross-channel comparison.

### CLI Integration

New CLI flags for channel selection:

```bash
gg run --channel matrix                    # default, current behavior
gg run --channel continuous --noise 0.2    # Tier 1
gg run --channel structured                # Tier 2
gg run --channel llm --llm                 # Tier 3 (requires --llm)
```

Tier 3 implies `--llm` since the sender must be an LLM. Tiers 1-2 work with both heuristic and LLM agents.

---

## What to Build First

Phase 1 (channel abstraction) is non-negotiable — it's pure refactoring with no behavior change and unblocks everything else.

After that, the decision depends on what question matters most:

- **"How does signal granularity affect game dynamics?"** → Phase 2 (continuous). Cheapest way to get richer signals. No LLM dependency. Clean math.
- **"Can LLMs strategically persuade each other?"** → Phase 4 (natural language). Skips Tiers 1-2. Most novel research question. Highest implementation cost.
- **"How does partial disclosure work?"** → Phase 3 (structured). Models real-world information design. Mid-range complexity.

Phases 2 and 4 are independent and can be developed in parallel after Phase 1.
