Architecture
============

Overview
--------

Garbling Gym is structured around three core concepts:

1. **Game Engine**: Manages the simulation loop and payoff calculations
2. **Agents**: Implement sender and receiver strategies (rule-based or LLM-powered)
3. **Storage & Analysis**: Persist results and provide analysis tools

Component Structure
-------------------

Core Game Loop
~~~~~~~~~~~~~~

The simulation follows this sequence:

1. Nature draws a random state
2. Sender observes state and produces a signal
3. Receiver observes signal and makes a buy/no-buy decision
4. Payoffs are calculated based on state and decision

This repeats for the configured number of rounds.

Sender Strategies
~~~~~~~~~~~~~~~~~

Senders can use various strategies:

- **Informative**: Always reveals the true state
- **Uninformative**: Always sends the same signal regardless of state
- **LLM-based**: Uses language model reasoning to choose signals strategically

See :class:`garbling_gym.core.agents.base.Sender` for implementation details.

Receiver Strategies
~~~~~~~~~~~~~~~~~~~

Receivers process signals and make decisions:

- **Rule-based**: Uses Bayesian inference on signal statistics
- **LLM-based**: Uses language model reasoning with game context

See :class:`garbling_gym.core.agents.base.Receiver` for implementation details.

LLM Integration
~~~~~~~~~~~~~~~

When ``use_llm=True``, agents are wrapped with:

- Structured prompts explaining the game mechanics
- Conversation history for learning across rounds
- Pydantic-AI validation of responses

The LLM sees:
- Current round number and total rounds
- For senders: the true state
- For receivers: the signal and prior statistics
- Previous round outcomes

Results Storage
~~~~~~~~~~~~~~~

Results are stored in SQLite with the following schema:

- **runs**: Metadata about each experiment run
- **rounds**: Individual round outcomes (state, signal, decision, payoffs)
- Full-text search and aggregation queries supported

See :class:`garbling_gym.core.storage.ResultsStore` for API details.

Extensibility
-------------

Adding New Strategies
~~~~~~~~~~~~~~~~~~~~~

To add a custom sender strategy::

    from garbling_gym.core.agents.base import Sender

    class MyCustomSender(Sender):
        def send_signal(self, state: int) -> int:
            # Your strategy logic here
            return signal

Register it in the factory::

    from garbling_gym.core.agents.factory import SENDER_STRATEGIES
    SENDER_STRATEGIES['my_custom'] = MyCustomSender

Adding New Analysis
~~~~~~~~~~~~~~~~~~~

The dashboard is built with Marimo reactive notebooks. To add new visualizations:

1. Edit ``src/garbling_gym/web/notebooks/dashboard.py``
2. Add new ``@app.cell`` with your analysis
3. Use Plotly for interactive charts

Module Organization
-------------------

::

    garbling_gym/
    ├── core/
    │   ├── agents/          # Sender/receiver implementations
    │   ├── game.py          # Core game engine
    │   └── storage.py       # Results persistence
    ├── cli/                 # Command-line interface
    └── web/
        └── notebooks/       # Marimo dashboard
