Garbling Gym Documentation
==========================

**Garbling Gym** is an extensible experiment framework for garbling economics simulations, focusing on information economics and Bayesian persuasion.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   architecture
   api
   cli

Overview
--------

Garbling Gym provides a flexible framework for running economic simulations where:

- A **Sender** observes the true state and sends signals to influence decisions
- A **Receiver** observes signals and decides whether to "buy" (take action)
- Payoffs depend on states, signals, and receiver decisions

The framework supports both:

- **Rule-based agents** using deterministic strategies
- **LLM-powered agents** using language models for strategic reasoning

Key Features
------------

- **Extensible Architecture**: Easy to add new sender/receiver strategies
- **LLM Integration**: Optional pydantic-ai integration for GPT-powered agents
- **Result Storage**: SQLite-based storage with rich querying capabilities
- **Interactive Dashboard**: Marimo-based web dashboard for analysis
- **CLI Tools**: Command-line interface for running experiments
- **Comprehensive Testing**: Full test coverage with pytest

Quick Start
-----------

Install dependencies::

    uv sync
    uv sync --extra llm  # for LLM support

Run a simulation::

    uv run gg run --sender-strategy informative --num-rounds 10

View results in the dashboard::

    uv run gg serve

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
