Quick Start Guide
=================

Installation
------------

Clone the repository and install dependencies using uv::

    git clone <repo-url>
    cd garbling-sims
    uv sync

For LLM support, install the optional dependencies::

    uv sync --extra llm

Running Your First Simulation
------------------------------

Basic simulation with rule-based agents::

    uv run gg run --sender-strategy informative --num-rounds 10

LLM-powered simulation (requires OpenAI API key)::

    export OPENAI_API_KEY=your-key-here
    uv run gg run --use-llm --num-rounds 5

Viewing Results
---------------

Launch the interactive dashboard::

    uv run gg serve

List recent runs::

    uv run gg list

View detailed analysis of a specific run::

    uv run gg visualize <run-id>

Compare multiple runs::

    uv run gg compare <run-id-1> <run-id-2>

Running Experiments
-------------------

Create an experiment configuration file (YAML)::

    experiments:
      - name: "baseline"
        sender_strategy: "informative"
        num_rounds: 100

      - name: "llm_test"
        use_llm: true
        num_rounds: 50

Run all experiments::

    uv run gg experiment config.yaml

Configuration
-------------

The framework uses environment variables for configuration:

- ``OPENAI_API_KEY``: Required for LLM-powered agents
- ``RESULTS_DB_PATH``: Custom location for results database (default: ``run_results/results.db``)

See :doc:`cli` for complete command reference.
