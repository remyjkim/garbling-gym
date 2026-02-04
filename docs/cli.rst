Command Line Interface
======================

The ``gg`` command provides a comprehensive CLI for running simulations and analyzing results.

Main Command
------------

.. click:: garbling_gym.cli.__main__:main
   :prog: gg
   :nested: full

Commands Overview
-----------------

run
~~~

Run a single simulation with specified parameters::

    gg run [OPTIONS]

Key options:

- ``--num-rounds``: Number of rounds to play (default: 100)
- ``--sender-strategy``: Strategy for sender (informative, uninformative, random)
- ``--use-llm``: Use LLM-powered agents
- ``--name``: Name for this run
- ``--tags``: Comma-separated tags

Examples::

    # Basic run with informative sender
    gg run --sender-strategy informative --num-rounds 50

    # LLM-powered run
    gg run --use-llm --num-rounds 10 --name "gpt4_test"

    # Tagged run
    gg run --tags "baseline,test" --num-rounds 100

list-runs
~~~~~~~~~

List past experiment runs::

    gg list-runs [OPTIONS]

Options:

- ``--limit``: Maximum number of runs to show (default: 20)
- ``--tags``: Filter by tags
- ``--use-llm``: Filter by LLM usage

Examples::

    # Show last 10 runs
    gg list-runs --limit 10

    # Show only LLM runs
    gg list-runs --use-llm

visualize
~~~~~~~~~

Visualize results from a completed run::

    gg visualize <RUN_ID>

Creates visualizations showing:

- State distribution over time
- Signal patterns
- Buy decisions
- Cumulative payoffs
- Informativeness metrics

compare
~~~~~~~

Compare two runs side-by-side::

    gg compare <RUN_ID_1> <RUN_ID_2>

Shows comparative analysis of:

- Final payoffs
- Buy rates
- Signal informativeness
- Strategy effectiveness

serve
~~~~~

Launch the interactive web dashboard::

    gg serve

Opens a Marimo-based dashboard at http://localhost:2718 with:

- Interactive run browser
- Filtering and sorting
- Multi-run comparison charts
- Aggregate statistics

experiment
~~~~~~~~~~

Run batch experiments from a YAML configuration::

    gg experiment <CONFIG_FILE>

Example configuration file::

    experiments:
      - name: "baseline_informative"
        sender_strategy: "informative"
        num_rounds: 100
        tags: ["baseline"]

      - name: "llm_experiment"
        use_llm: true
        num_rounds: 50
        tags: ["llm", "test"]

export
~~~~~~

Export run results to static files::

    gg export <RUN_ID> [OPTIONS]

Options:

- ``--format``: Output format (json, csv)
- ``--output``: Output file path

Environment Variables
---------------------

- ``OPENAI_API_KEY``: Required for LLM-powered agents
- ``RESULTS_DB_PATH``: Custom database location (default: ``run_results/results.db``)
- ``GG_VERBOSE``: Enable verbose logging

Configuration Files
-------------------

Experiment configuration files use YAML format::

    experiments:
      - name: string          # Required: experiment name
        sender_strategy: str  # Optional: sender strategy
        num_rounds: int       # Optional: number of rounds (default: 100)
        use_llm: bool         # Optional: use LLM agents (default: false)
        tags: list[str]       # Optional: tags for filtering

See :doc:`quickstart` for more examples.
