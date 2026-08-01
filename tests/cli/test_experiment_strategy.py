# ABOUTME: Regression test that `gg experiment` honors receiver_strategy from YAML
# ABOUTME: Guards the bug at experiment.py:_run_single_experiment (create_receiver
# ABOUTME: was called without strategy_name, so all batch runs used the default).

import yaml

from garbling_gym.cli.commands.experiment import _create_game_config, _run_single_experiment
from garbling_gym.core.agents.strategies.bayesian import DirichletBayesianStrategy


def test_create_game_config_reads_receiver_strategy():
    cfg = _create_game_config({
        "name": "t",
        "rounds": 5,
        "receiver_strategy": "bayesian",
    })
    assert cfg.receiver_strategy == "bayesian"


def test_run_single_experiment_uses_configured_receiver_strategy(monkeypatch):
    """The Game built by _run_single_experiment must receive the strategy
    named in config, not silently fall back to 'heuristic'."""
    captured = {}

    real_create_receiver = None
    from garbling_gym.core.agents import agent_factory, registry as agent_registry_mod
    real_create_receiver = agent_factory.create_receiver

    def spy_create_receiver(self, model="gpt-4o-mini", strategy_name="heuristic", config=None):
        captured["strategy_name"] = strategy_name
        return real_create_receiver(model=model, strategy_name=strategy_name, config=config)

    monkeypatch.setattr(agent_registry_mod.AgentFactory, "create_receiver", spy_create_receiver)

    cfg = _create_game_config({"name": "t", "rounds": 3, "receiver_strategy": "bayesian"})
    _run_single_experiment(cfg, {"name": "t", "rounds": 3, "receiver_strategy": "bayesian"})

    assert captured.get("strategy_name") == "bayesian", (
        "experiment should pass config.receiver_strategy through to the factory"
    )
