# ABOUTME: Tests that receiver strategies honor config-injected prior/payoffs
# ABOUTME: Regression guard: defaults must remain unchanged after the refactor

import pytest

from garbling_gym.core.agents.strategies import ReceiverStrategy
from garbling_gym.core.agents.strategies.bayesian import DirichletBayesianStrategy
from garbling_gym.core.agents.strategies.regret import RegretMatchingStrategy, HedgeStrategy
from garbling_gym.core.agents.strategies.game_theoretic import LevelKStrategy
from garbling_gym.core.agents.strategies.legacy_heuristic import LegacyHeuristicStrategy
from garbling_gym.core.agents.strategies.bandit import BanditStrategy
from garbling_gym.core.types import Action, AssetQuality, Signal


# The full (action, quality) receiver-payoff table, keyed by names.
DEFAULT_PAYOFFS = {
    ("BUY", "LOW"): -15.0, ("BUY", "MEDIUM"): 5.0, ("BUY", "HIGH"): 20.0,
    ("PASS", "LOW"): 0.0, ("PASS", "MEDIUM"): 0.0, ("PASS", "HIGH"): 0.0,
}


class TestConfigureHook:
    """The ABC gains an optional configure() hook; default is a no-op so
    strategies that don't override it keep working unchanged."""

    def test_abc_has_configure_method(self):
        assert hasattr(ReceiverStrategy, "configure")

    def test_default_configure_is_noop(self):
        # A strategy that does not override configure must still construct.
        s = DirichletBayesianStrategy()
        # No-op default: calling it must not raise and must not change behavior.
        s.configure(prior={"LOW": 0.3, "MEDIUM": 0.4, "HIGH": 0.3},
                    receiver_payoffs=DEFAULT_PAYOFFS)


class TestDefaultsUnchanged:
    """The exact-arithmetic safety net: after refactor, the default-config
    behavior must reproduce the pre-refactor numbers exactly."""

    def test_bayesian_posterior_mean_unchanged(self):
        s = DirichletBayesianStrategy()
        s.update(Signal.GOOD, Action.BUY, AssetQuality.HIGH, 10, 20)
        pm = s._posterior_mean()
        # one update on HIGH/GOOD with uniform prior_strength=1
        assert pm["HIGH"]["GOOD"] == pytest.approx(2 / 4, abs=1e-12)
        assert pm["HIGH"]["BAD"] == pytest.approx(1 / 4, abs=1e-12)

    def test_regret_counterfactual_unchanged(self):
        s = RegretMatchingStrategy()
        s.update(Signal.GOOD, Action.PASS, AssetQuality.HIGH, 0, 0)
        # counterfactual regret of BUY vs PASS on HIGH: 20 - 0 = 20
        assert s._regret["GOOD"]["BUY"] == pytest.approx(20.0, abs=1e-12)


class TestConfigOverride:
    """When configure() is given non-default values, decisions follow them."""

    def test_bayesian_uses_injected_prior_and_payoffs(self):
        s = DirichletBayesianStrategy()
        # Prior concentrated on LOW; GOOD signal very costly to buy.
        s.configure(
            prior={"LOW": 0.9, "MEDIUM": 0.05, "HIGH": 0.05},
            receiver_payoffs={
                ("BUY", "LOW"): -15.0, ("BUY", "MEDIUM"): 5.0, ("BUY", "HIGH"): 20.0,
                ("PASS", "LOW"): 0.0, ("PASS", "MEDIUM"): 0.0, ("PASS", "HIGH"): 0.0,
            },
        )
        s.update(Signal.GOOD, Action.BUY, AssetQuality.LOW, 10, -15)
        # With LOW-dominated prior, even a GOOD signal should yield E[BUY] < 0 → PASS
        assert s.choose_action(Signal.GOOD, 1, 1) == Action.PASS

    def test_regret_uses_injected_payoffs(self):
        s = RegretMatchingStrategy()
        # Make BUY on HIGH worth +99 instead of +20
        payoffs = dict(DEFAULT_PAYOFFS)
        payoffs[("BUY", "HIGH")] = 99.0
        s.configure(prior={"LOW": 0.3, "MEDIUM": 0.4, "HIGH": 0.3}, receiver_payoffs=payoffs)
        s.update(Signal.GOOD, Action.PASS, AssetQuality.HIGH, 0, 0)
        # counterfactual regret of BUY vs PASS on HIGH: 99 - 0 = 99
        assert s._regret["GOOD"]["BUY"] == pytest.approx(99.0, abs=1e-12)
