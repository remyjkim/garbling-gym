# ABOUTME: Tests for game_spec — converting GameConfig into the solver's numpy view
# ABOUTME: Verifies prior/payoff extraction, ordering, and the sender indirect value

import pytest
import numpy as np

from garbling_gym.core.config import GameConfig
from garbling_gym.core.payoffs import PayoffStructure
from garbling_gym.core.types import Action, AssetQuality
from garbling_gym.core.theory.game_spec import game_spec, GameSpec


class TestGameSpecDefault:
    def test_prior_shape_and_sum(self):
        spec = game_spec(GameConfig())
        assert spec.mu0.shape == (3,)
        assert spec.mu0.sum() == pytest.approx(1.0, abs=1e-12)

    def test_prior_ordered_low_medium_high(self):
        spec = game_spec(GameConfig())
        # Order must be LOW, MEDIUM, HIGH (AssetQuality.value 0,1,2)
        assert spec.mu0.tolist() == [0.3, 0.4, 0.3]

    def test_receiver_buy_payoffs(self):
        spec = game_spec(GameConfig())
        assert spec.receiver_buy.shape == (3,)
        assert spec.receiver_buy.tolist() == [-15.0, 5.0, 20.0]

    def test_sender_buy_is_state_independent(self):
        # The gym's defining assumption: +10 on BUY regardless of state.
        spec = game_spec(GameConfig())
        assert spec.sender_buy.shape == (3,)
        assert np.all(spec.sender_buy == 10.0)

    def test_buy_threshold_is_zero(self):
        # Receiver buys iff E[receiver_buy | posterior] > 0
        spec = game_spec(GameConfig())
        assert spec.buy_threshold == 0.0

    def test_n_states(self):
        assert game_spec(GameConfig()).n_states == 3


class TestGameSpecCustomConfig:
    def test_custom_prior_is_respected(self):
        config = GameConfig(prior={
            AssetQuality.LOW: 0.5,
            AssetQuality.MEDIUM: 0.3,
            AssetQuality.HIGH: 0.2,
        })
        spec = game_spec(config)
        assert spec.mu0.tolist() == [0.5, 0.3, 0.2]
        assert spec.mu0.sum() == pytest.approx(1.0, abs=1e-12)

    def test_custom_payoffs_are_respected(self):
        custom = PayoffStructure(payoffs={
            Action.BUY: {
                AssetQuality.LOW: (7, -99.0),
                AssetQuality.MEDIUM: (7, 1.0),
                AssetQuality.HIGH: (7, 50.0),
            },
            Action.PASS: {
                AssetQuality.LOW: (0, 0), AssetQuality.MEDIUM: (0, 0), AssetQuality.HIGH: (0, 0),
            },
        })
        spec = game_spec(GameConfig(payoffs=custom))
        assert spec.receiver_buy.tolist() == [-99.0, 1.0, 50.0]
        assert spec.sender_buy.tolist() == [7.0, 7.0, 7.0]


class TestIndirectSenderValue:
    """v_S(mu) = sender_buy if receiver buys at mu, else 0.

    The receiver buys iff E[receiver_buy | mu] > threshold.
    With the default payoffs, E[BUY | prior] = 0.3*(-15) + 0.4*5 + 0.3*20 = 3.5 > 0,
    so the receiver buys at the prior.
    """

    def test_receiver_buys_at_default_prior(self):
        spec = game_spec(GameConfig())
        assert spec.receiver_buys_at(spec.mu0) is True

    def test_sender_value_positive_when_receiver_buys(self):
        spec = game_spec(GameConfig())
        assert spec.indirect_sender_value(spec.mu0) == pytest.approx(10.0, abs=1e-12)

    def test_sender_value_zero_when_low_dominated_posterior(self):
        # A posterior concentrated on LOW: E[BUY] = -15 < 0 → receiver passes → sender gets 0
        spec = game_spec(GameConfig())
        low_only = np.array([1.0, 0.0, 0.0])
        assert spec.receiver_buys_at(low_only) is False
        assert spec.indirect_sender_value(low_only) == pytest.approx(0.0, abs=1e-12)

    def test_sender_value_step_function_at_threshold(self):
        # Find the breakeven posterior on the LOW-HIGH edge and check the step.
        # E[BUY] = mu_low*(-15) + mu_high*20 = 0  (mu_med=0)
        # => 15*mu_low = 20*(1 - mu_low) => mu_low = 20/35
        spec = game_spec(GameConfig())
        mu_low = 20.0 / 35.0
        edge = np.array([mu_low, 0.0, 1.0 - mu_low])
        # Exactly at threshold (==), receiver does NOT buy (strict > threshold)
        just_below = edge * np.array([1.0001, 1.0, 0.9999])
        just_below = just_below / just_below.sum()
        assert spec.indirect_sender_value(just_below) == pytest.approx(0.0, abs=1e-9)
        just_above = edge * np.array([0.9999, 1.0, 1.0001])
        just_above = just_above / just_above.sum()
        assert spec.indirect_sender_value(just_above) == pytest.approx(10.0, abs=1e-9)
