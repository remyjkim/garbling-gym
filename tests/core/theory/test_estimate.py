# ABOUTME: Tests for empirical channel estimation from game history
# ABOUTME: L_hat(s|theta) via Laplace smoothing; Dirichlet credible interval

import numpy as np
import pytest

from garbling_gym.core.theory.estimate import estimate_channel, ChannelEstimate


def _history(pairs):
    """Build a minimal history list of dicts like GameResults.history entries.

    pairs: list of (quality_name, signal_name).
    """
    return [
        {"quality": q, "signal": s, "action": "BUY",
         "sender_payoff": 0.0, "receiver_payoff": 0.0}
        for q, s in pairs
    ]


class TestEstimateChannel:
    def test_recovers_identity_from_identity_run(self):
        # Every round: signal == quality (perfect revelation). With Laplace
        # smoothing the diagonal is (n+1)/(n+3) and off-diagonal is 1/(n+3),
        # approaching identity as n grows.
        pairs = [("LOW", "BAD"), ("MEDIUM", "NEUTRAL"), ("HIGH", "GOOD")] * 30  # n=30
        est = estimate_channel(_history(pairs))
        assert np.allclose(np.diag(est.L), 31 / 33, atol=1e-9)
        off_diag = est.L[~np.eye(3, dtype=bool)]
        assert np.allclose(off_diag, 1 / 33, atol=1e-9)
        # And it converges to identity with much more data
        pairs_long = [("LOW", "BAD"), ("MEDIUM", "NEUTRAL"), ("HIGH", "GOOD")] * 1000
        est_long = estimate_channel(_history(pairs_long))
        assert np.allclose(est_long.L, np.eye(3), atol=2e-3)

    def test_recovers_uniform_from_random_run(self):
        # Each quality produces each signal equally often (10 each) -> L = 1/3.
        pairs = []
        for q in ["LOW", "MEDIUM", "HIGH"]:
            for s in ["BAD", "NEUTRAL", "GOOD"]:
                pairs.extend([(q, s)] * 10)
        est = estimate_channel(_history(pairs))
        assert np.allclose(est.L, np.ones((3, 3)) / 3, atol=1e-9)

    def test_row_stochastic(self):
        pairs = [("LOW", "BAD"), ("MEDIUM", "GOOD"), ("LOW", "NEUTRAL"),
                 ("HIGH", "GOOD"), ("HIGH", "BAD"), ("MEDIUM", "NEUTRAL")] * 5
        est = estimate_channel(_history(pairs))
        assert np.allclose(est.L.sum(axis=1), 1.0, atol=1e-12)
        assert est.L.shape == (3, 3)

    def test_ci_width_shrinks_with_more_data(self):
        # More data per state => narrower credible intervals. Feed all three
        # states so every row is informed (unobserved rows don't shrink).
        short = [("LOW", "BAD"), ("MEDIUM", "NEUTRAL"), ("HIGH", "GOOD")] * 3
        long = [("LOW", "BAD"), ("MEDIUM", "NEUTRAL"), ("HIGH", "GOOD")] * 300
        est_short = estimate_channel(_history(short))
        est_long = estimate_channel(_history(long))
        assert est_long.ci_width.max() < est_short.ci_width.max()

    def test_laplace_smoothing_on_empty_history(self):
        # No observations: Laplace smoothing yields uniform 1/3 everywhere.
        est = estimate_channel(_history([]))
        assert np.allclose(est.L, np.ones((3, 3)) / 3, atol=1e-12)

    def test_laplace_smoothing_partial_history(self):
        # Only LOW observed, always BAD: L[LOW] = (n+1)/(n+3) for BAD,
        # 1/(n+3) for the others.
        pairs = [("LOW", "BAD")] * 9   # n=9
        est = estimate_channel(_history(pairs))
        # LOW row: BAD = 10/12, others = 1/12
        assert est.L[0, 0] == pytest.approx(10 / 12, abs=1e-12)
        assert est.L[0, 1] == pytest.approx(1 / 12, abs=1e-12)
        assert est.L[0, 2] == pytest.approx(1 / 12, abs=1e-12)
        # Unobserved states (MEDIUM, HIGH): uniform 1/3
        assert np.allclose(est.L[1], [1 / 3, 1 / 3, 1 / 3], atol=1e-12)
        assert np.allclose(est.L[2], [1 / 3, 1 / 3, 1 / 3], atol=1e-12)

    def test_channel_estimate_has_mi(self):
        # The estimate carries a mutual-information field computed via metrics.
        # With enough data, a near-identity run yields MI close to log(3).
        pairs = [("LOW", "BAD"), ("MEDIUM", "NEUTRAL"), ("HIGH", "GOOD")] * 2000
        est = estimate_channel(_history(pairs), mu0=np.array([1 / 3, 1 / 3, 1 / 3]))
        assert est.mutual_information == pytest.approx(np.log(3), abs=2e-2)
        # And a uniform run yields MI near 0
        uni = []
        for q in ["LOW", "MEDIUM", "HIGH"]:
            for s in ["BAD", "NEUTRAL", "GOOD"]:
                uni.extend([(q, s)] * 50)
        est_u = estimate_channel(_history(uni), mu0=np.array([1 / 3, 1 / 3, 1 / 3]))
        assert est_u.mutual_information == pytest.approx(0.0, abs=1e-9)
