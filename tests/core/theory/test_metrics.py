# ABOUTME: Tests for information metrics: mutual information and the Blackwell garbling test
# ABOUTME: MI uses nats; the Blackwell test is an LP feasibility problem.

import numpy as np
import pytest

from garbling_gym.core.theory.metrics import mutual_information, is_garbling_of


class TestMutualInformation:
    """I(theta; s) = sum_theta mu0(theta) sum_s L[s|theta] log(L[s|theta] / p(s))."""

    def test_identity_channel_is_entropy(self):
        L = np.eye(3)
        mu0 = np.array([1 / 3, 1 / 3, 1 / 3])
        # I(theta; theta) = H(theta) = log(3) nats
        assert mutual_information(L, mu0) == pytest.approx(np.log(3), abs=1e-10)

    def test_uniform_channel_is_zero(self):
        L = np.ones((3, 3)) / 3
        mu0 = np.array([1 / 3, 1 / 3, 1 / 3])
        assert mutual_information(L, mu0) == pytest.approx(0.0, abs=1e-12)

    def test_nonnegative_for_random_channels(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            # random row-stochastic L
            L = rng.dirichlet(np.ones(3), size=3)
            mu0 = rng.dirichlet(np.ones(3))
            assert mutual_information(L, mu0) >= -1e-12

    def test_bounded_by_entropy(self):
        # I(theta;s) <= H(theta) always.
        L = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
        mu0 = np.array([0.2, 0.5, 0.3])
        h = -np.sum(mu0 * np.log(mu0))
        assert mutual_information(L, mu0) <= h + 1e-9

    def test_zero_when_signal_uninformative_of_state(self):
        # All rows identical => signal independent of state => I = 0.
        L = np.tile([0.5, 0.3, 0.2], (3, 1))
        mu0 = np.array([0.4, 0.4, 0.2])
        assert mutual_information(L, mu0) == pytest.approx(0.0, abs=1e-12)


class TestIsGarblingOf:
    """is_garbling_of(L_prime, L) = exists row-stochastic K with L_prime = L @ K."""

    def test_identity_dominates_noise(self):
        # Noise is a garbling of identity (add state-independent uniform noise).
        L_better = np.eye(3)
        L_worse = np.ones((3, 3)) / 3
        assert is_garbling_of(L_worse, L_better) is True
        assert is_garbling_of(L_better, L_worse) is False

    def test_reflexive(self):
        # Every channel is a garbling of itself (K = identity).
        L = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        assert is_garbling_of(L, L) is True

    def test_more_informative_garbled_into_less(self):
        # A noisy version of an informative channel: L' = L @ K for a noisy K.
        L = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        K = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
        L_prime = L @ K
        assert is_garbling_of(L_prime, L) is True

    def test_incomparable_channels(self):
        # Two channels that resolve different states but neither dominates.
        # Binary: L1 is informative about state 0, L2 about state 1, neither
        # can be obtained from the other by state-independent post-processing.
        L1 = np.array([[0.9, 0.1], [0.5, 0.5]])
        L2 = np.array([[0.5, 0.5], [0.1, 0.9]])
        assert is_garbling_of(L2, L1) is False
        assert is_garbling_of(L1, L2) is False

    def test_permutation_channels_are_garblings(self):
        # Permuting signals is a valid stochastic kernel, so permuted channels
        # are Blackwell-equivalent (each is a garbling of the other).
        L = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        L_perm = np.array([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])  # signal permute
        assert is_garbling_of(L_perm, L) is True
        assert is_garbling_of(L, L_perm) is True

    def test_identity_preserved_under_trivial_kernel(self):
        # K = identity => L' = L
        L = np.eye(3)
        assert is_garbling_of(L @ np.eye(3), L) is True
