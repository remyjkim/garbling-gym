# ABOUTME: Tests for the benchmark bundle assembler
# ABOUTME: Oracle 4 (LR <= KG) enforced here; bundle is the solver's public API.

import numpy as np
import pytest

from garbling_gym.core.config import GameConfig
from garbling_gym.core.theory.benchmarks import compute_benchmarks, BenchmarkBundle


class TestBenchmarkBundleFields:
    def test_bundle_has_all_fields(self):
        b = compute_benchmarks(GameConfig())
        for key in [
            "babbling",
            "cav",
            "qcav",
            "weak_inst",
            "floor",
            "ceiling",
            "value_of_commitment",
        ]:
            assert key in b.as_dict(), f"missing benchmark field: {key}"

    def test_weak_inst_is_a_curve(self):
        b = compute_benchmarks(GameConfig(), chi_grid=[0.0, 0.5, 1.0])
        assert isinstance(b.weak_inst, dict)
        assert set(b.weak_inst.keys()) == {0.0, 0.5, 1.0}

    def test_babbling_is_sender_value_at_prior_no_info(self):
        # Babbling: receiver's posterior == prior; sender gets the prior value.
        b = compute_benchmarks(GameConfig())
        spec_mu0 = np.array([0.3, 0.4, 0.3])
        receiver_payoff = np.array([-15.0, 5.0, 20.0])
        # Receiver buys at the prior iff E[BUY|prior] > 0 (it's 3.5 > 0).
        # Sender gets +10 (state-independent) when the receiver buys.
        assert b.babbling == pytest.approx(10.0, abs=1e-9)


class TestOracleLRLeCav:
    """Oracle 4: V_qcav(mu0) <= V_cav(mu0) pointwise."""

    def test_qcav_le_cav_default_config(self):
        b = compute_benchmarks(GameConfig())
        assert b.qcav <= b.cav + 1e-9

    def test_qcav_le_cav_across_priors(self):
        for low, med, high in [(0.2, 0.3, 0.5), (0.5, 0.3, 0.2), (0.1, 0.8, 0.1), (0.34, 0.33, 0.33)]:
            from garbling_gym.core.types import AssetQuality
            cfg = GameConfig(prior={
                AssetQuality.LOW: low, AssetQuality.MEDIUM: med, AssetQuality.HIGH: high
            })
            b = compute_benchmarks(cfg)
            assert b.qcav <= b.cav + 1e-9, (
                f"qcav > cav at prior ({low},{med},{high}): {b.qcav} > {b.cav}"
            )


class TestValueOfCommitment:
    def test_value_of_commitment_is_cav_minus_qcav(self):
        b = compute_benchmarks(GameConfig())
        assert b.value_of_commitment == pytest.approx(b.cav - b.qcav, abs=1e-12)

    def test_value_of_commitment_nonneg(self):
        b = compute_benchmarks(GameConfig())
        assert b.value_of_commitment >= -1e-9


class TestFloorCeiling:
    def test_floor_and_ceiling_derived_from_payoffs(self):
        # floor = receiver's no-information payoff (always PASS => 0);
        # ceiling = receiver's perfect-information payoff under the prior.
        b = compute_benchmarks(GameConfig())
        # Perfect-info receiver buys only MEDIUM/HIGH: E = 0.4*5 + 0.3*20 = 8.0
        assert b.ceiling == pytest.approx(0.4 * 5 + 0.3 * 20, abs=1e-9)
        assert b.floor == pytest.approx(0.0, abs=1e-9)


class TestBundleSerializable:
    def test_as_dict_is_json_compatible(self):
        import json

        b = compute_benchmarks(GameConfig(), chi_grid=[0.0, 1.0])
        d = b.as_dict()
        # Must be JSON-serializable (it gets stored in results.json).
        json.dumps(d)
