# ABOUTME: The four closed-form oracle cases that validate the theory solver
# ABOUTME: (Proposal 08 §4). Each must pass to <= 1e-3. This file accumulates all four.

import numpy as np
import pytest

from garbling_gym.core.theory.envelopes import cav, qcav


class TestOracleKGProsecutorJudge:
    """Oracle 1: Kamenica & Gentzkow (2011) prosecutor-judge.

    Binary state; prior mu0 = 0.3 (probability of guilty/HIGH); the receiver
    convicts (BUY) iff posterior > 0.5; the sender wants conviction (payoff 1
    on BUY, 0 otherwise). The full-commitment (Bayesian-persuasion) value is
    the concave envelope of the step value function at the prior:

        V_cav(0.3) = cav(v)(0.3) = 0.60.

    This is the canonical worked example, computed for the literature-review
    figure; the solver must reproduce it.
    """

    MU0 = 0.3
    THRESHOLD = 0.5
    EXPECTED_V_CAV = 0.60

    def test_oracle_kg_prosecutor_judge(self):
        xs = np.linspace(0.0, 1.0, 2001)
        # sender value: 1 iff the receiver convicts (posterior > threshold)
        vs = (xs > self.THRESHOLD).astype(float)
        envelope = cav(xs, vs)
        idx = int(np.argmin(np.abs(xs - self.MU0)))
        v_cav = envelope[idx]
        assert v_cav == pytest.approx(self.EXPECTED_V_CAV, abs=1e-3), (
            f"KG oracle failed: cav({self.MU0}) = {v_cav}, expected {self.EXPECTED_V_CAV}"
        )

    def test_value_of_commitment_is_positive(self):
        # The cheap-talk (qcav) value at the prior must be 0 (the receiver never
        # convicts at prior 0.3 < 0.5 under babbling), while cav = 0.60.
        # So the value of commitment is 0.60 — the entire point of KG.
        xs = np.linspace(0.0, 1.0, 2001)
        vs = (xs > self.THRESHOLD).astype(float)
        idx = int(np.argmin(np.abs(xs - self.MU0)))
        v_qcav = qcav(xs, vs)[idx]
        v_cav = cav(xs, vs)[idx]
        assert v_qcav == pytest.approx(0.0, abs=1e-3)
        assert v_cav - v_qcav == pytest.approx(0.60, abs=1e-3)
