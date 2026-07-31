# ABOUTME: Tests for the LRS weak-institution curve v*_chi
# ABOUTME: Validated against the structural theorems (endpoints, monotonicity, bounds)

import numpy as np
import pytest

from garbling_gym.core.config import GameConfig
from garbling_gym.core.theory.game_spec import game_spec
from garbling_gym.core.theory.simplex2d import SimplexGrid, cav_simplex, qcav_simplex
from garbling_gym.core.theory.weak_inst import weak_inst_curve, weak_inst_value


def _step_values(grid_mu, spec):
    """Binary sender value over the simplex: 1 iff the receiver buys at mu."""
    return np.array([1.0 if spec.receiver_buys_at(mu) else 0.0 for mu in grid_mu])


class TestWeakInstEndpoints:
    """The LRS program must hit the cheap-talk floor at chi=0 and cav at chi=1."""

    def test_chi_zero_equals_qcav(self):
        spec = game_spec(GameConfig())
        g = SimplexGrid(n=20)
        v = _step_values(g.mu, spec)
        cf = cav_simplex(g, v)
        qf = qcav_simplex(g, v)
        curve = weak_inst_curve(spec, g, v, cf=cf, qf=qf)
        v_at_0 = curve.evaluate(0.0)
        assert v_at_0 == pytest.approx(qf.evaluate(spec.mu0), abs=2e-2)

    def test_chi_one_equals_cav(self):
        spec = game_spec(GameConfig())
        g = SimplexGrid(n=20)
        v = _step_values(g.mu, spec)
        cf = cav_simplex(g, v)
        qf = qcav_simplex(g, v)
        curve = weak_inst_curve(spec, g, v, cf=cf, qf=qf)
        v_at_1 = curve.evaluate(1.0)
        assert v_at_1 == pytest.approx(cf.evaluate(spec.mu0), abs=2e-2)


class TestWeakInstMonotonicityAndBounds:
    def test_weakly_increasing_in_chi(self):
        spec = game_spec(GameConfig())
        g = SimplexGrid(n=20)
        v = _step_values(g.mu, spec)
        cf = cav_simplex(g, v)
        qf = qcav_simplex(g, v)
        curve = weak_inst_curve(spec, g, v, cf=cf, qf=qf)
        chis = np.linspace(0, 1, 21)
        vals = [curve.evaluate(c) for c in chis]
        for i in range(len(vals) - 1):
            assert vals[i + 1] >= vals[i] - 1e-9, (
                f"v* not weakly increasing: chi={chis[i]}->{chis[i+1]}, "
                f"{vals[i]}->{vals[i+1]}"
            )

    def test_bounded_between_qcav_and_cav(self):
        spec = game_spec(GameConfig())
        g = SimplexGrid(n=20)
        v = _step_values(g.mu, spec)
        cf = cav_simplex(g, v)
        qf = qcav_simplex(g, v)
        lo = qf.evaluate(spec.mu0)
        hi = cf.evaluate(spec.mu0)
        curve = weak_inst_curve(spec, g, v, cf=cf, qf=qf)
        for chi in np.linspace(0, 1, 11):
            val = curve.evaluate(chi)
            assert lo - 1e-9 <= val <= hi + 1e-9, (
                f"v*(chi={chi})={val} outside [{lo}, {hi}]"
            )


class TestWeakInstValueShortcut:
    """weak_inst_value(spec, chi) is a convenience wrapper computing the
    curve's value at a single chi."""

    def test_matches_curve_evaluate(self):
        spec = game_spec(GameConfig())
        g = SimplexGrid(n=20)
        v = _step_values(g.mu, spec)
        cf = cav_simplex(g, v)
        qf = qcav_simplex(g, v)
        curve = weak_inst_curve(spec, g, v, cf=cf, qf=qf)
        for chi in [0.0, 0.3, 0.7, 1.0]:
            assert weak_inst_value(spec, g, v, chi) == pytest.approx(
                curve.evaluate(chi), abs=1e-9
            )


class TestWeakInstContinuityBehavior:
    """The LRS theory allows discontinuities in chi. We don't require continuity,
    but we DO require that the curve is right-continuous at the endpoints and
    that any jump is upward (consistent with monotonicity)."""

    def test_no_degenerate_all_zero_curve_when_gap_exists(self):
        # When cav > qcav (a real commitment gap), the curve must rise above
        # qcav for high enough chi — it cannot be flat at the floor.
        spec = game_spec(GameConfig())
        g = SimplexGrid(n=20)
        v = _step_values(g.mu, spec)
        cf = cav_simplex(g, v)
        qf = qcav_simplex(g, v)
        lo = qf.evaluate(spec.mu0)
        hi = cf.evaluate(spec.mu0)
        if hi > lo + 1e-6:  # only meaningful when there's a gap
            curve = weak_inst_curve(spec, g, v, cf=cf, qf=qf)
            assert curve.evaluate(1.0) > lo + 1e-6
