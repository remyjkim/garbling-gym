# ABOUTME: Tests for the 2-D (3-state) simplex concave & quasiconcave envelopes
# ABOUTME: Property tests + boundary-collapse agreement with the 1-D solver

import numpy as np
import pytest
from hypothesis import given, strategies as st

from garbling_gym.core.theory.simplex2d import (
    SimplexGrid,
    cav_simplex,
    qcav_simplex,
)


def _step_value(grid_mu, receiver_payoff, threshold=0.0):
    """Binary sender value: 1 iff receiver buys at mu."""
    return (grid_mu @ receiver_payoff > threshold).astype(float)


class TestSimplexGrid:
    def test_grid_covers_simplex(self):
        g = SimplexGrid(n=10)
        # All grid points are in the simplex
        assert np.all(g.mu >= -1e-12)
        assert np.all(g.mu.sum(axis=1) == pytest.approx(1.0, abs=1e-12))
        # Number of points = (n+1)(n+2)/2
        assert g.mu.shape == (((10 + 1) * (10 + 2)) // 2, 3)

    def test_grid_includes_vertices(self):
        g = SimplexGrid(n=10)
        # The three vertices LOW=(1,0,0), MEDIUM=(0,1,0), HIGH=(0,0,1) are present
        for v in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
            assert np.any(np.all(np.isclose(g.mu, v, atol=1e-12), axis=1))


class TestCavSimplexProperties:
    """cav_simplex(mu0) >= v(mu0); cav is concave; endpoints collapse to 1-D."""

    def test_cav_is_majorant_at_grid_points(self):
        rp = np.array([-15.0, 5.0, 20.0])
        g = SimplexGrid(n=20)
        v = _step_value(g.mu, rp)
        cf = cav_simplex(g, v)
        for i, mu0 in enumerate(g.mu):
            assert cf.evaluate(mu0) >= v[i] - 1e-9

    def test_cav_preserves_a_linear_function(self):
        # A linear v is already concave; cav == v everywhere.
        g = SimplexGrid(n=15)
        a = np.array([2.0, -1.0, 3.0])
        v = g.mu @ a
        cf = cav_simplex(g, v)
        for mu0 in [(0.3, 0.4, 0.3), (0.1, 0.1, 0.8), (1.0, 0.0, 0.0)]:
            assert cf.evaluate(np.array(mu0)) == pytest.approx(np.array(mu0) @ a, abs=1e-9)

    def test_cav_at_vertex_equals_v_at_vertex(self):
        rp = np.array([-15.0, 5.0, 20.0])
        g = SimplexGrid(n=20)
        v = _step_value(g.mu, rp)
        cf = cav_simplex(g, v)
        # At each pure-state vertex, cav = v (no splitting possible)
        for i, mu0 in enumerate([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]):
            idx = np.argmax([np.linalg.norm(g.mu[k] - np.array(mu0)) for k in range(len(g.mu))])
            # find the actual grid index of this vertex
            grid_idx = np.where(np.all(np.isclose(g.mu, mu0, atol=1e-12), axis=1))[0][0]
            assert cf.evaluate(np.array(mu0)) == pytest.approx(v[grid_idx], abs=1e-9)

    def test_cav_boundary_collapse_matches_1d(self):
        """On the edge mu_medium=0, 2-D cav must equal the validated 1-D cav."""
        from garbling_gym.core.theory.envelopes import cav as cav_1d
        rp = np.array([-15.0, 5.0, 20.0])
        n = 25
        g = SimplexGrid(n=n)
        v = _step_value(g.mu, rp)
        cf = cav_simplex(g, v)
        xs1d = np.linspace(0, 1, n + 1)
        vs1d = np.array([1.0 if np.array([x, 0.0, 1 - x]) @ rp > 0 else 0.0 for x in xs1d])
        cav1 = cav_1d(xs1d, vs1d)
        for x, c1 in zip(xs1d, cav1):
            c2 = cf.evaluate(np.array([x, 0.0, 1 - x]))
            assert c2 == pytest.approx(c1, abs=2e-3), f"mismatch at x={x}: 2d={c2}, 1d={c1}"


class TestQcavSimplexProperties:
    def test_qcav_is_majorant_at_grid_points(self):
        rp = np.array([-15.0, 5.0, 20.0])
        g = SimplexGrid(n=20)
        v = _step_value(g.mu, rp)
        qf = qcav_simplex(g, v)
        for i, mu0 in enumerate(g.mu):
            assert qf.evaluate(mu0) >= v[i] - 1e-9

    def test_qcav_le_cav(self):
        rp = np.array([-15.0, 5.0, 20.0])
        g = SimplexGrid(n=20)
        v = _step_value(g.mu, rp)
        cf = cav_simplex(g, v)
        qf = qcav_simplex(g, v)
        for mu0 in g.mu:
            assert qf.evaluate(mu0) <= cf.evaluate(mu0) + 1e-9

    def test_qcav_boundary_collapse_matches_1d(self):
        from garbling_gym.core.theory.envelopes import qcav as qcav_1d
        rp = np.array([-15.0, 5.0, 20.0])
        n = 25
        g = SimplexGrid(n=n)
        v = _step_value(g.mu, rp)
        qf = qcav_simplex(g, v)
        xs1d = np.linspace(0, 1, n + 1)
        vs1d = np.array([1.0 if np.array([x, 0.0, 1 - x]) @ rp > 0 else 0.0 for x in xs1d])
        qc1 = qcav_1d(xs1d, vs1d)
        for x, q1 in zip(xs1d, qc1):
            q2 = qf.evaluate(np.array([x, 0.0, 1 - x]))
            assert q2 == pytest.approx(q1, abs=2e-3), f"mismatch at x={x}: 2d={q2}, 1d={q1}"


class TestEvaluateArbitrary:
    """The evaluators must accept beliefs that are NOT on the grid."""

    def test_cav_off_grid(self):
        rp = np.array([-15.0, 5.0, 20.0])
        g = SimplexGrid(n=15)
        v = _step_value(g.mu, rp)
        cf = cav_simplex(g, v)
        # (0.27, 0.41, 0.32) is not on the n=15 grid
        mu0 = np.array([0.27, 0.41, 0.32])
        val = cf.evaluate(mu0)
        assert isinstance(val, float) and 0.0 - 1e-9 <= val <= 1.0 + 1e-9

    def test_qcav_off_grid(self):
        rp = np.array([-15.0, 5.0, 20.0])
        g = SimplexGrid(n=15)
        v = _step_value(g.mu, rp)
        qf = qcav_simplex(g, v)
        mu0 = np.array([0.27, 0.41, 0.32])
        val = qf.evaluate(mu0)
        assert isinstance(val, float) and 0.0 - 1e-9 <= val <= 1.0 + 1e-9


class TestSimplexOracles:
    """Closed-form 2-D checks that validate the INTERIOR computation
    (boundary-collapse only validates the edges)."""

    def test_cav_of_vertex_indicator_is_mass_to_that_vertex(self):
        """v = 1 only at the HIGH vertex (0,0,1), 0 elsewhere. The concave
        envelope at a belief mu is mu_HIGH (the largest weight you can move
        to the HIGH vertex under Bayes plausibility). So cav(mu0) = mu0[HIGH].

        This is a clean interior oracle: cav(mu) = mu[2] analytically."""
        g = SimplexGrid(n=20)
        v = np.zeros(len(g.mu))
        high_idx = int(np.where(np.all(np.isclose(g.mu, [0, 0, 1], atol=1e-12), axis=1))[0][0])
        v[high_idx] = 1.0
        cf = cav_simplex(g, v)
        for mu0 in [(0.3, 0.4, 0.3), (0.1, 0.2, 0.7), (0.0, 0.5, 0.5), (0.5, 0.5, 0.0)]:
            got = cf.evaluate(np.array(mu0))
            expected = mu0[2]
            assert got == pytest.approx(expected, abs=5e-3), (
                f"cav{mu0} = {got}, expected mu_HIGH = {expected}"
            )

    def test_qcav_of_vertex_indicator_is_zero_off_vertex(self):
        """Same v (1 only at HIGH vertex). qcav(mu) = 1 iff mu is in the convex
        hull of {HIGH} = {HIGH} itself, i.e. only at the pure-HIGH belief.
        Everywhere else qcav = 0 (the floor). Verifies qcav's strictness vs cav."""
        g = SimplexGrid(n=20)
        v = np.zeros(len(g.mu))
        high_idx = int(np.where(np.all(np.isclose(g.mu, [0, 0, 1], atol=1e-12), axis=1))[0][0])
        v[high_idx] = 1.0
        qf = qcav_simplex(g, v)
        # Off the HIGH vertex, qcav is 0
        assert qf.evaluate(np.array([0.3, 0.4, 0.3])) == pytest.approx(0.0, abs=1e-9)
        assert qf.evaluate(np.array([0.0, 0.0, 1.0])) == pytest.approx(1.0, abs=1e-9)

