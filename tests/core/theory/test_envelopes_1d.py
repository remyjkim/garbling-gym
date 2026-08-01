# ABOUTME: Tests for the 1-D concave (cav) and quasiconcave (qcav) envelopes
# ABOUTME: Property tests + exact-shape checks; the KG oracle lives in test_oracles.py

import numpy as np
import pytest
from hypothesis import given, strategies as st

from garbling_gym.core.theory.envelopes import cav, qcav


def _grid(n=101):
    return np.linspace(0.0, 1.0, n)


class TestCavProperties:
    """cav(xs, vs) = the least concave majorant of v over xs."""

    @given(n=st.integers(min_value=5, max_value=200),
           seed=st.integers(min_value=0, max_value=10_000))
    def test_cav_is_a_majorant(self, n, seed):
        # cav >= v pointwise
        rng = np.random.default_rng(seed)
        xs = np.sort(rng.uniform(0, 1, n))
        xs[0], xs[-1] = 0.0, 1.0
        vs = rng.uniform(-5, 5, n)
        out = cav(xs, vs)
        assert np.all(out >= vs - 1e-9)

    @given(n=st.integers(min_value=5, max_value=100),
           seed=st.integers(min_value=0, max_value=10_000))
    def test_cav_is_concave(self, n, seed):
        # For any triple i < j < k, the value at j must be >= linear interpolation
        # of the values at i and k (discrete concavity).
        rng = np.random.default_rng(seed)
        xs = np.sort(rng.uniform(0, 1, n))
        xs[0], xs[-1] = 0.0, 1.0
        vs = rng.uniform(-5, 5, n)
        out = cav(xs, vs)
        for _ in range(50):
            i, k = sorted(rng.integers(0, n, size=2))
            if k - i < 2:
                continue
            j = rng.integers(i + 1, k)
            t = (xs[j] - xs[i]) / (xs[k] - xs[i])
            interp = out[i] + t * (out[k] - out[i])
            assert out[j] >= interp - 1e-9

    def test_cav_of_concave_function_is_itself(self):
        xs = _grid(201)
        vs = -(xs - 0.5) ** 2  # strictly concave
        assert np.allclose(cav(xs, vs), vs, atol=1e-9)

    def test_cav_of_linear_is_itself(self):
        xs = _grid(51)
        vs = 3.0 * xs + 2.0
        assert np.allclose(cav(xs, vs), vs, atol=1e-9)

    def test_cav_preserves_endpoints(self):
        xs = _grid(51)
        vs = np.sin(4 * xs)  # non-monotone, non-concave
        out = cav(xs, vs)
        assert out[0] == pytest.approx(vs[0], abs=1e-9)
        assert out[-1] == pytest.approx(vs[-1], abs=1e-9)

    def test_cav_flat_step_lifts_interior(self):
        # v = 1 on [0.25, 0.75], 0 elsewhere. cav must connect (0,0)->(0.25,1)
        # and (0.75,1)->(1,0) with straight lines (the least concave majorant).
        xs = _grid(401)
        vs = np.where((xs >= 0.25) & (xs <= 0.75), 1.0, 0.0)
        out = cav(xs, vs)
        # At x=0.125 (halfway from 0 to 0.25), cav = 0.5
        idx = np.argmin(np.abs(xs - 0.125))
        assert out[idx] == pytest.approx(0.5, abs=2e-3)


class TestQcavProperties:
    """qcav(xs, vs) = min(prefix_max, suffix_max) — the quasiconcave envelope on a line."""

    @given(n=st.integers(min_value=5, max_value=200),
           seed=st.integers(min_value=0, max_value=10_000))
    def test_qcav_is_a_majorant(self, n, seed):
        rng = np.random.default_rng(seed)
        xs = np.sort(rng.uniform(0, 1, n))
        xs[0], xs[-1] = 0.0, 1.0
        vs = rng.uniform(-5, 5, n)
        out = qcav(xs, vs)
        assert np.all(out >= vs - 1e-9)

    @given(n=st.integers(min_value=5, max_value=100),
           seed=st.integers(min_value=0, max_value=10_000))
    def test_qcav_is_quasiconcave(self, n, seed):
        # A function is quasiconcave iff its superlevel sets are convex.
        # On a 1-D grid that means out is unimodal: non-decreasing up to a mode,
        # then non-increasing.
        rng = np.random.default_rng(seed)
        xs = np.sort(rng.uniform(0, 1, n))
        xs[0], xs[-1] = 0.0, 1.0
        vs = rng.uniform(-5, 5, n)
        out = qcav(xs, vs)
        # Unimodality: the first differences change sign (from + to -) at most once.
        diffs = np.diff(out)
        # Ignore exact-equal neighbors (diff == 0); track sign transitions of strict changes.
        pos = diffs > 1e-12
        neg = diffs < -1e-12
        # Once we see a negative step, no positive step may follow.
        seen_neg = False
        ok = True
        for p, ng in zip(pos, neg):
            if ng:
                seen_neg = True
            if p and seen_neg:
                ok = False
                break
        assert ok, "qcav output is not unimodal (rose again after falling)"

    def test_qcav_preserves_endpoints(self):
        xs = _grid(51)
        vs = np.sin(4 * xs)
        out = qcav(xs, vs)
        assert out[0] == pytest.approx(vs[0], abs=1e-9)
        assert out[-1] == pytest.approx(vs[-1], abs=1e-9)

    def test_qcav_step_up_is_unimodal_at_max(self):
        # v steps 0.2 -> 0.8 at x=0.5. qcav is the smallest quasiconcave majorant:
        # it leaves the left plateau at 0.2 (already quasiconcave-compatible there)
        # and raises the right side to 0.8. Result is monotone non-decreasing
        # (a valid quasiconcave function) and a majorant of v.
        xs = _grid(101)
        vs = np.where(xs < 0.5, 0.2, 0.8)
        out = qcav(xs, vs)
        # majorant
        assert np.all(out >= vs - 1e-12)
        # left half stays at 0.2 (prefix_max=0.2 there, suffix_max=0.8, min=0.2)
        assert np.allclose(out[xs < 0.5], 0.2, atol=1e-12)
        # right half is 0.8
        assert np.allclose(out[xs >= 0.5], 0.8, atol=1e-12)


class TestCavQcavRelationship:
    def test_qcav_le_cav_for_random_functions(self):
        # The quasiconcave envelope is always <= the concave envelope.
        rng = np.random.default_rng(42)
        xs = _grid(101)
        vs = rng.uniform(-5, 5, 101)
        assert np.all(qcav(xs, vs) <= cav(xs, vs) + 1e-9)
