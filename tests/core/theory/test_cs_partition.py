# ABOUTME: Tests for the Crawford-Sobel partition utility N(b) and cutoffs
# ABOUTME: Oracle: N(b) = largest N with 2N(N-1)b < 1; collapse 7->5->3->2->1

import pytest

from garbling_gym.core.theory.cs_partition import max_partition_count, cs_cutoffs


class TestMaxPartitionCount:
    """N(b) = largest integer N >= 1 with 2*N*(N-1)*b < 1."""

    def test_oracle_collapse_sequence(self):
        # The proposal's documented collapse over the standard bias grid.
        for b, expected in [(0.01, 7), (0.02, 5), (0.05, 3), (0.10, 2), (0.25, 1)]:
            got = max_partition_count(b)
            assert got == expected, f"N({b}) = {got}, expected {expected}"

    def test_babbling_for_large_bias(self):
        # b > 1/4 => only babbling (N=1) survives (2*1*0*b = 0 < 1 always holds
        # for N=1, but N=2 requires 2*2*1*b = 4b < 1 => b < 0.25).
        assert max_partition_count(0.30) == 1
        assert max_partition_count(0.50) == 1
        assert max_partition_count(1.0) == 1

    def test_fully_informative_for_tiny_bias(self):
        # As b -> 0, N grows without bound; check a few.
        assert max_partition_count(0.001) >= 20

    def test_n1_always_feasible(self):
        # N=1 (babbling) is always feasible: 2*1*0*b = 0 < 1.
        for b in [0.0, 0.5, 1.0, 100.0]:
            assert max_partition_count(b) >= 1

    def test_monotone_decreasing_in_b(self):
        bs = [0.005 * k for k in range(1, 60)]
        counts = [max_partition_count(b) for b in bs]
        for i in range(len(counts) - 1):
            assert counts[i + 1] <= counts[i]


class TestCsCutoffs:
    """Cutoffs satisfy the Crawford-Sobel recurrence a_{i+1} = 2 a_i - a_{i-1} + 4b."""

    def test_recurrence_holds(self):
        b = 0.05
        N = max_partition_count(b)
        cuts = cs_cutoffs(b, N)
        # a_0 = 0, a_N = 1
        assert cuts[0] == pytest.approx(0.0, abs=1e-12)
        assert cuts[-1] == pytest.approx(1.0, abs=1e-9)
        # recurrence
        for i in range(1, len(cuts) - 1):
            expected = 2 * cuts[i] - cuts[i - 1] + 4 * b
            assert cuts[i + 1] == pytest.approx(expected, abs=1e-9)

    def test_cutoffs_partition_unit_interval(self):
        b = 0.02
        N = max_partition_count(b)
        cuts = cs_cutoffs(b, N)
        assert len(cuts) == N + 1
        # strictly increasing within [0,1]
        for i in range(len(cuts) - 1):
            assert cuts[i + 1] > cuts[i]

    def test_babbling_single_partition(self):
        # b large -> N=1 -> cutoffs are just [0, 1]
        cuts = cs_cutoffs(0.5, 1)
        assert len(cuts) == 2
        assert cuts[0] == pytest.approx(0.0, abs=1e-12)
        assert cuts[1] == pytest.approx(1.0, abs=1e-12)
