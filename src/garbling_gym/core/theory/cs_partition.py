# ABOUTME: Crawford-Sobel (1982) partition equilibrium utilities
# ABOUTME: N(b) and the uniform-quadratic cutoff recurrence.

import numpy as np

__all__ = ["max_partition_count", "cs_cutoffs"]


def max_partition_count(b: float) -> int:
    """Largest number of partition intervals in a Crawford-Sobel equilibrium.

    For the uniform-quadratic Crawford-Sobel game with sender bias ``b``, an
    equilibrium with ``N`` actions exists iff the partition with ``N``
    intervals is feasible, which holds iff

        2 * N * (N - 1) * b < 1.

    This returns the largest such integer ``N >= 1``.  As ``b -> 0`` it grows
    without bound (communication becomes arbitrarily precise); for ``b >= 1/4``
    only babbling (``N = 1``) survives.

    Args:
        b: sender bias (preference misalignment).

    Returns:
        The maximum partition count ``N(b)``.
    """
    if b <= 0.0:
        # b = 0 is the aligned limit; return a large sentinel (fully informative).
        return 10_000
    n = 1
    while 2 * n * (n - 1) * b < 1.0:
        n += 1
    return n - 1


def cs_cutoffs(b: float, N: int) -> np.ndarray:
    """Crawford-Sobel partition cutoffs for ``N`` intervals.

    Returns the boundary points ``0 = a_0 < a_1 < ... < a_N = 1`` of the
    uniform-quadratic partition equilibrium with ``N`` intervals, satisfying
    the recurrence

        a_{i+1} = 2 a_i - a_{i-1} + 4 b.

    Args:
        b: sender bias.
        N: number of intervals (must satisfy ``N <= max_partition_count(b)``).

    Returns:
        Array of ``N + 1`` cutoffs in ``[0, 1]``.
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    a = np.zeros(N + 1)
    a[0] = 0.0
    if N == 1:
        a[1] = 1.0
        return a
    # Solve for a_1 from the terminal condition a_N = 1 via the recurrence.
    # The recurrence a_{i+1} = 2 a_i - a_{i-1} + 4b has the closed-form
    #   a_i = i * a_1 + 2 b i (i-1),
    # and a_N = 1 pins a_1 = (1 - 2 b N (N-1)) / N.
    a_1 = (1.0 - 2.0 * b * N * (N - 1)) / N
    a[1] = a_1
    for i in range(1, N):
        a[i + 1] = 2.0 * a[i] - a[i - 1] + 4.0 * b
    # Pin the endpoint exactly (numerical drift safeguard).
    a[N] = 1.0
    return a
