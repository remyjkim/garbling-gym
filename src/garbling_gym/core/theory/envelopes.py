# ABOUTME: Concave (cav) and quasiconcave (qcav) envelopes of a sender value function
# ABOUTME: 1-D (binary-state) implementations here; 2-D simplex in simplex2d.py.

import numpy as np

__all__ = ["cav", "qcav"]


def cav(xs: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """Least concave majorant of ``v`` sampled on ``xs`` (1-D).

    With full sender commitment (Kamenica & Gentzkow 2011), the achievable
    sender value at any belief is the concave envelope of the indirect value
    function.  This computes that envelope on a 1-D belief segment — the
    binary-state case, or the boundary of the 2-D simplex where one state has
    probability zero.

    Algorithm: monotone-chain upper convex hull of the lifted points
    ``{(x_i, v_i)}``.  Walking left to right, we maintain a stack of hull
    vertices; when a new point would make the last hull edge convex from below
    (i.e. the triple turns *left*, indicating the boundary is not concave), we
    pop.  The hull is then interpolated back onto the input grid.

    Args:
        xs: 1-D grid of belief coordinates, strictly increasing. ``xs[0]`` and
            ``xs[-1]`` are typically 0 and 1.
        vs: sender value at each ``xs[i]``.

    Returns:
        The concave envelope evaluated at each ``xs[i]`` — a majorant of ``vs``
        that is concave (discrete midpoint-concave) on the grid.
    """
    xs = np.asarray(xs, dtype=float)
    vs = np.asarray(vs, dtype=float)
    n = len(xs)
    if n <= 2:
        return vs.copy()

    # Monotone chain for the UPPER hull (least concave majorant).
    # For an upper hull we walk left-to-right and keep the chain turning RIGHT
    # (negative cross product), popping when the new point turns LEFT.
    hull: list = []  # list of (x, v)
    for i in range(n):
        p = (xs[i], vs[i])
        while len(hull) >= 2:
            a = hull[-2]
            b = hull[-1]
            # cross product of (b-a) x (p-a): >0 means p is to the LEFT of a->b
            # (a left turn). For an upper hull we pop on left turns.
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if cross >= -1e-12:
                hull.pop()
            else:
                break
        hull.append(p)

    hull_x = np.array([p[0] for p in hull])
    hull_v = np.array([p[1] for p in hull])

    # Interpolate the hull back onto the input grid. hull is piecewise linear
    # and concave, so np.interp (linear) is exact on each hull segment.
    return np.interp(xs, hull_x, hull_v)


def qcav(xs: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """Quasiconcave envelope of ``v`` sampled on ``xs`` (1-D).

    Without commitment but with transparent (state-independent) motives, the
    sender's achievable value is the quasiconcave envelope (Lipnowski & Ravid
    2020).  On a 1-D belief segment the quasiconcave envelope is the smallest
    quasiconcave majorant, which equals

        qcav v(x) = min(prefix_max(x), suffix_max(x)),

    i.e. at each point take the larger of the running maximum seen from the left
    and the running maximum seen from the right.  This is unimodal and a
    majorant of ``v``.

    Args:
        xs: 1-D grid of belief coordinates (strictly increasing).
        vs: sender value at each ``xs[i]``.

    Returns:
        The quasiconcave envelope evaluated at each ``xs[i]``.
    """
    vs = np.asarray(vs, dtype=float)
    n = len(vs)
    if n == 0:
        return vs.copy()

    # prefix_max[i] = max(vs[0..i]); suffix_max[i] = max(vs[i..n-1])
    prefix_max = np.maximum.accumulate(vs)
    suffix_max = np.maximum.accumulate(vs[::-1])[::-1]
    return np.minimum(prefix_max, suffix_max)
